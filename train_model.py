
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from dual_stream_detector import (
    DualStreamFeatureExtractor,
    DualStreamDetector,
)

# --------------------------------------------------
# 1. Project settings
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

PARQUET_PATH = Path(
    r"G:\My Drive\Deepfake_Audio_Detection"
    r"\ASVspoof2019_LA\test_dataset.parquet"
)

TRAIN_CSV = PROJECT_DIR / "train_paths.csv"
TEST_CSV = PROJECT_DIR / "test_paths.csv"

CACHE_PATH = PROJECT_DIR / "prepared_features.pt"
MODEL_PATH = PROJECT_DIR / "spectratrust_dual_stream.pth"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cpu")

# --------------------------------------------------
# 2. Check files
# --------------------------------------------------

for file_path in [PARQUET_PATH, TRAIN_CSV, TEST_CSV]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

print("=" * 60)
print("SPECTRATRUST - DUAL-STREAM MODEL TRAINING")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

required_columns = {"path", "label"}

if not required_columns.issubset(train_df.columns):
    raise ValueError("train_paths.csv must contain path and label columns.")

if not required_columns.issubset(test_df.columns):
    raise ValueError("test_paths.csv must contain path and label columns.")

# Check for overlapping audio paths.
if set(train_df["path"]) & set(test_df["path"]):
    raise ValueError("Training and testing paths overlap.")

print("Training samples requested:", len(train_df))
print("Testing samples requested:", len(test_df))

# --------------------------------------------------
# 3. Prepare feature extractor
# --------------------------------------------------

extractor = DualStreamFeatureExtractor().to(DEVICE)
extractor.eval()

# --------------------------------------------------
# 4. Load cached features if available
# --------------------------------------------------

if CACHE_PATH.exists():
    print("\nLoading previously prepared features...")
    cached = torch.load(
        CACHE_PATH,
        map_location="cpu",
        weights_only=True,
    )

    train_lfcc = cached["train_lfcc"]
    train_mel = cached["train_mel"]
    train_labels = cached["train_labels"]

    test_lfcc = cached["test_lfcc"]
    test_mel = cached["test_mel"]
    test_labels = cached["test_labels"]

    print("Feature cache loaded.")

else:
    print("\nPreparing features from the Parquet dataset.")
    print("This first pass may take some time.")
    print("Audio files will not be extracted to disk.")

    train_lookup = dict(
        zip(train_df["path"], train_df["label"].astype(int))
    )
    test_lookup = dict(
        zip(test_df["path"], test_df["label"].astype(int))
    )

    all_lookup = {**train_lookup, **test_lookup}

    if len(all_lookup) != len(train_lookup) + len(test_lookup):
        raise ValueError("Duplicate paths found between CSV files.")

    train_features_lfcc = []
    train_features_mel = []
    train_labels_list = []

    test_features_lfcc = []
    test_features_mel = []
    test_labels_list = []

    found_paths = set()
    parquet_file = pq.ParquetFile(str(PARQUET_PATH))

    total_row_groups = parquet_file.num_row_groups
    start_time = time.time()

    with torch.no_grad():
        for group_index in range(total_row_groups):
            batches = parquet_file.iter_batches(
                batch_size=32,
                row_groups=[group_index],
                columns=["path", "audio", "label"],
            )

            for batch in batches:
                for row in batch.to_pylist():
                    path = row["path"]

                    if path not in all_lookup:
                        continue

                    if path in found_paths:
                        raise ValueError(
                            f"Duplicate path found in Parquet: {path}"
                        )

                    actual_label = int(row["label"])
                    expected_label = int(all_lookup[path])

                    if actual_label != expected_label:
                        raise ValueError(
                            f"Label mismatch for {path}: "
                            f"CSV={expected_label}, Parquet={actual_label}"
                        )

                    audio_bytes = row["audio"]["bytes"]

                    audio, sample_rate = sf.read(
                        io.BytesIO(audio_bytes),
                        dtype="float32",
                    )

                    if audio.ndim == 2:
                        audio = audio.mean(axis=1)

                    if sample_rate != 16000:
                        raise ValueError(
                            f"Unexpected sample rate {sample_rate} for {path}"
                        )

                    audio_tensor = torch.from_numpy(
                        np.asarray(audio, dtype=np.float32)
                    )

                    # Avoid reflect-padding errors on extremely short audio.
                    if audio_tensor.numel() < 1024:
                        audio_tensor = torch.nn.functional.pad(
                            audio_tensor,
                            (0, 1024 - audio_tensor.numel()),
                        )

                    lfcc, mel = extractor(audio_tensor)

                    if not torch.isfinite(lfcc).all():
                        raise ValueError(f"Invalid LFCC features for {path}")

                    if not torch.isfinite(mel).all():
                        raise ValueError(f"Invalid Log-Mel features for {path}")

                    label = actual_label

                    if path in train_lookup:
                        train_features_lfcc.append(lfcc.squeeze(0).cpu())
                        train_features_mel.append(mel.squeeze(0).cpu())
                        train_labels_list.append(label)
                    else:
                        test_features_lfcc.append(lfcc.squeeze(0).cpu())
                        test_features_mel.append(mel.squeeze(0).cpu())
                        test_labels_list.append(label)

                    found_paths.add(path)

            print(
                f"Scanned row group {group_index + 1}/{total_row_groups} "
                f"| selected samples found: {len(found_paths)}/"
                f"{len(all_lookup)}"
            )

    missing_paths = set(all_lookup) - found_paths

    if missing_paths:
        raise ValueError(
            f"{len(missing_paths)} selected paths were not found. "
            f"Example: {next(iter(missing_paths))}"
        )

    if not train_labels_list or not test_labels_list:
        raise ValueError("No training or testing features were generated.")

    train_lfcc = torch.stack(train_features_lfcc).float()
    train_mel = torch.stack(train_features_mel).float()
    train_labels = torch.tensor(train_labels_list, dtype=torch.long)

    test_lfcc = torch.stack(test_features_lfcc).float()
    test_mel = torch.stack(test_features_mel).float()
    test_labels = torch.tensor(test_labels_list, dtype=torch.long)

    torch.save(
        {
            "train_lfcc": train_lfcc,
            "train_mel": train_mel,
            "train_labels": train_labels,
            "test_lfcc": test_lfcc,
            "test_mel": test_mel,
            "test_labels": test_labels,
        },
        CACHE_PATH,
    )

    print("\nFeature cache saved to:", CACHE_PATH)
    print("Feature preparation time:", round(time.time() - start_time, 1), "seconds")

# --------------------------------------------------
# 5. Validate feature shapes and labels
# --------------------------------------------------

print("\nTraining LFCC shape:", tuple(train_lfcc.shape))
print("Training Log-Mel shape:", tuple(train_mel.shape))
print("Testing LFCC shape:", tuple(test_lfcc.shape))
print("Testing Log-Mel shape:", tuple(test_mel.shape))

if train_lfcc.shape[1] != 120:
    raise ValueError("Expected 120 LFCC features per sample.")

if train_mel.shape[1:] != (1, 128, 126):
    raise ValueError("Unexpected Log-Mel feature dimensions.")

if set(train_labels.tolist()) != {0, 1}:
    raise ValueError("Training set must contain both classes.")

if set(test_labels.tolist()) != {0, 1}:
    raise ValueError("Testing set must contain both classes.")

train_dataset = TensorDataset(
    train_lfcc,
    train_mel,
    train_labels,
)

test_dataset = TensorDataset(
    test_lfcc,
    test_mel,
    test_labels,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
)

# --------------------------------------------------
# 6. Create model and optimizer
# --------------------------------------------------

model = DualStreamDetector().to(DEVICE)

# Balanced training samples mean ordinary cross-entropy is suitable.
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# --------------------------------------------------
# 7. Train the model
# --------------------------------------------------

print("\nStarting training...")
print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for lfcc_batch, mel_batch, labels_batch in train_loader:
        lfcc_batch = lfcc_batch.to(DEVICE)
        mel_batch = mel_batch.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(lfcc_batch, mel_batch)
        loss = criterion(outputs, labels_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels_batch.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels_batch).sum().item()
        total += labels_batch.size(0)

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Loss: {epoch_loss:.4f} "
        f"| Training accuracy: {epoch_accuracy:.4f}"
    )

# --------------------------------------------------
# 8. Evaluate on the held-out test split
# --------------------------------------------------

print("\nEvaluating model...")
model.eval()

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
)

all_predictions = []
all_labels = []

with torch.no_grad():
    for lfcc_batch, mel_batch, labels_batch in test_loader:
        outputs = model(
            lfcc_batch.to(DEVICE),
            mel_batch.to(DEVICE),
        )

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels_batch.tolist())

accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(
    all_labels, all_predictions, zero_division=0
)
recall = recall_score(
    all_labels, all_predictions, zero_division=0
)
f1 = f1_score(
    all_labels, all_predictions, zero_division=0
)

print("\n" + "=" * 60)
print("EXPERIMENTAL TEST RESULTS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1],
        target_names=["bonafide", "spoof"],
        zero_division=0,
    )
)

print("Confusion matrix (rows=actual, columns=predicted):")
print(
    confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )
)

# --------------------------------------------------
# 9. Save model checkpoint
# --------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "sample_rate": 16000,
        "n_lfcc": 20,
        "n_mels": 128,
        "target_frames": 126,
        "labels": {"bonafide": 0, "spoof": 1},
        "epochs": EPOCHS,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
    },
    MODEL_PATH,
)

print("\nModel saved to:", MODEL_PATH)
print("Feature cache saved to:", CACHE_PATH)
print("\nTraining and evaluation completed.")