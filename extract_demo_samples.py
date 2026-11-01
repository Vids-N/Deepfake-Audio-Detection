
import os
import io
import pandas as pd
import soundfile as sf

DATASET_PATH = r"G:\My Drive\Deepfake_Audio_Detection\ASVspoof2019_LA\test_dataset.parquet"
OUTPUT_DIR = "samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Opening dataset metadata...", flush=True)

parquet_file = pd.read_parquet  # Not used to load the full dataset
import pyarrow.parquet as pq

parquet = pq.ParquetFile(DATASET_PATH)
print("Dataset opened.", flush=True)
print("Searching for two labelled samples...", flush=True)

real_saved = False
fake_saved = False

for batch in parquet.iter_batches(
    batch_size=100,
    columns=["path", "audio", "label"]
):
    for row in batch.to_pylist():
        label = str(row["label"]).lower()
        audio_value = row["audio"]

        if not isinstance(audio_value, dict):
            continue

        audio_bytes = audio_value.get("bytes")
        if not audio_bytes:
            continue

        if label in ("bonafide", "0", "real") and not real_saved:
            output_path = os.path.join(OUTPUT_DIR, "real_voice.wav")
        elif label in ("spoof", "1", "fake") and not fake_saved:
            output_path = os.path.join(OUTPUT_DIR, "ai_cloned.wav")
        else:
            continue

        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            sf.write(output_path, audio, sr)
            print(f"Saved {output_path}", flush=True)

            if "real_voice" in output_path:
                real_saved = True
            else:
                fake_saved = True

        except Exception as error:
            print("Skipping unreadable audio:", error, flush=True)

        if real_saved and fake_saved:
            break

    if real_saved and fake_saved:
        break

if not real_saved or not fake_saved:
    print(
        "Could not extract both samples. "
        f"Real found: {real_saved}; Spoof found: {fake_saved}",
        flush=True
    )
else:
    print("Both samples extracted successfully.", flush=True)
