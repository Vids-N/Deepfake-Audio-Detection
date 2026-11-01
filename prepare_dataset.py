
import csv
import random
from collections import defaultdict

import pyarrow.parquet as pq

# Dataset location
DATASET_PATH = (
    r"G:\My Drive\Deepfake_Audio_Detection"
    r"\ASVspoof2019_LA\test_dataset.parquet"
)

# Settings
SAMPLES_PER_CLASS = 1000
RANDOM_SEED = 42

print("Scanning dataset labels...")

parquet_file = pq.ParquetFile(DATASET_PATH)

# Store paths separately for each class
class_rows = defaultdict(list)

# Store the actual label for every audio path
label_lookup = {}

# Read only paths and labels, not audio bytes
for group_index in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(
        group_index,
        columns=["path", "label"]
    )

    paths = table.column("path").to_pylist()
    labels = table.column("label").to_pylist()

    for path, label in zip(paths, labels):
        if label in (0, 1):
            class_rows[label].append(path)
            label_lookup[path] = label

    print(
        f"Scanned row group {group_index + 1}/"
        f"{parquet_file.num_row_groups}"
    )

# Check that both classes have enough samples
if len(class_rows[0]) < SAMPLES_PER_CLASS:
    raise ValueError("Not enough bonafide samples.")

if len(class_rows[1]) < SAMPLES_PER_CLASS:
    raise ValueError("Not enough spoof samples.")

# Select an equal number from each class
random.seed(RANDOM_SEED)

bonafide = random.sample(
    class_rows[0],
    SAMPLES_PER_CLASS
)

spoof = random.sample(
    class_rows[1],
    SAMPLES_PER_CLASS
)

# Shuffle each class before splitting
random.shuffle(bonafide)
random.shuffle(spoof)


# Split each class into 80% training and 20% testing
def split_class(paths):
    split_index = int(len(paths) * 0.8)
    return paths[:split_index], paths[split_index:]


train_bonafide, test_bonafide = split_class(bonafide)
train_spoof, test_spoof = split_class(spoof)

# Combine both classes
train_paths = train_bonafide + train_spoof
test_paths = test_bonafide + test_spoof

random.shuffle(train_paths)
random.shuffle(test_paths)


# Display dataset summary
print("\nDataset preparation summary")

print("Training samples:", len(train_paths))
print("  Bonafide:", len(train_bonafide))
print("  Spoof:", len(train_spoof))

print("Testing samples:", len(test_paths))
print("  Bonafide:", len(test_bonafide))
print("  Spoof:", len(test_spoof))


# Save paths and labels to CSV
def save_paths(filename, paths):
    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["path", "label"])

        for path in paths:
            writer.writerow([path, label_lookup[path]])


save_paths("train_paths.csv", train_paths)
save_paths("test_paths.csv", test_paths)

print("\nSaved train_paths.csv and test_paths.csv with labels.")
print("Dataset selection completed!")
