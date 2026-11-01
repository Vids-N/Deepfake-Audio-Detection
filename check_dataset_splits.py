
import json
import pyarrow.parquet as pq

DATASET_PATH = (
    r"G:\My Drive\Deepfake_Audio_Detection"
    r"\ASVspoof2019_LA\test_dataset.parquet"
)

parquet_file = pq.ParquetFile(DATASET_PATH)
split_counts = {}
label_counts_by_split = {}

print("Checking dataset splits...")

for i in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(
        i,
        columns=["label", "notes"]
    )

    for label, notes in zip(
        table["label"].to_pylist(),
        table["notes"].to_pylist()
    ):
        metadata = json.loads(notes) if notes else {}
        split = metadata.get("subset", "unknown")

        split_counts[split] = split_counts.get(split, 0) + 1

        if split not in label_counts_by_split:
            label_counts_by_split[split] = {}

        label_counts_by_split[split][label] = (
            label_counts_by_split[split].get(label, 0) + 1
        )

print("\nSamples per subset:")
for split, count in split_counts.items():
    print(f"{split}: {count}")

print("\nLabels per subset:")
for split, counts in label_counts_by_split.items():
    print(f"{split}: {counts}")
