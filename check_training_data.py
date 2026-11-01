
from collections import Counter
import pyarrow.parquet as pq

DATASET_PATH = (
    r"G:\My Drive\Deepfake_Audio_Detection"
    r"\ASVspoof2019_LA\test_dataset.parquet"
)

print("Counting labels across the full dataset...")

parquet_file = pq.ParquetFile(DATASET_PATH)
counts = Counter()

for i in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(i, columns=["label"])
    counts.update(table.column("label").to_pylist())

    print(
        f"Processed row group {i + 1}/"
        f"{parquet_file.num_row_groups}"
    )

total = sum(counts.values())

print("\nFull dataset label counts:")
print("Bonafide (0):", counts[0])
print("Spoof (1):", counts[1])
print("Total:", total)

print("\nClass percentages:")
print(f"Bonafide: {counts[0] / total * 100:.2f}%")
print(f"Spoof:    {counts[1] / total * 100:.2f}%")
