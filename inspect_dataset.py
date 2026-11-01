
import io
import pyarrow.parquet as pq
import soundfile as sf

DATASET_PATH = (
    r"G:\My Drive\Deepfake_Audio_Detection"
    r"\ASVspoof2019_LA\test_dataset.parquet"
)

print("Reading dataset metadata...")
parquet_file = pq.ParquetFile(DATASET_PATH)

print("Total rows:", parquet_file.metadata.num_rows)
print("Row groups:", parquet_file.num_row_groups)

# Step 1: Count labels without loading the audio data
print("\nReading labels from row groups...")
label_counts = {}

for i in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(i, columns=["label"])
    labels = table["label"].to_pylist()

    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

print("Label counts:", label_counts)

# Step 2: Read and decode one audio sample directly from Parquet
print("\nReading one audio sample...")
table = parquet_file.read_row_group(0, columns=["path", "audio", "label", "notes"])
row = table.slice(0, 1).to_pylist()[0]

audio_bytes = row["audio"]["bytes"]

audio_data, sample_rate = sf.read(
    io.BytesIO(audio_bytes),
    dtype="float32"
)

print("Audio path:", row["path"])
print("Sample rate:", sample_rate)
print("Audio shape:", audio_data.shape)
print("Duration (seconds):", round(len(audio_data) / sample_rate, 2))
print("Label:", row["label"])
print("Notes:", row["notes"])

print("\nInspection completed.")

