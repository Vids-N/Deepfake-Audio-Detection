
import json
import io

import pyarrow.parquet as pq
import soundfile as sf
import numpy as np

DATASET_PATH = (
    r"G:\My Drive\Deepfake_Audio_Detection"
    r"\ASVspoof2019_LA\test_dataset.parquet"
)


def load_audio_batch(row_group_index=0, batch_size=8):
    parquet_file = pq.ParquetFile(DATASET_PATH)

    if row_group_index >= parquet_file.num_row_groups:
        raise IndexError("Row group index is out of range.")

    # Read only a small batch instead of the entire row group.
    batches = parquet_file.iter_batches(
        batch_size=batch_size,
        row_groups=[row_group_index],
        columns=["path", "audio", "label", "notes"]
    )

    batch = next(iter(batches))
    samples = []

    for row in batch.to_pylist():
        audio_bytes = row["audio"]["bytes"]

        audio, sample_rate = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32"
        )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        notes = json.loads(row["notes"]) if row["notes"] else {}

        samples.append({
            "path": row["path"],
            "audio": audio,
            "sample_rate": sample_rate,
            "label": row["label"],
            "notes": notes
        })

    return samples


if __name__ == "__main__":
    print("Loading a small audio batch...")

    samples = load_audio_batch(row_group_index=0, batch_size=8)

    print("Samples loaded:", len(samples))

    for sample in samples:
        duration = len(sample["audio"]) / sample["sample_rate"]

        print(
            sample["path"],
            "| Label:", sample["label"],
            "| Rate:", sample["sample_rate"],
            "| Duration:", round(duration, 2), "seconds"
        )

    print("\nData loader test completed.")
