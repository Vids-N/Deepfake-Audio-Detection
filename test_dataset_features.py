
import torch

from dataset_loader import load_audio_batch
from dual_stream_detector import (
    DualStreamFeatureExtractor,
    DualStreamDetector
)

print("Loading real audio samples...")

samples = load_audio_batch(row_group_index=0, batch_size=2)

extractor = DualStreamFeatureExtractor()
model = DualStreamDetector()

extractor.eval()
model.eval()

print("Feature extractor and model loaded successfully!")

with torch.no_grad():
    for sample in samples:
        audio = torch.tensor(sample["audio"], dtype=torch.float32)

        # Extract LFCC and Log-Mel features
        lfcc, mel = extractor(audio)

        # Classify the extracted features
        output = model(lfcc, mel)
        prediction = torch.argmax(output, dim=1).item()

        print("\nFile:", sample["path"])
        print("Actual label:", sample["label"])
        print("LFCC shape:", tuple(lfcc.shape))
        print("Log-Mel shape:", tuple(mel.shape))
        print("Model output shape:", tuple(output.shape))
        print(
            "Predicted label:",
            "spoof" if prediction == 1 else "bonafide"
        )

print("\nReal dataset feature test completed!")

