import torch
from dual_stream_detector import DualStreamFeatureExtractor, DualStreamDetector


def main():
    extractor = DualStreamFeatureExtractor()
    model = DualStreamDetector()
    model.eval()

    # Dummy audio only: 2 samples, 4 seconds each at 16 kHz.
    audio = torch.randn(2, 16000 * 4)

    with torch.no_grad():
        lfcc, mel = extractor(audio)
        output = model(lfcc, mel)

    print("LFCC shape:", tuple(lfcc.shape))
    print("Log-Mel shape:", tuple(mel.shape))
    print("Output shape:", tuple(output.shape))
    print("Model test completed successfully!")


if __name__ == "__main__":
    main()
