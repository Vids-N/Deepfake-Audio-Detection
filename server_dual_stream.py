
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from dual_stream_detector import (
    DualStreamDetector,
    DualStreamFeatureExtractor,
)


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spectratrust_dual_stream.pth"
SAMPLES_DIR = BASE_DIR / "samples"

SAMPLE_RATE = 16000
DEVICE = torch.device("cpu")

app = Flask(__name__)
CORS(app)

# Maximum upload size: 25 MB
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

model = None
feature_extractor = None


# ============================================================
# 2. LOAD THE TRAINED MODEL
# ============================================================

def load_model():
    global model, feature_extractor

    if model is not None:
        return model

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    feature_extractor = DualStreamFeatureExtractor(
        sample_rate=checkpoint.get("sample_rate", SAMPLE_RATE),
        n_lfcc=checkpoint.get("n_lfcc", 20),
        n_mels=checkpoint.get("n_mels", 128),
        target_frames=checkpoint.get("target_frames", 126),
    ).to(DEVICE)

    feature_extractor.eval()

    model = DualStreamDetector().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("SpectraTrust dual-stream model loaded successfully.")

    return model


# ============================================================
# 3. WEBSITE ROUTES
# ============================================================

@app.route("/")
@app.route("/index.html")
def home():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/app.js")
def javascript():
    return send_from_directory(str(BASE_DIR), "app.js")


@app.route("/style.css")
def stylesheet():
    return send_from_directory(str(BASE_DIR), "style.css")


# Serve real_voice.wav and ai_cloned.wav from the samples folder.
@app.route("/samples/<path:filename>")
def serve_sample(filename):
    if not SAMPLES_DIR.is_dir():
        return jsonify({
            "success": False,
            "error": "The samples folder does not exist.",
        }), 404

    return send_from_directory(str(SAMPLES_DIR), filename)


# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    try:
        load_model()

        return jsonify({
            "status": "ready",
            "model": "SpectraTrust Dual-Stream LFCC + Log-Mel",
            "device": str(DEVICE),
        })

    except Exception as exc:
        app.logger.exception("Health check failed")

        return jsonify({
            "status": "error",
            "error": str(exc),
        }), 500


# ============================================================
# 5. AUDIO DEEPFAKE DETECTION
# ============================================================

@app.route("/api/detect", methods=["POST"])
def detect():
    audio_file = request.files.get("audio")

    if audio_file is None or not audio_file.filename:
        return jsonify({
            "success": False,
            "error": "Please upload an audio file.",
        }), 400

    temp_path = None

    try:
        loaded_model = load_model()

        
        suffix = Path(audio_file.filename or "recording.webm").suffix.lower()
        if not suffix:
            suffix = ".webm"

        temp_path = None
        wav_path = None

        try:
            # Save the browser recording to a temporary file.
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:
                temp_path = temp_file.name

            audio_file.save(temp_path)

            # Convert the recording to a standard WAV file.
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav",
            ) as wav_file:
                wav_path = wav_file.name

            import subprocess

            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", temp_path,
                    "-vn",
                    "-ac", "1",
                    "-ar", str(SAMPLE_RATE),
                    "-c:a", "pcm_s16le",
                    wav_path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    "Audio conversion failed: " + result.stderr[-1500:]
                )

            # Load the converted WAV for model prediction.
            audio, sample_rate = librosa.load(
                wav_path,
                sr=SAMPLE_RATE,
                mono=True,
            )

        finally:
            # Remove temporary files even if conversion fails.
            for path in (temp_path, wav_path):
                if path and os.path.exists(path):
                    os.remove(path)


        if audio.size == 0:
            raise ValueError("The uploaded audio file is empty.")

        if len(audio) < 1024:
            raise ValueError(
                "Audio is too short. Please upload a longer clip."
            )

        # Convert the audio to a PyTorch tensor.
        waveform = torch.from_numpy(
            np.asarray(audio, dtype=np.float32)
        ).unsqueeze(0).to(DEVICE)

        # Extract LFCC and Log-Mel features and run inference.
        with torch.no_grad():
            lfcc, mel = feature_extractor(waveform)
            logits = loaded_model(lfcc, mel)
            probabilities = torch.softmax(logits, dim=1)[0]

        predicted_label = int(
            torch.argmax(probabilities).item()
        )

        confidence = float(
            probabilities[predicted_label].item() * 100
        )

        labels = {
            0: "Real Audio",
            1: "Fake Audio",
        }

        is_fake = predicted_label == 1

        return jsonify({
            "success": True,
            "prediction": labels[predicted_label],
            "is_fake": is_fake,
            "confidence": round(confidence, 1),
            "confidence_note": (
                "Model score, not a guarantee of authenticity."
            ),
            "duration_seconds": round(
                len(audio) / sample_rate, 2
            ),
            "sample_rate": int(sample_rate),
            "features": {
                "lfcc_shape": list(lfcc.shape),
                "log_mel_shape": list(mel.shape),
            },
            "notes": (
                "The model classified this clip as likely synthetic."
                if is_fake
                else "The model classified this clip as likely genuine."
            ),
        })

    except Exception as exc:
        app.logger.exception("Audio detection failed")

        return jsonify({
            "success": False,
            "error": str(exc),
        }), 500

    finally:
        # Remove the temporary uploaded file.
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                app.logger.warning(
                    "Could not remove temporary upload: %s",
                    temp_path,
                )


# ============================================================
# 6. ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({
        "success": False,
        "error": "File is too large. Maximum upload size is 25 MB.",
    }), 413


# ============================================================
# 7. START THE SERVER
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Starting SpectraTrust Dual-Stream API")
    print("=" * 55)
    print(f"Project folder: {BASE_DIR}")
    print(f"Model file:     {MODEL_PATH}")
    print(f"Samples folder: {SAMPLES_DIR}")
    print(f"Device:         {DEVICE}")

    if not SAMPLES_DIR.is_dir():
        print("WARNING: The samples folder was not found.")

    try:
        load_model()
    except Exception as exc:
        print(f"MODEL LOADING ERROR: {exc}")
        raise

    print("Website: http://127.0.0.1:5000")
    print("Health:  http://127.0.0.1:5000/api/health")
    print("Ready to receive audio.")
    print("=" * 55)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )

