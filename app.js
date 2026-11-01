
"use strict";

// ============================================================
// SpectraTrust - Deepfake Audio Detection
// ============================================================

let audioContext = null;
let currentAudioBuffer = null;
let currentAudioBlob = null;
let currentAudioFileName = "";
let currentAudioObjectURL = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordTimerInterval = null;
let currentAnalysisResult = null;

const API_BASE_URL = "";

// ============================================================
// DOM ELEMENTS
// ============================================================

const btnRecord = document.getElementById("btn-record");
const recTimer = document.getElementById("rec-timer");
const recStatus = document.getElementById("rec-status-text");

const fileInput = document.getElementById("file-input");
const btnBrowse = document.getElementById("btn-browse");
const dropZone = document.getElementById("drop-zone");
const selectedFileName = document.getElementById("selected-file-name");

const playerWrapper = document.getElementById("player-wrapper");
const audioPlayback = document.getElementById("audio-playback");

const btnAnalyze = document.getElementById("btn-analyze");
const btnDemoReal = document.getElementById("btn-demo-real");
const btnDemoFake = document.getElementById("btn-demo-fake");
const btnExport = document.getElementById("btn-export");
const btnReset = document.getElementById("btn-reset");

const scanningCard = document.getElementById("scanning-card");
const scanProgress = document.getElementById("scan-progress");
const scanLog = document.getElementById("scan-log");
const resultsSection = document.getElementById("results-section");

const waveformCanvas = document.getElementById("waveform-canvas");
const spectrogramCanvas = document.getElementById("spectrogram-canvas");

// ============================================================
// 1. AUDIO CONTEXT
// ============================================================

function getAudioContext() {
  if (!audioContext) {
    const AudioContextClass =
      window.AudioContext || window.webkitAudioContext;

    if (!AudioContextClass) {
      throw new Error("AudioContext is not supported in this browser.");
    }

    audioContext = new AudioContextClass();
  }

  return audioContext;
}

// ============================================================
// 2. LOAD AUDIO
// ============================================================

async function loadAudioFromBlob(blob, fileName) {
  if (!blob || blob.size === 0) {
    throw new Error("The selected audio file is empty.");
  }

  currentAudioBlob = blob;
  currentAudioFileName = fileName;
  currentAnalysisResult = null;
  currentAudioBuffer = null;

  if (selectedFileName) {
    selectedFileName.textContent = fileName;
  }

  playerWrapper?.classList.remove("hidden");
  resultsSection?.classList.add("hidden");

  if (currentAudioObjectURL) {
    URL.revokeObjectURL(currentAudioObjectURL);
  }

  currentAudioObjectURL = URL.createObjectURL(blob);

  if (audioPlayback) {
    audioPlayback.src = currentAudioObjectURL;
    audioPlayback.load();
  }

  try {
    const audioData = await blob.arrayBuffer();

    currentAudioBuffer = await getAudioContext().decodeAudioData(
      audioData.slice(0)
    );
  } catch (error) {
    console.warn(
      "Audio loaded, but waveform decoding is unavailable:",
      error
    );
  }

  if (btnAnalyze) {
    btnAnalyze.disabled = false;
  }
}

// ============================================================
// 3. LOAD DEMO SAMPLES
// ============================================================

async function loadSample(path, fileName) {
  try {
    if (btnDemoReal) btnDemoReal.disabled = true;
    if (btnDemoFake) btnDemoFake.disabled = true;

    const response = await fetch(path);

    if (!response.ok) {
      throw new Error(
        `Could not load ${fileName}. Check the samples folder and Flask route.`
      );
    }

    const blob = await response.blob();

    await loadAudioFromBlob(blob, fileName);
  } catch (error) {
    console.error("Sample loading error:", error);
    alert(error.message);
  } finally {
    if (btnDemoReal) btnDemoReal.disabled = false;
    if (btnDemoFake) btnDemoFake.disabled = false;
  }
}

if (btnDemoReal) {
  btnDemoReal.addEventListener("click", () => {
    loadSample("/samples/real_voice.wav", "real_voice.wav");
  });
}

if (btnDemoFake) {
  btnDemoFake.addEventListener("click", () => {
    loadSample("/samples/ai_cloned.wav", "ai_cloned.wav");
  });
}

// ============================================================
// 4. FILE UPLOAD
// ============================================================

if (btnBrowse && fileInput) {
  btnBrowse.addEventListener("click", () => {
    fileInput.click();
  });
}

if (fileInput) {
  fileInput.addEventListener("change", async (event) => {
    const file = event.target.files?.[0];

    if (file) {
      await handleFile(file);
    }

    fileInput.value = "";
  });
}

async function handleFile(file) {
  const validExtension =
    /\.(wav|mp3|m4a|ogg|webm|flac)$/i.test(file.name);

  if (!file.type.startsWith("audio/") && !validExtension) {
    alert("Please select a supported audio file.");
    return;
  }

  try {
    await loadAudioFromBlob(file, file.name);
  } catch (error) {
    alert(error.message);
  }
}

// ============================================================
// 5. DRAG AND DROP
// ============================================================

if (dropZone) {
  dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.style.borderColor = "var(--accent-cyan)";
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.style.borderColor = "var(--border-color)";
  });

  dropZone.addEventListener("drop", async (event) => {
    event.preventDefault();
    dropZone.style.borderColor = "var(--border-color)";

    const file = event.dataTransfer.files?.[0];

    if (file) {
      await handleFile(file);
    }
  });
}

// ============================================================
// 6. MICROPHONE RECORDING - 5 SECONDS
// ============================================================

if (btnRecord) {
  btnRecord.addEventListener("click", async () => {
    if (mediaRecorder?.state === "recording") {
      return;
    }

    if (
      !navigator.mediaDevices?.getUserMedia ||
      !window.MediaRecorder
    ) {
      alert("Microphone recording is not supported in this browser.");
      return;
    }

    let stream;

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: true
      });

      let mimeType = "";

      if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
        mimeType = "audio/webm;codecs=opus";
      } else if (MediaRecorder.isTypeSupported("audio/webm")) {
        mimeType = "audio/webm";
      }

      mediaRecorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined
      );

      recordedChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        try {
          const blob = new Blob(recordedChunks, {
            type: mediaRecorder.mimeType || "audio/webm"
          });

          const extension = blob.type.includes("ogg")
            ? "ogg"
            : blob.type.includes("webm")
              ? "webm"
              : "audio";

          await loadAudioFromBlob(
            blob,
            `Microphone_Capture_5s.${extension}`
          );

          if (recStatus) {
            recStatus.textContent =
              "Recording complete. You can now analyze the audio.";
          }
        } catch (error) {
          alert(error.message);
        } finally {
          stream.getTracks().forEach((track) => track.stop());

          btnRecord.disabled = false;
          btnRecord.textContent = "Record Again";
        }
      };

      mediaRecorder.onerror = (event) => {
        console.error("Recording error:", event.error);
      };

      mediaRecorder.start();

      btnRecord.disabled = true;
      btnRecord.textContent = "Recording...";

      if (recStatus) {
        recStatus.textContent = "Recording for 5 seconds...";
      }

      let timeLeft = 5;

      if (recTimer) {
        recTimer.textContent = "05.00s";
      }

      if (recordTimerInterval) {
        clearInterval(recordTimerInterval);
      }

      recordTimerInterval = setInterval(() => {
        timeLeft -= 0.05;

        if (timeLeft <= 0) {
          clearInterval(recordTimerInterval);
          recordTimerInterval = null;

          if (recTimer) {
            recTimer.textContent = "00.00s";
          }

          if (mediaRecorder?.state === "recording") {
            mediaRecorder.stop();
          }
        } else if (recTimer) {
          recTimer.textContent = timeLeft.toFixed(2) + "s";
        }
      }, 50);
    } catch (error) {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }

      console.error("Microphone error:", error);

      alert(
        "Unable to access the microphone. Check browser permissions and try again."
      );

      btnRecord.disabled = false;
      btnRecord.textContent = "Start 5s Recording";
    }
  });
}

// ============================================================
// 7. CALL PYTHON DETECTION API
// ============================================================

if (btnAnalyze) {
  btnAnalyze.addEventListener("click", async () => {
    if (!currentAudioBlob) {
      alert("Please upload, record, or select an audio sample first.");
      return;
    }

    btnAnalyze.disabled = true;

    scanningCard?.classList.remove("hidden");
    resultsSection?.classList.add("hidden");

    if (scanProgress) scanProgress.style.width = "10%";
    if (scanLog) scanLog.textContent = "Connecting to the model...";

    const progressSteps = [
      [25, "Uploading audio..."],
      [45, "Preparing audio features..."],
      [70, "Running the detection model..."],
      [90, "Preparing forensic measurements..."]
    ];

    let stepIndex = 0;

    const progressTimer = setInterval(() => {
      if (stepIndex < progressSteps.length) {
        if (scanProgress) {
          scanProgress.style.width =
            progressSteps[stepIndex][0] + "%";
        }

        if (scanLog) {
          scanLog.textContent = progressSteps[stepIndex][1];
        }

        stepIndex++;
      }
    }, 600);

    try {
      const formData = new FormData();

      formData.append(
        "audio",
        currentAudioBlob,
        currentAudioFileName || "uploaded_audio.wav"
      );

      const response = await fetch(
        `${API_BASE_URL}/api/detect`,
        {
          method: "POST",
          body: formData
        }
      );

      let result;

      try {
        result = await response.json();
      } catch {
        throw new Error("The server returned an invalid response.");
      }

      if (!response.ok || result.success === false) {
        throw new Error(
          result.error ||
          `Detection request failed: HTTP ${response.status}`
        );
      }

      if (scanProgress) scanProgress.style.width = "100%";
      if (scanLog) scanLog.textContent = "Analysis completed.";

      currentAnalysisResult = result;

      renderResults(result);
    } catch (error) {
      console.error("Detection error:", error);

      alert(
        "Audio analysis failed.\n\n" +
        error.message +
        "\n\nCheck that server_dual_stream.py is running."
      );
    } finally {
      clearInterval(progressTimer);

      btnAnalyze.disabled = false;

      setTimeout(() => {
        scanningCard?.classList.add("hidden");
      }, 500);
    }
  });
}

// ============================================================
// 8. DISPLAY MODEL RESULTS
// ============================================================

function renderResults(result) {
  resultsSection?.classList.remove("hidden");

  const prediction = String(
    result.prediction ?? result.label ?? ""
  ).toLowerCase();

  const isFake =
    result.is_fake !== undefined
      ? Boolean(result.is_fake)
      : result.class !== undefined
        ? Number(result.class) === 1
        : prediction.includes("fake") ||
          prediction.includes("spoof");

  const badge = document.getElementById("verdict-badge");
  const valScore = document.getElementById("val-score");
  const barScore = document.getElementById("bar-score");

  const valSpectral = document.getElementById("val-spectral");
  const valPhase = document.getElementById("val-phase");
  const valNoise = document.getElementById("val-noise");

  const explText = document.getElementById("explanation-text");
  const fileMeta = document.getElementById("file-meta");

  // Prediction
  if (badge) {
    badge.textContent = isFake
      ? "MODEL PREDICTION: FAKE"
      : "MODEL PREDICTION: REAL";

    badge.className =
      "verdict-pill " + (isFake ? "verdict-fake" : "verdict-real");
  }

  // Confidence score
  let confidence = result.confidence;

  if (confidence !== null && confidence !== undefined) {
    confidence = Number(confidence);

    if (!Number.isFinite(confidence)) {
      confidence = null;
    } else if (confidence >= 0 && confidence <= 1) {
      confidence *= 100;
    }
  }

  if (valScore) {
    valScore.textContent =
      confidence == null
        ? "Score unavailable"
        : `${Math.max(0, Math.min(100, confidence)).toFixed(1)}% model score`;
  }

  if (barScore) {
    const score =
      confidence == null
        ? 0
        : Math.max(0, Math.min(100, confidence));

    barScore.style.width = `${score}%`;
    barScore.style.background = isFake
      ? "var(--color-fake)"
      : "var(--color-real)";
  }

  // ----------------------------------------------------------
  // FORENSIC ANALYSIS
  // Must match the JSON returned by server_dual_stream.py
  // ----------------------------------------------------------

  const analysis = result.forensic_analysis || {};

  // A. Spectral discontinuity
  const spectral = analysis.spectral_discontinuity || {};

  if (
    spectral.status === "calculated" &&
    spectral.upper_band_energy_percent != null
  ) {
    const energy = Number(spectral.upper_band_energy_percent);

    const rolloff = spectral.spectral_rolloff_hz;

    if (valSpectral) {
      valSpectral.textContent =
        `Upper-band energy: ${energy.toFixed(2)}%` +
        (
          rolloff != null
            ? ` | Roll-off: ${Number(rolloff).toFixed(1)} Hz`
            : ""
        );
    }
  } else if (valSpectral) {
    valSpectral.textContent =
      spectral.message || "Not available";
  }

  // B. Phase coherence
  const phase = analysis.phase_coherence || {};

  if (
    phase.status === "calculated" &&
    phase.phase_consistency != null
  ) {
    if (valPhase) {
      valPhase.textContent =
        `Phase consistency: ${
          Number(phase.phase_consistency).toFixed(2)
        }%`;
    }
  } else if (valPhase) {
    valPhase.textContent =
      phase.message || "Not available";
  }

  // C. Acoustic room noise
  const noise = analysis.acoustic_room_noise || {};

  if (
    noise.status === "calculated" &&
    noise.estimated_noise_floor_db != null
  ) {
    const noiseFloor = Number(noise.estimated_noise_floor_db);
    const silence = noise.near_silence_percent;

    if (valNoise) {
      valNoise.textContent =
        `Quiet-frame level: ${noiseFloor.toFixed(2)} dBFS` +
        (
          silence != null
            ? ` | Near-silence: ${Number(silence).toFixed(2)}%`
            : ""
        );
    }
  } else if (valNoise) {
    valNoise.textContent =
      noise.message || "Not available";
  }

  // File information and explanation
  if (explText) {
    explText.textContent =
      result.notes ||
      result.message ||
      "This result is a model estimate and is not proof of audio authenticity.";
  }

  if (fileMeta) {
    fileMeta.textContent =
      `File: ${currentAudioFileName} | ` +
      `Duration: ${result.duration_seconds ?? "Unknown"} seconds | ` +
      `Sample rate: ${result.sample_rate ?? "Unknown"} Hz`;
  }

  // Store a complete report snapshot
  currentAnalysisResult = {
    fileName: currentAudioFileName,
    prediction: result.prediction ?? (isFake ? "Fake Audio" : "Real Audio"),
    confidence,
    features: result.features || {},
    forensicAnalysis: analysis,
    duration: result.duration_seconds ?? "Unknown",
    sampleRate: result.sample_rate ?? "Unknown",
    timestamp: new Date().toLocaleString()
  };

  // Audio visualizations
  if (currentAudioBuffer) {
    drawWaveformHeatmap(
      currentAudioBuffer.getChannelData(0),
      isFake
    );

    drawSpectrogram(
      currentAudioBuffer.getChannelData(0),
      isFake
    );
  } else {
    clearCanvas(waveformCanvas, "Waveform unavailable");
    clearCanvas(spectrogramCanvas, "Spectrogram unavailable");
  }
}

// ============================================================
// 9. WAVEFORM VISUALIZATION
// ============================================================

function drawWaveformHeatmap(channelData, isFake) {
  if (!waveformCanvas) return;

  const canvas = waveformCanvas;
  const context = canvas.getContext("2d");

  const width = Math.max(
    1,
    canvas.parentElement?.clientWidth || 600
  );

  const height = 120;
  const dpr = window.devicePixelRatio || 1;

  canvas.width = width * dpr;
  canvas.height = height * dpr;

  canvas.style.width = width + "px";
  canvas.style.height = height + "px";

  context.setTransform(dpr, 0, 0, dpr, 0, 0);

  context.fillStyle = "#f1f5f9";
  context.fillRect(0, 0, width, height);

  const middle = height / 2;
  const step = Math.max(1, Math.ceil(channelData.length / width));

  context.strokeStyle = "#cbd5e1";
  context.beginPath();
  context.moveTo(0, middle);
  context.lineTo(width, middle);
  context.stroke();

  for (let x = 0; x < width; x++) {
    let minimum = 1;
    let maximum = -1;

    for (let j = 0; j < step; j++) {
      const value = channelData[x * step + j];

      if (value === undefined) break;

      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }

    const highlighted =
      isFake && x / width > 0.35 && x / width < 0.75;

    context.fillStyle = highlighted ? "#ef4444" : "#10b981";

    const barHeight = Math.max(
      2,
      (maximum - minimum) * height * 0.45
    );

    context.fillRect(
      x,
      middle - barHeight / 2,
      1.5,
      barHeight
    );
  }
}

// ============================================================
// 10. SPECTROGRAM-STYLE VISUALIZATION
// Illustrative visualization, not the model's raw spectrogram.
// ============================================================

function drawSpectrogram(channelData, isFake) {
  if (!spectrogramCanvas) return;

  const canvas = spectrogramCanvas;
  const context = canvas.getContext("2d");

  const width = Math.max(
    1,
    canvas.parentElement?.clientWidth || 600
  );

  const height = 130;
  const dpr = window.devicePixelRatio || 1;

  canvas.width = width * dpr;
  canvas.height = height * dpr;

  canvas.style.width = width + "px";
  canvas.style.height = height + "px";

  context.setTransform(dpr, 0, 0, dpr, 0, 0);

  context.fillStyle = "#111827";
  context.fillRect(0, 0, width, height);

  const columns = 90;
  const rows = 28;
  const columnWidth = width / columns;
  const rowHeight = height / rows;

  for (let column = 0; column < columns; column++) {
    const start = Math.floor(
      (column / columns) * channelData.length
    );

    const end = Math.max(
      start + 1,
      Math.floor(((column + 1) / columns) * channelData.length)
    );

    let energy = 0;
    let count = 0;

    for (let i = start; i < end; i += 4) {
      energy += channelData[i] ** 2;
      count++;
    }

    energy = Math.sqrt(energy / Math.max(1, count));

    const highlighted =
      isFake && column / columns > 0.35 && column / columns < 0.75;

    for (let row = 0; row < rows; row++) {
      const frequencyFactor = 1 - row / rows;
      const intensity = Math.min(
        1,
        energy * 8 * frequencyFactor
      );

      if (intensity < 0.12) {
        context.fillStyle = "#1f2937";
      } else if (highlighted) {
        context.fillStyle =
          intensity > 0.5 ? "#ef4444" : "#f59e0b";
      } else {
        context.fillStyle =
          intensity > 0.5 ? "#10b981" : "#06b6d4";
      }

      context.fillRect(
        column * columnWidth,
        height - (row + 1) * rowHeight,
        Math.max(1, columnWidth - 1),
        Math.max(1, rowHeight - 1)
      );
    }
  }
}

// ============================================================
// 11. CLEAR CANVAS
// ============================================================

function clearCanvas(canvas, message) {
  if (!canvas) return;

  const context = canvas.getContext("2d");

  const width = Math.max(
    1,
    canvas.parentElement?.clientWidth || 600
  );

  const height =
    canvas.id === "waveform-canvas" ? 120 : 130;

  const dpr = window.devicePixelRatio || 1;

  canvas.width = width * dpr;
  canvas.height = height * dpr;

  canvas.style.width = width + "px";
  canvas.style.height = height + "px";

  context.setTransform(dpr, 0, 0, dpr, 0, 0);

  context.fillStyle = "#f1f5f9";
  context.fillRect(0, 0, width, height);

  context.fillStyle = "#475569";
  context.font = "14px sans-serif";
  context.fillText(message, 12, height / 2);
}

// ============================================================
// 12. EXPORT REPORT
// ============================================================

if (btnExport) {
  btnExport.addEventListener("click", () => {
    if (!currentAnalysisResult) {
      alert("Analyze an audio file before exporting a report.");
      return;
    }

    const report = [
      "=====================================================",
      "SPECTRATRUST - DEEPFAKE AUDIO DETECTION REPORT",
      "=====================================================",
      `File: ${currentAnalysisResult.fileName}`,
      `Date: ${currentAnalysisResult.timestamp}`,
      `Prediction: ${currentAnalysisResult.prediction}`,
      `Model score: ${
        currentAnalysisResult.confidence == null
          ? "Unavailable"
          : currentAnalysisResult.confidence.toFixed(1) + "%"
      }`,
      `Duration: ${currentAnalysisResult.duration}`,
      `Sample rate: ${currentAnalysisResult.sampleRate}`,
      "",
      "FORENSIC MEASUREMENTS",
      JSON.stringify(
        currentAnalysisResult.forensicAnalysis || {},
        null,
        2
      ),
      "",
      "MODEL FEATURES",
      JSON.stringify(currentAnalysisResult.features || {}, null, 2),
      "",
      "IMPORTANT NOTE:",
      "The model prediction and supplementary signal measurements are estimates.",
      "They should not be treated as proof that an audio recording is real or fake.",
      "====================================================="
    ].join("\n");

    const blob = new Blob([report], {
      type: "text/plain;charset=utf-8"
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = `SpectraTrust_Report_${Date.now()}.txt`;

    document.body.appendChild(link);
    link.click();
    link.remove();

    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}

// ============================================================
// 13. RESET THE INTERFACE
// ============================================================

if (btnReset) {
  btnReset.addEventListener("click", () => {
    resultsSection?.classList.add("hidden");
    playerWrapper?.classList.add("hidden");
    scanningCard?.classList.add("hidden");

    if (audioPlayback) {
      audioPlayback.pause();
      audioPlayback.removeAttribute("src");
      audioPlayback.load();
    }

    if (selectedFileName) {
      selectedFileName.textContent = "No file selected";
    }

    currentAudioBlob = null;
    currentAudioBuffer = null;
    currentAudioFileName = "";
    currentAnalysisResult = null;

    if (currentAudioObjectURL) {
      URL.revokeObjectURL(currentAudioObjectURL);
      currentAudioObjectURL = null;
    }

    if (fileInput) fileInput.value = "";

    if (btnAnalyze) btnAnalyze.disabled = false;

    if (btnRecord) {
      btnRecord.disabled = false;
      btnRecord.textContent = "Start 5s Recording";
    }

    if (recTimer) recTimer.textContent = "05.00s";

    if (recStatus) {
      recStatus.textContent =
        "Press to start a timed 5-second voice capture.";
    }

    if (scanProgress) scanProgress.style.width = "0%";

    if (recordTimerInterval) {
      clearInterval(recordTimerInterval);
      recordTimerInterval = null;
    }

    clearCanvas(waveformCanvas, "Waveform unavailable");
    clearCanvas(spectrogramCanvas, "Spectrogram unavailable");
  });
}

// ============================================================
// 14. INITIALIZE THE PAGE
// ============================================================

clearCanvas(waveformCanvas, "Waveform will appear here");
clearCanvas(spectrogramCanvas, "Spectrogram will appear here");

console.log("SpectraTrust app.js loaded successfully.");
