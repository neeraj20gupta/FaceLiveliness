const preview = document.getElementById("preview");
const startBtn = document.getElementById("startBtn");
const statusEl = document.getElementById("status");
const resultsSection = document.getElementById("results");
const decisionEl = document.getElementById("decision");
const riskScoreEl = document.getElementById("riskScore");
const confidenceEl = document.getElementById("confidence");
const signalsList = document.getElementById("signalsList");
const reasonsList = document.getElementById("reasonsList");
const annotatedFrames = document.getElementById("annotatedFrames");
const captureCanvas = document.getElementById("captureCanvas");

const CAPTURE_DURATION_MS = 3000;
const TARGET_FPS = 12;

let stream;

async function initCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 } },
    audio: false,
  });
  preview.srcObject = stream;
}

function updateStatus(text) {
  statusEl.textContent = text;
}

function clearResults() {
  resultsSection.classList.add("hidden");
  signalsList.innerHTML = "";
  reasonsList.innerHTML = "";
  annotatedFrames.innerHTML = "";
}

function appendSignal(label, value) {
  const li = document.createElement("li");
  li.textContent = `${label}: ${value}`;
  signalsList.appendChild(li);
}

function appendReason(reason) {
  const li = document.createElement("li");
  li.textContent = reason;
  reasonsList.appendChild(li);
}

function renderAnnotatedFrames(frames) {
  if (!frames || frames.length === 0) {
    annotatedFrames.textContent = "No annotated frames returned.";
    return;
  }
  frames.forEach((frame) => {
    const img = document.createElement("img");
    img.src = `data:image/jpeg;base64,${frame.image_base64_jpeg}`;
    img.alt = `Annotated frame ${frame.frame_index}`;
    annotatedFrames.appendChild(img);
  });
}

async function captureFrames() {
  const frames = [];
  const { videoWidth: width, videoHeight: height } = preview;
  captureCanvas.width = width;
  captureCanvas.height = height;
  const ctx = captureCanvas.getContext("2d");

  const interval = 1000 / TARGET_FPS;
  const totalFrames = Math.round(CAPTURE_DURATION_MS / interval);

  for (let i = 0; i < totalFrames; i += 1) {
    ctx.drawImage(preview, 0, 0, width, height);
    const blob = await new Promise((resolve) =>
      captureCanvas.toBlob(resolve, "image/jpeg", 0.85)
    );
    if (blob) {
      frames.push(blob);
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  return { frames, width, height };
}

async function sendFrames({ frames, width, height }) {
  const formData = new FormData();
  frames.forEach((frame, index) => {
    formData.append("frames", frame, `frame_${index}.jpg`);
  });
  formData.append("fps", TARGET_FPS.toString());
  formData.append("width", width.toString());
  formData.append("height", height.toString());

  const response = await fetch("/api/liveness", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(errText || "Liveness request failed.");
  }

  return response.json();
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  clearResults();

  try {
    updateStatus("Capturing…");
    const payload = await captureFrames();

    updateStatus("Uploading…");
    const result = await sendFrames(payload);

    updateStatus("Processing…");
    decisionEl.textContent = result.liveness;
    riskScoreEl.textContent = result.score.toFixed(2);
    confidenceEl.textContent = result.confidence;

    appendSignal("Frames used", result.signals.frames_used);
    appendSignal("Face present ratio", result.signals.face_present_ratio.toFixed(2));
    appendSignal("Blink count", result.signals.blink_count);
    appendSignal("Blink confidence", result.signals.blink_confidence.toFixed(2));
    appendSignal(
      "Head pose delta",
      `yaw ${result.signals.head_pose_delta.yaw.toFixed(1)}, pitch ${
        result.signals.head_pose_delta.pitch.toFixed(1)
      }, roll ${result.signals.head_pose_delta.roll.toFixed(1)}`
    );
    appendSignal(
      "Optical flow variance",
      result.signals.optical_flow_variance.toFixed(3)
    );
    appendSignal("Motion pattern", result.signals.motion_pattern);
    appendSignal("Screen detected", result.signals.screen_detected);
    appendSignal(
      "Screen confidence",
      result.signals.screen_confidence_max.toFixed(2)
    );

    result.reasons.forEach(appendReason);
    renderAnnotatedFrames(result.annotated_frames);

    resultsSection.classList.remove("hidden");
    updateStatus("Result ready");
  } catch (error) {
    updateStatus(`Error: ${error.message}`);
  } finally {
    startBtn.disabled = false;
  }
});

initCamera().catch((error) => {
  updateStatus(`Camera error: ${error.message}`);
  startBtn.disabled = true;
});
