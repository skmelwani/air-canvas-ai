/*
 * Air Canvas AI — web version
 * ============================
 * Browser port of the Python/OpenCV desktop app. Runs entirely client-side:
 * MediaPipe's HandLandmarker (WebAssembly build) finds 21 hand landmarks per
 * frame from the webcam, geometry on those points decides which single
 * finger is extended, and that finger both picks a color and drives the
 * brush position on a persistent drawing layer composited over the video.
 *
 * Gesture -> action (mirrors the Python version's FINGER_ACTIONS mapping):
 *   index only  -> draw Brown
 *   middle only -> draw Blue
 *   ring only   -> draw Green
 *   thumb only  -> Eraser
 *   fist / other -> pen lifted, no drawing
 */

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const VISION_PACKAGE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1";
const WASM_BASE_URL = `${VISION_PACKAGE}/wasm`;

const BRUSH_THICKNESS = 8;
const ERASER_THICKNESS = 45;

// 21-point hand landmark indices (MediaPipe hand model)
const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_MCP = 5, INDEX_PIP = 6, INDEX_TIP = 8;
const MIDDLE_PIP = 10, MIDDLE_TIP = 12;
const RING_PIP = 14, RING_TIP = 16;
const PINKY_PIP = 18, PINKY_TIP = 20;

// Each single-finger gesture maps to (display name, canvas color, tip landmark).
const FINGER_ACTIONS = {
  index: { name: "Brown", color: "#8B4513", tip: INDEX_TIP },
  middle: { name: "Blue", color: "#0064FF", tip: MIDDLE_TIP },
  ring: { name: "Green", color: "#00C800", tip: RING_TIP },
  thumb: { name: "Eraser", color: "#3A3F4B", tip: THUMB_TIP },
};
const GESTURE_ORDER = ["index", "middle", "ring", "thumb"];

// Standard 21-point hand skeleton connections, for the overlay drawing.
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

// --------------------------------------------------------------------------- //
// Gesture helpers (ported from the Python fingers_extended / classify_gesture)
// --------------------------------------------------------------------------- //

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function fingersExtended(pts) {
  const index = pts[INDEX_TIP].y < pts[INDEX_PIP].y;
  const middle = pts[MIDDLE_TIP].y < pts[MIDDLE_PIP].y;
  const ring = pts[RING_TIP].y < pts[RING_PIP].y;

  const palmScale = dist(pts[WRIST], pts[INDEX_MCP]) + 1e-6;
  const thumb = dist(pts[THUMB_TIP], pts[INDEX_MCP]) > 0.6 * palmScale;

  return { thumb, index, middle, ring };
}

function classifyGesture(fingers) {
  const { thumb, index, middle, ring } = fingers;
  if (index && !middle && !ring) return "index";
  if (middle && !index && !ring) return "middle";
  if (ring && !index && !middle) return "ring";
  if (thumb && !index && !middle && !ring) return "thumb";
  if (!thumb && !index && !middle && !ring) return "fist";
  return "other";
}

// --------------------------------------------------------------------------- //
// App state
// --------------------------------------------------------------------------- //

const video = document.getElementById("video");
const output = document.getElementById("output");
const outCtx = output.getContext("2d");
const overlayMsg = document.getElementById("overlayMsg");
const statusText = document.getElementById("statusText");
const startBtn = document.getElementById("startBtn");
const clearBtn = document.getElementById("clearBtn");
const saveBtn = document.getElementById("saveBtn");
const stopBtn = document.getElementById("stopBtn");

let handLandmarker = null;
let stream = null;
let rafId = null;
let running = false;

let drawLayer = null; // persistent, transparent, same size as video
let drawCtx = null;
let prevPoint = null;

let lastFrameTime = performance.now();
const fpsHistory = [];

function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.classList.toggle("error", isError);
}

// --------------------------------------------------------------------------- //
// Setup
// --------------------------------------------------------------------------- //

async function loadHandLandmarker() {
  const { HandLandmarker, FilesetResolver } = await import(VISION_PACKAGE);
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE_URL);
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });
}

async function startCamera() {
  startBtn.disabled = true;
  setStatus("Loading hand-tracking model…");

  try {
    if (!handLandmarker) {
      handLandmarker = await loadHandLandmarker();
    }
  } catch (err) {
    console.error(err);
    setStatus("Failed to load the hand-tracking model. Check your internet connection and reload.", true);
    startBtn.disabled = false;
    return;
  }

  setStatus("Requesting camera access…");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    console.error(err);
    setStatus("Camera access was denied or unavailable. Allow camera access and try again.", true);
    startBtn.disabled = false;
    return;
  }

  video.srcObject = stream;
  await video.play();

  const width = video.videoWidth;
  const height = video.videoHeight;
  output.width = width;
  output.height = height;

  drawLayer = document.createElement("canvas");
  drawLayer.width = width;
  drawLayer.height = height;
  drawCtx = drawLayer.getContext("2d");
  prevPoint = null;

  overlayMsg.classList.add("hidden");
  clearBtn.disabled = false;
  saveBtn.disabled = false;
  stopBtn.disabled = false;

  running = true;
  lastFrameTime = performance.now();
  rafId = requestAnimationFrame(renderLoop);
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  clearBtn.disabled = true;
  saveBtn.disabled = true;
  stopBtn.disabled = true;
  startBtn.disabled = false;
  overlayMsg.classList.remove("hidden");
  setStatus("");
}

// --------------------------------------------------------------------------- //
// Drawing
// --------------------------------------------------------------------------- //

function landmarkPixels(landmarks, width, height) {
  return landmarks.map((lm) => ({ x: lm.x * width, y: lm.y * height }));
}

function drawHandSkeleton(ctx, pts, width) {
  ctx.strokeStyle = "rgba(255,255,255,0.8)";
  ctx.lineWidth = 1;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(width - pts[a].x, pts[a].y);
    ctx.lineTo(width - pts[b].x, pts[b].y);
    ctx.stroke();
  }
  ctx.fillStyle = "#ffff00";
  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(width - p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawGestureLegend(ctx, activeGesture) {
  const swatchW = 130;
  const y0 = 10;
  const y1 = 65;
  ctx.font = "13px -apple-system, sans-serif";
  ctx.textBaseline = "top";

  GESTURE_ORDER.forEach((key, i) => {
    const { name, color } = FINGER_ACTIONS[key];
    const x0 = 10 + i * (swatchW + 8);
    ctx.fillStyle = color;
    ctx.fillRect(x0, y0, swatchW, y1 - y0);
    ctx.strokeStyle = key === activeGesture ? "#ffffff" : "rgba(120,120,120,0.8)";
    ctx.lineWidth = key === activeGesture ? 4 : 1;
    ctx.strokeRect(x0, y0, swatchW, y1 - y0);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(`${key[0].toUpperCase()}${key.slice(1)}: ${name}`, x0 + 4, y1 + 6);
  });

  const fistX0 = 10 + GESTURE_ORDER.length * (swatchW + 8);
  ctx.fillStyle = "#282828";
  ctx.fillRect(fistX0, y0, swatchW, y1 - y0);
  ctx.strokeStyle = activeGesture === "fist" ? "#ffffff" : "rgba(120,120,120,0.8)";
  ctx.lineWidth = activeGesture === "fist" ? 4 : 1;
  ctx.strokeRect(fistX0, y0, swatchW, y1 - y0);
  ctx.fillStyle = "#ffffff";
  ctx.fillText("Fist: No Draw", fistX0 + 4, y1 + 6);
}

function renderLoop(now) {
  if (!running) return;

  const width = output.width;
  const height = output.height;

  const results = handLandmarker.detectForVideo(video, now);

  // Draw the mirrored webcam frame as the base layer.
  outCtx.save();
  outCtx.translate(width, 0);
  outCtx.scale(-1, 1);
  outCtx.drawImage(video, 0, 0, width, height);
  outCtx.restore();

  let gesture = "none";

  if (results.landmarks && results.landmarks.length > 0) {
    const pts = landmarkPixels(results.landmarks[0], width, height);
    const fingers = fingersExtended(pts);
    gesture = classifyGesture(fingers);

    if (gesture in FINGER_ACTIONS) {
      const { name, color, tip } = FINGER_ACTIONS[gesture];
      const point = { x: width - pts[tip].x, y: pts[tip].y };
      const thickness = name === "Eraser" ? ERASER_THICKNESS : BRUSH_THICKNESS;

      drawCtx.lineCap = "round";
      drawCtx.lineJoin = "round";
      drawCtx.lineWidth = thickness;
      drawCtx.globalCompositeOperation = name === "Eraser" ? "destination-out" : "source-over";
      drawCtx.strokeStyle = color;

      if (prevPoint) {
        drawCtx.beginPath();
        drawCtx.moveTo(prevPoint.x, prevPoint.y);
        drawCtx.lineTo(point.x, point.y);
        drawCtx.stroke();
      } else if (name !== "Eraser") {
        drawCtx.beginPath();
        drawCtx.arc(point.x, point.y, thickness / 2, 0, Math.PI * 2);
        drawCtx.fillStyle = color;
        drawCtx.fill();
      }
      drawCtx.globalCompositeOperation = "source-over";
      prevPoint = point;
    } else {
      prevPoint = null;
    }

    // Composite the persistent drawing layer, then the live skeleton on top.
    outCtx.drawImage(drawLayer, 0, 0);
    drawHandSkeleton(outCtx, pts, width);

    if (gesture in FINGER_ACTIONS) {
      const { tip } = FINGER_ACTIONS[gesture];
      const cursor = { x: width - pts[tip].x, y: pts[tip].y };
      outCtx.strokeStyle = "#00ff00";
      outCtx.lineWidth = 2;
      outCtx.beginPath();
      outCtx.arc(cursor.x, cursor.y, 10, 0, Math.PI * 2);
      outCtx.stroke();
    }
  } else {
    prevPoint = null;
    outCtx.drawImage(drawLayer, 0, 0);
  }

  drawGestureLegend(outCtx, gesture);

  // FPS (rolling average over the last 30 frames).
  const delta = now - lastFrameTime;
  lastFrameTime = now;
  fpsHistory.push(1000 / Math.max(delta, 1));
  if (fpsHistory.length > 30) fpsHistory.shift();
  const fps = fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length;

  outCtx.font = "16px -apple-system, sans-serif";
  outCtx.fillStyle = "#00ff00";
  outCtx.fillText(`FPS: ${fps.toFixed(1)}`, 10, height - 34);
  outCtx.fillStyle = "#ffff00";
  outCtx.fillText(`Gesture: ${gesture}`, 10, height - 12);

  rafId = requestAnimationFrame(renderLoop);
}

// --------------------------------------------------------------------------- //
// Controls
// --------------------------------------------------------------------------- //

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);

clearBtn.addEventListener("click", () => {
  if (!drawCtx) return;
  drawCtx.clearRect(0, 0, drawLayer.width, drawLayer.height);
  prevPoint = null;
});

saveBtn.addEventListener("click", () => {
  if (!drawLayer) return;
  const link = document.createElement("a");
  link.download = `air-canvas-drawing-${Date.now()}.png`;
  link.href = drawLayer.toDataURL("image/png");
  link.click();
});

window.addEventListener("beforeunload", () => {
  if (stream) stream.getTracks().forEach((track) => track.stop());
});
