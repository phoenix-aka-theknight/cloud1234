/**
 * Neural Canvas — script.js
 * Handles:
 *  - Smooth drawing on <canvas> (mouse + touch)
 *  - Sending canvas image to Flask /predict endpoint
 *  - Rendering prediction results (digit, confidence, probability grid)
 */

/* ── DOM refs ─────────────────────────────────────────────── */
const canvas      = document.getElementById("drawingCanvas");
const ctx         = canvas.getContext("2d");
const clearBtn    = document.getElementById("clearBtn");
const predictBtn  = document.getElementById("predictBtn");
const canvasHint  = document.getElementById("canvasHint");

const resultIdle    = document.getElementById("resultIdle");
const resultLoading = document.getElementById("resultLoading");
const resultOutput  = document.getElementById("resultOutput");
const resultError   = document.getElementById("resultError");
const errorText     = document.getElementById("errorText");

const digitNumber = document.getElementById("digitNumber");
const confValue   = document.getElementById("confValue");
const confBar     = document.getElementById("confBar");
const probGrid    = document.getElementById("probGrid");

/* ── Canvas state ─────────────────────────────────────────── */
let isDrawing  = false;
let lastX      = 0;
let lastY      = 0;
let hasDrawing = false;   // track whether user drew anything

/* ── Canvas setup ─────────────────────────────────────────── */
function setupCanvas() {
  // Fill canvas with white (MNIST background is black, but we invert server-side)
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Brush settings — thick stroke for easy recognition
  ctx.strokeStyle = "#1a1a1a";   // near-black
  ctx.lineWidth   = 18;
  ctx.lineCap     = "round";
  ctx.lineJoin    = "round";
}

setupCanvas();

/* ── Coordinate helpers ───────────────────────────────────── */
/**
 * Map a client (page) coordinate to canvas pixel coordinates,
 * accounting for any CSS scaling of the canvas element.
 */
function getPos(clientX, clientY) {
  const rect  = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top)  * scaleY
  };
}

/* ── Drawing handlers ─────────────────────────────────────── */
function startDraw(x, y) {
  isDrawing = true;
  [lastX, lastY] = [x, y];

  // Hide the "draw here" hint
  if (!hasDrawing) {
    hasDrawing = true;
    canvasHint.classList.add("hidden");
  }
}

function draw(x, y) {
  if (!isDrawing) return;

  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  [lastX, lastY] = [x, y];
}

function stopDraw() {
  isDrawing = false;
}

/* Mouse events */
canvas.addEventListener("mousedown", e => {
  const { x, y } = getPos(e.clientX, e.clientY);
  startDraw(x, y);
});

canvas.addEventListener("mousemove", e => {
  const { x, y } = getPos(e.clientX, e.clientY);
  draw(x, y);
});

canvas.addEventListener("mouseup",    stopDraw);
canvas.addEventListener("mouseleave", stopDraw);

/* Touch events (mobile) */
canvas.addEventListener("touchstart", e => {
  e.preventDefault();
  const touch = e.touches[0];
  const { x, y } = getPos(touch.clientX, touch.clientY);
  startDraw(x, y);
}, { passive: false });

canvas.addEventListener("touchmove", e => {
  e.preventDefault();
  const touch = e.touches[0];
  const { x, y } = getPos(touch.clientX, touch.clientY);
  draw(x, y);
}, { passive: false });

canvas.addEventListener("touchend", stopDraw);

/* ── Clear canvas ─────────────────────────────────────────── */
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  setupCanvas();
  hasDrawing = false;
  canvasHint.classList.remove("hidden");
  resetResult();
}

clearBtn.addEventListener("click", clearCanvas);

/* ── Result state helpers ─────────────────────────────────── */
function showState(state) {
  resultIdle.style.display    = state === "idle"    ? "flex" : "none";
  resultLoading.style.display = state === "loading" ? "flex" : "none";
  resultOutput.style.display  = state === "output"  ? "flex" : "none";
  resultError.style.display   = state === "error"   ? "flex" : "none";
}

function resetResult() {
  showState("idle");
}

/* ── Probability grid ─────────────────────────────────────── */
function buildProbGrid(probabilities, topDigit) {
  probGrid.innerHTML = "";

  probabilities.forEach((pct, digit) => {
    const item = document.createElement("div");
    item.className = "prob-item" + (digit === topDigit ? " top-pick" : "");

    const dLabel = document.createElement("div");
    dLabel.className = "prob-digit";
    dLabel.textContent = digit;

    const pLabel = document.createElement("div");
    pLabel.className = "prob-pct";
    pLabel.textContent = pct.toFixed(1) + "%";

    item.appendChild(dLabel);
    item.appendChild(pLabel);
    probGrid.appendChild(item);
  });
}

/* ── Predict ──────────────────────────────────────────────── */
async function predict() {
  if (!hasDrawing) {
    // Gently pulse the canvas hint
    canvasHint.classList.remove("hidden");
    canvasHint.style.color = "#e8890a";
    setTimeout(() => { canvasHint.style.color = ""; }, 1200);
    return;
  }

  // Disable buttons while request is in-flight
  predictBtn.disabled = true;
  clearBtn.disabled   = true;
  showState("loading");

  try {
    // Export canvas as base64 PNG
    const imageData = canvas.toDataURL("image/png");

    // POST to Flask backend
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData })
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || `Server error ${response.status}`);
    }

    const data = await response.json();

    // ── Render result ──────────────────────────────────────
    showState("output");

    // Predicted digit (big number)
    digitNumber.textContent = data.digit;

    // Confidence bar — animate after a short delay
    const conf = data.confidence;
    confValue.textContent = conf.toFixed(1) + "%";
    setTimeout(() => { confBar.style.width = conf + "%"; }, 80);

    // Probability grid
    buildProbGrid(data.probabilities, data.digit);

  } catch (err) {
    console.error("Prediction error:", err);
    errorText.textContent = err.message || "Could not reach the server. Make sure Flask is running.";
    showState("error");
  } finally {
    predictBtn.disabled = false;
    clearBtn.disabled   = false;
  }
}

predictBtn.addEventListener("click", predict);

/* ── Keyboard shortcut: Enter = Predict, Escape = Clear ───── */
document.addEventListener("keydown", e => {
  if (e.key === "Enter")  predict();
  if (e.key === "Escape") clearCanvas();
});