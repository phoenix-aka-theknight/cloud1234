/**
 * Neural Canvas — script.js
 * Fixed + Optimized Version
 */

/* ─────────────────────────────────────────────
   DOM REFERENCES
───────────────────────────────────────────── */

const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const canvasHint = document.getElementById("canvasHint");

const resultIdle = document.getElementById("resultIdle");
const resultLoading = document.getElementById("resultLoading");
const resultOutput = document.getElementById("resultOutput");
const resultError = document.getElementById("resultError");

const errorText = document.getElementById("errorText");

const digitNumber = document.getElementById("digitNumber");
const confValue = document.getElementById("confValue");
const confBar = document.getElementById("confBar");

const probGrid = document.getElementById("probGrid");

/* ─────────────────────────────────────────────
   DEBUG
───────────────────────────────────────────── */

console.log("SCRIPT LOADED");
console.log(canvas);

/* ─────────────────────────────────────────────
   CANVAS STATE
───────────────────────────────────────────── */

let isDrawing = false;
let lastX = 0;
let lastY = 0;
let hasDrawing = false;

/* ─────────────────────────────────────────────
   CANVAS SETUP
───────────────────────────────────────────── */

function setupCanvas() {

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#111111";
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
}

setupCanvas();

/* ─────────────────────────────────────────────
   COORDINATES
───────────────────────────────────────────── */

function getPos(clientX, clientY) {

  const rect = canvas.getBoundingClientRect();

  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY
  };
}

/* ─────────────────────────────────────────────
   DRAWING
───────────────────────────────────────────── */

function startDraw(x, y) {

  isDrawing = true;

  lastX = x;
  lastY = y;

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

  lastX = x;
  lastY = y;
}

function stopDraw() {
  isDrawing = false;
}

/* ─────────────────────────────────────────────
   MOUSE EVENTS
───────────────────────────────────────────── */

canvas.addEventListener("mousedown", (e) => {

  const pos = getPos(e.clientX, e.clientY);

  startDraw(pos.x, pos.y);
});

canvas.addEventListener("mousemove", (e) => {

  const pos = getPos(e.clientX, e.clientY);

  draw(pos.x, pos.y);
});

canvas.addEventListener("mouseup", stopDraw);

canvas.addEventListener("mouseleave", stopDraw);

/* ─────────────────────────────────────────────
   TOUCH EVENTS
───────────────────────────────────────────── */

canvas.addEventListener("touchstart", (e) => {

  e.preventDefault();

  const touch = e.touches[0];

  const pos = getPos(
    touch.clientX,
    touch.clientY
  );

  startDraw(pos.x, pos.y);

}, { passive: false });

canvas.addEventListener("touchmove", (e) => {

  e.preventDefault();

  const touch = e.touches[0];

  const pos = getPos(
    touch.clientX,
    touch.clientY
  );

  draw(pos.x, pos.y);

}, { passive: false });

canvas.addEventListener("touchend", stopDraw);

/* ─────────────────────────────────────────────
   CLEAR CANVAS
───────────────────────────────────────────── */

function clearCanvas() {

  ctx.clearRect(
    0,
    0,
    canvas.width,
    canvas.height
  );

  setupCanvas();

  hasDrawing = false;

  canvasHint.classList.remove("hidden");

  resetResult();
}

clearBtn.addEventListener(
  "click",
  clearCanvas
);

/* ─────────────────────────────────────────────
   RESULT STATES
───────────────────────────────────────────── */

function showState(state) {

  resultIdle.style.display =
    state === "idle" ? "flex" : "none";

  resultLoading.style.display =
    state === "loading" ? "flex" : "none";

  resultOutput.style.display =
    state === "output" ? "flex" : "none";

  resultError.style.display =
    state === "error" ? "flex" : "none";
}

function resetResult() {
  showState("idle");
}

/* ─────────────────────────────────────────────
   PROBABILITY GRID
───────────────────────────────────────────── */

function buildProbGrid(
  probabilities,
  topDigit
) {

  probGrid.innerHTML = "";

  probabilities.forEach((pct, digit) => {

    const item = document.createElement("div");

    item.className =
      "prob-item" +
      (digit === topDigit
        ? " top-pick"
        : "");

    const dLabel =
      document.createElement("div");

    dLabel.className = "prob-digit";

    dLabel.textContent = digit;

    const pLabel =
      document.createElement("div");

    pLabel.className = "prob-pct";

    pLabel.textContent =
      pct.toFixed(1) + "%";

    item.appendChild(dLabel);
    item.appendChild(pLabel);

    probGrid.appendChild(item);
  });
}

/* ─────────────────────────────────────────────
   PREDICT
───────────────────────────────────────────── */

async function predict() {

  if (!hasDrawing) {

    canvasHint.classList.remove("hidden");

    canvasHint.style.color = "#e8890a";

    setTimeout(() => {
      canvasHint.style.color = "";
    }, 1200);

    return;
  }

  predictBtn.disabled = true;
  clearBtn.disabled = true;

  showState("loading");

  try {

    // COMPRESSED IMAGE
    const imageData =
      canvas.toDataURL(
        "image/jpeg",
        0.5
      );

    console.log("Sending prediction request...");

    const response = await fetch(
      "/predict",
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json"
        },

        body: JSON.stringify({
          image: imageData
        })
      }
    );

    const rawText =
      await response.text();

    console.log(
      "RAW RESPONSE:",
      rawText
    );

    if (!rawText) {
      throw new Error(
        "Empty server response"
      );
    }

    let data;

    try {

      data = JSON.parse(rawText);

    } catch {

      throw new Error(
        "Invalid JSON response"
      );
    }

    if (!response.ok) {

      throw new Error(
        data.error ||
        `Server Error ${response.status}`
      );
    }

    /* RESULT */

    showState("output");

    digitNumber.textContent =
      data.digit;

    const conf = data.confidence;

    confValue.textContent =
      conf.toFixed(1) + "%";

    setTimeout(() => {

      confBar.style.width =
        conf + "%";

    }, 80);

    if (data.probabilities) {

      buildProbGrid(
        data.probabilities,
        data.digit
      );
    }

  } catch (err) {

    console.error(
      "Prediction error:",
      err
    );

    errorText.textContent =
      err.message ||
      "Prediction failed";

    showState("error");

  } finally {

    predictBtn.disabled = false;
    clearBtn.disabled = false;
  }
}

/* ─────────────────────────────────────────────
   BUTTON EVENTS
───────────────────────────────────────────── */

predictBtn.addEventListener(
  "click",
  predict
);

/* ─────────────────────────────────────────────
   KEYBOARD SHORTCUTS
───────────────────────────────────────────── */

document.addEventListener(
  "keydown",
  (e) => {

    if (e.key === "Enter") {
      predict();
    }

    if (e.key === "Escape") {
      clearCanvas();
    }
  }
);
