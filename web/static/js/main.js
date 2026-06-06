/**
 * Browser-side logic for the RadarSense dashboard.
 *
 * Receives live data from the server and updates the interface.
 */

const UI_DEBUG = false;
if (UI_DEBUG) console.log("[RadarSense] main.js loading...");

const MAX_LOG_ENTRIES = 120;

// Glow colors per gesture, used in CSS custom property.
const GESTURE_GLOWS = {
  none: "#4b5563", hold: "#06b6d4", push: "#10b981",
  pull: "#f59e0b", tap: "#8b5cf6", wave: "#ec4899",
};

// Emoji shown in the gesture card for each class.
const GESTURE_EMOJIS = {
  none: "✋", hold: "🤚", push: "👆", pull: "👇", tap: "👉", wave: "👋",
};

// Must match the server's label order. Update here if the model is retrained with new classes.
const LABELS = ["none", "hold", "push", "pull", "tap", "wave"];
const BINS = 128;
const ENV_ALPHA = 0.55;  // EMA smoothing factor for the waveform display

// State
let gestureEventCount = 0;
let smoothedBins = new Array(BINS).fill(0);
let elements = {};
let socket = null;
let heatmap = null;
let lastGesture = "none";
let appStartTime = null;

// Per-gesture counts for session statistics (incremented on is_event only).
const sessionStats = { hold: 0, push: 0, pull: 0, tap: 0, wave: 0 };

const $ = id => document.getElementById(id);

// ── Terminal log ──────────────────────────────────────────────────────────────

function logEntry(text, type = "info") {
  if (!elements.terminalLog) return;
  const ts = new Date().toLocaleTimeString("en-GB", { hour12: false });
  const div = document.createElement("div");
  const tag = { info: "INFO", sys: " SYS", warn: "WARN", gest: "GEST", ok: " OK " }[type] ?? "INFO";
  div.className = `terminal__entry terminal__entry--${type}`;
  div.textContent = `[${ts}][${tag}] ${text}`;
  elements.terminalLog.appendChild(div);
  while (elements.terminalLog.children.length > MAX_LOG_ENTRIES) {
    elements.terminalLog.removeChild(elements.terminalLog.firstChild);
  }
  elements.terminalLog.scrollTop = elements.terminalLog.scrollHeight;
}

// ── Connection state ──────────────────────────────────────────────────────────

function setConnectionState(connected, mode = "") {
  if (!elements.connLed) return;
  elements.connLed.className = `conn-indicator__led ${connected ? "conn-indicator__led--on" : ""}`;
  elements.connText.textContent = connected ? (mode || "Connected") : "Disconnected";
  elements.connIndicator.setAttribute("aria-label", connected ? `Connected — ${mode}` : "Disconnected");
  elements.statMode.textContent = mode || "—";
  elements.footerMode.textContent = connected ? mode.toUpperCase() : "Offline";

  // Start the uptime clock when the radar goes live.
  if (connected && !appStartTime) {
    appStartTime = Date.now();
    updateUptime();
  }

  // Reset uptime only on a real disconnect or error, not during connection attempts.
  if (!connected && mode !== "starting…" && !mode.startsWith("connecting")) {
    appStartTime = null;
    updateUptime();
  }

  // Keep Connect disabled while a connection attempt is already in progress.
  const isConnecting = mode === "starting…" || mode.startsWith("connecting");
  if ($("btnDisconnect")) $("btnDisconnect").disabled = !connected;
  if ($("btnRebaseline")) $("btnRebaseline").disabled = !connected;
  if ($("btnStartLive")) $("btnStartLive").disabled = connected || isConnecting;

  // Update the Radar row in the health panel.
  updateHealthRadar(connected, mode);
}

// ── Heatmap (Acconeer-style waterfall) ───────────────────────────────────────

class RadarHeatmap {
  constructor(canvasEl) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext("2d", { willReadFrequently: true });
    this.bins = BINS;
    this.timeCols = 300; // number of time columns shown in the waterfall

    // Set the canvas resolution.
    this.canvas.width = this.timeCols;
    this.canvas.height = this.bins;
    this.ctx.fillStyle = "#0f141c";
    this.ctx.fillRect(0, 0, this.timeCols, this.bins);

    // Precompute a viridis-like colormap (dark blue -> green -> yellow).
    this.colormap = this.generateColormap();
  }

  generateColormap() {
    const map = [];
    for (let i = 0; i < 256; i++) {
      let r, g, b;
      let t = i / 255;
      if (t < 0.33) {
        r = 15; g = 20 + t * 100; b = 28 + t * 220;
      } else if (t < 0.66) {
        let t2 = (t - 0.33) * 3;
        r = 15; g = 53 + t2 * 150; b = 101 - t2 * 50;
      } else {
        let t2 = (t - 0.66) * 3;
        r = 15 + t2 * 240; g = 203 + t2 * 52; b = 51 - t2 * 51;
      }
      map.push([Math.min(255, r), Math.min(255, g), Math.min(255, b)]);
    }
    return map;
  }

  append(dataArray) {
    // Shift everything one pixel to the left to make room for the new column.
    const imgData = this.ctx.getImageData(1, 0, this.timeCols - 1, this.bins);
    this.ctx.putImageData(imgData, 0, 0);

    // Draw the new column on the right edge.
    // y=0 is the top (far range), y=bins is the bottom (near range).
    const newColData = this.ctx.createImageData(1, this.bins);
    for (let y = 0; y < this.bins; y++) {
      let valFloat = dataArray[dataArray.length - 1 - y];
      let idx = Math.floor(Math.min(1.0, Math.max(0.0, valFloat)) * 255);
      const pxIdx = y * 4;
      const rgb = this.colormap[idx];
      newColData.data[pxIdx] = rgb[0];
      newColData.data[pxIdx + 1] = rgb[1];
      newColData.data[pxIdx + 2] = rgb[2];
      newColData.data[pxIdx + 3] = 255;
    }
    this.ctx.putImageData(newColData, this.timeCols - 1, 0);
  }
}

// ── Initialization ────────────────────────────────────────────────────────────

function init() {
  if (UI_DEBUG) console.log("[RadarSense] Initializing DOM and Socket...");
  try {
    // Grab all the DOM elements we'll be updating.
    elements = {
      connLed: $("connLed"),
      connText: $("connText"),
      connIndicator: $("connIndicator"),
      gestureEmoji: $("gestureEmoji"),
      gestureLabel: $("gestureLabel"),
      gestureConfidence: $("gestureConfidence"),
      gestureCard: $("gestureCard"),
      gestureStatusPill: $("gestureStatusPill"),
      confidenceBars: $("confidenceBars"),
      terminalLog: $("terminalLog"),
      statFps: $("statFps"),
      statCount: $("statCount"),
      statMode: $("statMode"),
      statUptime: $("statUptime"),
      footerMode: $("footerMode"),
      fpsTag: $("fpsTag"),
      peakDistPill: $("peakDistPill"),
      binCountPill: $("binCountPill"),
      heatmapCanvas: $("heatmapCanvas")
    };

    if (!elements.heatmapCanvas) throw new Error("heatmapCanvas not found");
    heatmap = new RadarHeatmap(elements.heatmapCanvas);

    if (typeof io === 'undefined') throw new Error("Socket.IO not found");
    socket = io();

    // ── Socket.IO event handlers ──────────────────────────────────────────

    socket.on("connect", () => {
      logEntry("Connected to server.", "sys");
      updateHealthSocket(true);
      // Ask the server for the current radar status right away.
      socket.emit("get_status");
    });

    socket.on("disconnect", () => {
      logEntry("Disconnected from server.", "warn");
      setConnectionState(false, "Offline");
      updateHealthSocket(false);
    });

    socket.on("radar_status", ({ connected, mode }) => {
      setConnectionState(connected, mode);
      logEntry(`Status: connected=${connected}, mode=${mode}`, "sys");
    });

    socket.on("radar_data", (payload) => {
      const frame = payload.frame || payload.data;
      if (!frame || frame.length < 10) return;

      // Update the bin count pill.
      if (elements.binCountPill && payload.bins) {
        elements.binCountPill.textContent = `${payload.bins} bins`;
      }

      // Find the peak bin and convert it to a distance in cm.
      if (elements.peakDistPill && payload.range_start !== undefined && payload.range_end !== undefined) {
        let peakIdx = 0;
        let peakVal = -Infinity;
        for (let i = 0; i < frame.length; i++) {
          if (frame[i] > peakVal) {
            peakVal = frame[i];
            peakIdx = i;
          }
        }
        const ratio = frame.length > 1 ? peakIdx / (frame.length - 1) : 0;
        const distM = payload.range_start + ratio * (payload.range_end - payload.range_start);
        elements.peakDistPill.textContent = `${Math.round(distM * 100)} cm`;
        elements.peakDistPill.classList.toggle("active", peakVal > 0);
      }

      // Downsample and EMA-smooth the frame before rendering.
      const rawLen = frame.length;
      for (let i = 0; i < BINS; i++) {
        const srcIdx = (i / (BINS - 1)) * (rawLen - 1);
        const idx0 = Math.floor(srcIdx);
        const idx1 = Math.min(rawLen - 1, idx0 + 1);
        const t = srcIdx - idx0;
        const val = frame[idx0] * (1 - t) + frame[idx1] * t;
        smoothedBins[i] = smoothedBins[i] * (1 - ENV_ALPHA) + val * ENV_ALPHA;
      }

      // Normalize to [0, 1] before passing to the colormap.
      let minV = 999999, maxV = -999999;
      for (let i = 0; i < BINS; i++) {
        if (smoothedBins[i] < minV) minV = smoothedBins[i];
        if (smoothedBins[i] > maxV) maxV = smoothedBins[i];
      }
      const rng = (maxV - minV) > 1 ? (maxV - minV) : 1;
      const normalized = smoothedBins.map(v => (v - minV) / rng);

      heatmap.append(normalized);
    });

    socket.on("prediction", (payload) => {
      const gesture = payload.gesture || "none";
      const probabilities = payload.probabilities || {};
      const fps = payload.fps || 0;

      if (elements.statFps) elements.statFps.textContent = Number(fps).toFixed(1);
      if (elements.fpsTag) elements.fpsTag.textContent = `${Number(fps).toFixed(1)} FPS`;

      // Server sets is_event=true on exactly one frame per confirmed gesture.
      // Counting this instead of label changes avoids double-counting display artifacts.
      if (payload.is_event && gesture !== "none") {
        gestureEventCount++;
        if (elements.statCount) elements.statCount.textContent = gestureEventCount;
        logEntry(`Gesture: ${gesture.toUpperCase()} detected`, "gest");
        updateSessionStats(gesture);
      }

      updateGestureDisplay(gesture, probabilities);
      updateConfidenceBars(probabilities);
    });

    // ── Button handlers ───────────────────────────────────────────────────

    if ($("btnStartLive")) {
      $("btnStartLive").addEventListener("click", () => {
        logEntry("Connecting to radar stream...", "sys");
        if ($("btnStartLive")) $("btnStartLive").disabled = true;
        socket.emit("start_radar");
      });
    }
    if ($("btnDisconnect")) {
      $("btnDisconnect").addEventListener("click", () => {
        logEntry("Disconnecting...", "warn");
        appStartTime = null;
        updateUptime();
        socket.emit("stop_radar");
      });
    }
    if ($("btnRebaseline")) {
      $("btnRebaseline").addEventListener("click", () => {
        logEntry("Recalibrating inference state...", "warn");
        socket.emit("rebase");
      });
    }
    if ($("btnClearLog")) {
      $("btnClearLog").addEventListener("click", () => {
        elements.terminalLog.innerHTML = `<div class="terminal__entry terminal__entry--sys">[SYS ] Log cleared.</div>`;
      });
    }

    buildConfidenceBars();
    setConnectionState(false);
    if (UI_DEBUG) console.log("[RadarSense] Initialization complete.");

  } catch (err) {
    console.error("[RadarSense] Initialization failed:", err);
    alert("Dashboard Initialization Failed: " + err.message);
  }
}

// ── Confidence bars ───────────────────────────────────────────────────────────

function buildConfidenceBars() {
  if (!elements.confidenceBars) return;
  elements.confidenceBars.innerHTML = LABELS.map(l => `
    <div class="conf-bar" id="conf-bar-${l}">
      <div class="conf-bar__label">${l.toUpperCase()}</div>
      <div class="conf-bar__track"><div class="conf-bar__fill" style="width: 0%; --gesture-glow: ${GESTURE_GLOWS[l]}"></div></div>
      <div class="conf-bar__value">0%</div>
    </div>
  `).join("");
}

function updateConfidenceBars(probabilities) {
  if (!elements.confidenceBars) return;
  for (const label of LABELS) {
    const row = document.getElementById(`conf-bar-${label}`);
    if (!row) continue;
    const pct = Math.round((probabilities[label] || 0) * 100);
    row.querySelector(".conf-bar__fill").style.width = pct + "%";
    row.querySelector(".conf-bar__value").textContent = pct + "%";
    if (label !== "none" && pct > 65) {
      row.classList.add("conf-bar--active");
    } else {
      row.classList.remove("conf-bar--active");
    }
  }
}

// ── Gesture display ───────────────────────────────────────────────────────────

function updateGestureDisplay(gesture, probabilities) {
  // Show the FSM-confirmed gesture from the server.
  // Raw CNN probabilities only go to the confidence bars, not the main label.

  updateGestureChips(gesture || "none");

  if (gesture !== "none") {
    elements.gestureEmoji.textContent = GESTURE_EMOJIS[gesture] || "❓";
    elements.gestureLabel.textContent = gesture.toUpperCase();
    elements.gestureCard.style.setProperty("--gesture-glow", GESTURE_GLOWS[gesture]);
    elements.gestureStatusPill.className = "pill pill--active";
    elements.gestureStatusPill.textContent = "ACTIVE";
    elements.gestureCard.classList.add("gesture-card--active");

    const outConf = Number(probabilities?.[gesture] ?? 0);
    elements.gestureConfidence.textContent = `${Math.round(outConf * 100)}% confidence`;

    lastGesture = gesture;
  } else {
    resetGestureToNone();
    elements.gestureConfidence.textContent = "No confirmed gesture";
  }
}

function updateGestureChips(activeGesture) {
  document.querySelectorAll(".gesture-chip").forEach(chip => {
    const gesture = chip.dataset.gesture || "none";
    chip.classList.toggle("gesture-chip--active", gesture === activeGesture);
  });
}

function resetGestureToNone() {
  updateGestureChips("none");

  elements.gestureEmoji.textContent = GESTURE_EMOJIS["none"];
  elements.gestureLabel.textContent = "NONE";
  elements.gestureCard.style.setProperty("--gesture-glow", GESTURE_GLOWS["none"]);
  elements.gestureStatusPill.className = "pill pill--status";
  elements.gestureStatusPill.textContent = "IDLE";
  elements.gestureCard.classList.remove("gesture-card--active");
  elements.gestureConfidence.textContent = "Awaiting signal...";
  lastGesture = "none";
}

// ── Uptime counter ────────────────────────────────────────────────────────────

function updateUptime() {
  if (!elements.statUptime) return;

  if (!appStartTime) {
    elements.statUptime.textContent = "00:00";
    return;
  }

  const seconds = Math.floor((Date.now() - appStartTime) / 1000);
  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");
  elements.statUptime.textContent = `${mm}:${ss}`;
}

// ── Session Statistics ────────────────────────────────────────────────────────

function updateSessionStats(gesture) {
  if (!sessionStats.hasOwnProperty(gesture)) return;
  sessionStats[gesture]++;

  const el = $(`sessCount-${gesture}`);
  if (el) el.textContent = sessionStats[gesture];

  // Update the total pill.
  const total = Object.values(sessionStats).reduce((a, b) => a + b, 0);
  const pill = $("sessTotal");
  if (pill) pill.textContent = `${total} event${total !== 1 ? "s" : ""}`;
}

// ── System Health ─────────────────────────────────────────────────────────────

function updateHealthRadar(connected, mode) {
  const dot = $("healthDotRadar");
  const val = $("healthValRadar");
  if (!dot || !val) return;

  if (connected) {
    dot.className = "health-dot health-dot--live";
    val.className = "health-val health-val--live";
    val.textContent = "Live";
  } else {
    dot.className = "health-dot";
    val.className = "health-val";
    val.textContent = mode === "error" ? "Error" : "Offline";
  }
}

function updateHealthSocket(connected) {
  const dot = $("healthDotSocket");
  const val = $("healthValSocket");
  if (!dot || !val) return;

  if (connected) {
    dot.className = "health-dot health-dot--ok";
    val.className = "health-val health-val--ok";
    val.textContent = "Connected";
  } else {
    dot.className = "health-dot";
    val.className = "health-val";
    val.textContent = "Disconnected";
  }
}

// ── View Switcher ────────────────────────────────────────────────────────────

function initTabs() {
  const tabLive = $("tabLive");
  const tabReport = $("tabReport");
  const liveView = $("liveView");
  const reportView = $("reportView");

  if (!tabLive || !tabReport || !liveView || !reportView) return;

  tabLive.addEventListener("click", () => {
    tabLive.classList.add("view-tab--active");
    tabLive.setAttribute("aria-selected", "true");
    tabReport.classList.remove("view-tab--active");
    tabReport.setAttribute("aria-selected", "false");

    liveView.style.display = "";
    reportView.style.display = "none";
  });

  tabReport.addEventListener("click", () => {
    tabReport.classList.add("view-tab--active");
    tabReport.setAttribute("aria-selected", "true");
    tabLive.classList.remove("view-tab--active");
    tabLive.setAttribute("aria-selected", "false");

    liveView.style.display = "none";
    reportView.style.display = "flex";
  });
}

// ── Image Modal ───────────────────────────────────────────────────────────────

function initImageModal() {
  const modal = $("imageModal");
  const modalImg = $("modalImage");
  const backdrop = $("modalBackdrop");
  if (!modal || !modalImg || !backdrop) return;

  const chartImages = document.querySelectorAll(".chart-img img");

  chartImages.forEach(img => {
    img.addEventListener("click", () => {
      modalImg.src = img.src;
      modal.classList.add("image-modal--active");
      modal.setAttribute("aria-hidden", "false");
    });
  });

  const closeModal = () => {
    modal.classList.remove("image-modal--active");
    modal.setAttribute("aria-hidden", "true");
    setTimeout(() => { modalImg.src = ""; }, 200);
  };

  backdrop.addEventListener("click", closeModal);
  modalImg.addEventListener("click", closeModal);
}

// Run init when the page is ready, then tick the uptime counter every second.
document.addEventListener("DOMContentLoaded", () => {
  init();
  initTabs();
  initImageModal();
  setInterval(updateUptime, 1000);
});
