/**
 * AntRadar Dashboard — main.js  (v2)
 * Native Canvas Waterfall/Heatmap + Confidence UI
 */

console.log("[AntRadar] main.js loading...");

const MAX_LOG_ENTRIES = 120;
const GESTURE_GLOWS = {
  none: "#4b5563", hold: "#06b6d4", push: "#10b981",
  pull: "#f59e0b", tap:  "#8b5cf6", wave: "#ec4899",
};
const GESTURE_EMOJIS = {
  none: "✋", hold: "🤚", push: "👆", pull: "👇", tap: "👉", wave: "👋",
};
const LABELS = ["none", "hold", "push", "pull", "tap", "wave"];
const BINS = 128;
const ENV_ALPHA = 0.55; 

// STATE
let gestureEventCount = 0;
let smoothedBins      = new Array(BINS).fill(0);
let elements = {};
let socket = null;
let heatmap = null;
let lastGesture = "none";

const $ = id => document.getElementById(id);

// UI HELPERS
function logEntry(text, type = "info") {
  if (!elements.terminalLog) return;
  const ts  = new Date().toLocaleTimeString("en-GB", { hour12: false });
  const div = document.createElement("div");
  const tag = { info: "INFO", sys: " SYS", warn: "WARN", gest: "GEST", ok: " OK " }[type] ?? "INFO";
  div.className   = `terminal__entry terminal__entry--${type}`;
  div.textContent = `[${ts}][${tag}] ${text}`;
  elements.terminalLog.appendChild(div);
  while (elements.terminalLog.children.length > MAX_LOG_ENTRIES) {
    elements.terminalLog.removeChild(elements.terminalLog.firstChild);
  }
  elements.terminalLog.scrollTop = elements.terminalLog.scrollHeight;
}

function setConnectionState(connected, mode = "") {
  if (!elements.connLed) return;
  elements.connLed.className   = `conn-indicator__led ${connected ? "conn-indicator__led--on" : ""}`;
  elements.connText.textContent = connected ? (mode || "Connected") : "Disconnected";
  elements.connIndicator.setAttribute("aria-label", connected ? `Connected — ${mode}` : "Disconnected");
  elements.statMode.textContent   = mode || "—";
  elements.footerMode.textContent = connected ? mode.toUpperCase() : "Offline";

  if ($("btnDisconnect")) $("btnDisconnect").disabled  = !connected;
  if ($("btnRebaseline")) $("btnRebaseline").disabled  = !connected;
  if ($("btnStartLive"))  $("btnStartLive").disabled   = connected;
}

// HEATMAP CLASS (Acconeer style Waterfall)
class RadarHeatmap {
  constructor(canvasEl) {
    this.canvas = canvasEl;
    this.ctx = canvasEl.getContext("2d", { willReadFrequently: true });
    this.bins = BINS;
    this.timeCols = 300; // how many columns to keep in history
    
    // Resize internal resolution
    this.canvas.width = this.timeCols;
    this.canvas.height = this.bins;
    this.ctx.fillStyle = "#0f141c"; // base dark color
    this.ctx.fillRect(0, 0, this.timeCols, this.bins);
    
    // Precompute viridis-like colormap (dark blue -> green -> yellow)
    this.colormap = this.generateColormap();
  }
  
  generateColormap() {
    const map = [];
    for (let i = 0; i < 256; i++) {
      let r, g, b;
      let t = i / 255;
      if (t < 0.33) {
        r = 15; g = 20 + t*100; b = 28 + t*220;
      } else if (t < 0.66) {
        let t2 = (t - 0.33) * 3;
        r = 15; g = 53 + t2*150; b = 101 - t2*50;
      } else {
        let t2 = (t - 0.66) * 3;
        r = 15 + t2*240; g = 203 + t2*52; b = 51 - t2*51;
      }
      map.push([Math.min(255,r), Math.min(255,g), Math.min(255,b)]);
    }
    return map;
  }
  
  append(dataArray) {
    // 1. Shift canvas left by 1 pixel
    const imgData = this.ctx.getImageData(1, 0, this.timeCols - 1, this.bins);
    this.ctx.putImageData(imgData, 0, 0);
    
    // 2. Map data to colors and draw new column on the right edge
    const newColData = this.ctx.createImageData(1, this.bins);
    for (let y = 0; y < this.bins; y++) {
      // Data is 128 bins. Acconeer puts far distances at the top usually, 
      // y=0 is top canvas (far), y=bins is bottom canvas (near).
      let valFloat = dataArray[dataArray.length - 1 - y]; 
      
      // Auto-scale to 0-255 based on normalised range
      let idx = Math.floor(Math.min(1.0, Math.max(0.0, valFloat)) * 255);
      
      const pxIdx = y * 4;
      const rgb = this.colormap[idx];
      newColData.data[pxIdx]   = rgb[0];
      newColData.data[pxIdx+1] = rgb[1];
      newColData.data[pxIdx+2] = rgb[2];
      newColData.data[pxIdx+3] = 255;
    }
    this.ctx.putImageData(newColData, this.timeCols - 1, 0);
  }
}

// INITIALIZATION
function init() {
  console.log("[AntRadar] Initializing DOM and Socket...");
  try {
    elements = {
      connLed:           $("connLed"),
      connText:          $("connText"),
      connIndicator:     $("connIndicator"),
      gestureEmoji:      $("gestureEmoji"),
      gestureLabel:      $("gestureLabel"),
      gestureConfidence: $("gestureConfidence"),
      gestureCard:       $("gestureCard"),
      gestureStatusPill: $("gestureStatusPill"),
      confidenceBars:    $("confidenceBars"),
      terminalLog:       $("terminalLog"),
      statFps:           $("statFps"),
      statCount:         $("statCount"),
      statMode:          $("statMode"),
      statUptime:        $("statUptime"),
      footerMode:        $("footerMode"),
      heatmapCanvas:     $("heatmapCanvas")
    };

    if (!elements.heatmapCanvas) throw new Error("heatmapCanvas not found");
    
    heatmap = new RadarHeatmap(elements.heatmapCanvas);

    // Setup Socket.IO
    if (typeof io === 'undefined') throw new Error("Socket.IO not found");
    socket = io();

    socket.on("connect", () => {
      logEntry("Connected to server.", "sys");
      // Request current status immediately on connection/refresh
      socket.emit("get_status");
    });
    socket.on("disconnect", () => {
      logEntry("Disconnected from server.", "warn");
      setConnectionState(false, "Offline");
    });
    socket.on("radar_status", ({ connected, mode }) => {
      setConnectionState(connected, mode);
      logEntry(`Status: connected=${connected}, mode=${mode}`, "sys");
    });

    socket.on("radar_data", (payload) => {
      const frame = payload.frame || payload.data;
      if (!frame || frame.length < 10) return;
      
      // We must downsample/interpolate `frame` down to `BINS=128` to fit the canvas height
      const rawLen = frame.length;
      for (let i = 0; i < BINS; i++) {
        // Simple linear interpolation
        const srcIdx = (i / (BINS - 1)) * (rawLen - 1);
        const idx0 = Math.floor(srcIdx);
        const idx1 = Math.min(rawLen - 1, idx0 + 1);
        const t = srcIdx - idx0;
        const val = frame[idx0] * (1 - t) + frame[idx1] * t;
        
        smoothedBins[i] = smoothedBins[i] * (1 - ENV_ALPHA) + val * ENV_ALPHA;
      }
      
      // Auto-scale dynamically finding max for the frame to make the heatmap glow vividly
      let minV = 999999;
      let maxV = -999999;
      for (let i=0; i<BINS; i++) {
        if(smoothedBins[i] < minV) minV = smoothedBins[i];
        if(smoothedBins[i] > maxV) maxV = smoothedBins[i];
      }
      
      const rng = (maxV - minV) > 1 ? (maxV - minV) : 1;
      const normalized = smoothedBins.map(v => (v - minV) / rng);

      heatmap.append(normalized);
    });

    socket.on("prediction", ({ gesture, probabilities, raw_probabilities, fps, debug }) => {
      if (elements.statFps) elements.statFps.textContent = Number(fps || 0).toFixed(1);
      updateGestureDisplay(gesture, probabilities, debug || {}, raw_probabilities || probabilities);
      updateConfidenceBars(probabilities);
    });

    // Attach Event Listeners
    if ($("btnStartLive")) {
      $("btnStartLive").addEventListener("click", () => {
        logEntry("Requesting Live Radar...", "sys");
        socket.emit("start_radar");
      });
    }
    if ($("btnDisconnect")) {
      $("btnDisconnect").addEventListener("click", () => {
        logEntry("Disconnecting...", "warn");
        socket.emit("stop_radar");
      });
    }
    if ($("btnRebaseline")) {
      $("btnRebaseline").addEventListener("click", () => {
        logEntry("Re-baselining...", "warn");
        socket.emit("rebaseline");
      });
    }
    if ($("btnClearLog")) {
      $("btnClearLog").addEventListener("click", () => {
        elements.terminalLog.innerHTML = `<div class="terminal__entry terminal__entry--sys">[SYS ] Log cleared.</div>`;
      });
    }

    buildConfidenceBars();
    setConnectionState(false);
    console.log("[AntRadar] Initialization complete.");

  } catch (err) {
    console.error("[AntRadar] Initialization failed:", err);
    alert("Dashboard Initialization Failed: " + err.message);
  }
}

// CORE RE-IMPLEMENTATIONS
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

function getTopPrediction(probabilities) {
  let topLabel = "none";
  let topValue = -1;
  for (const label of LABELS) {
    const val = Number(probabilities?.[label] || 0);
    if (val > topValue) {
      topValue = val;
      topLabel = label;
    }
  }
  return { topLabel, topConf: Math.max(0, topValue) };
}

function updateGestureDisplay(gesture, probabilities, debug = {}, rawProbabilities = probabilities) {
  const displayTop = getTopPrediction(probabilities);
  const rawTop = getTopPrediction(rawProbabilities);
  const rawTopLabel = debug.raw_top || rawTop.topLabel;
  const rawTopConf = Number(debug.raw_conf ?? rawProbabilities?.[rawTopLabel] ?? rawTop.topConf ?? 0);
  const gateTag = debug.gate ? ` | gate=${debug.gate}` : "";

  // Keep UI strictly synchronized with final server gesture.
  if (gesture !== "none") {
    elements.gestureEmoji.textContent = GESTURE_EMOJIS[gesture] || "❓";
    elements.gestureLabel.textContent = gesture.toUpperCase();
    elements.gestureCard.style.setProperty("--gesture-glow", GESTURE_GLOWS[gesture]);
    elements.gestureStatusPill.className = "pill pill--active";
    elements.gestureStatusPill.textContent = "ACTIVE";
    const outConf = Number(probabilities?.[gesture] ?? displayTop.topConf ?? 0);
    const rawTag = rawTopLabel !== gesture
      ? ` | raw=${rawTopLabel.toUpperCase()} ${Math.round(rawTopConf * 100)}%`
      : "";
    elements.gestureConfidence.textContent = `${Math.round(outConf * 100)}%${gateTag}${rawTag}`;

    if (gesture !== lastGesture) {
      logEntry(`Gesture: ${gesture.toUpperCase()} detected`, "gest");
      lastGesture = gesture;
      gestureEventCount++;
      if (elements.statCount) elements.statCount.textContent = gestureEventCount;
    }
  } else {
    resetGestureToNone();
    const noneConf = Number(probabilities?.none ?? displayTop.topConf ?? 1);
    const rawTag = rawTopLabel !== "none"
      ? ` | raw=${rawTopLabel.toUpperCase()} ${Math.round(rawTopConf * 100)}%`
      : "";
    elements.gestureConfidence.textContent = `NONE ${Math.round(noneConf * 100)}%${gateTag}${rawTag}`;
  }
}

function resetGestureToNone() {
  elements.gestureEmoji.textContent = GESTURE_EMOJIS["none"];
  elements.gestureLabel.textContent = "NONE";
  elements.gestureCard.style.setProperty("--gesture-glow", GESTURE_GLOWS["none"]);
  elements.gestureStatusPill.className = "pill pill--status";
  elements.gestureStatusPill.textContent = "IDLE";
  elements.gestureConfidence.textContent = "Awaiting signal...";
  lastGesture = "none";
}

// Start
document.addEventListener("DOMContentLoaded", init);
