/**
 * Gesture Radar - Production Frontend
 * Enhanced UI with OUT vs RAW, debug panel, and proper visualization
 */

// =============================================================================
// CONFIGURATION
// =============================================================================
const CONFIG = {
    HISTORY_LENGTH: 100,
    HEATMAP_PERCENTILE_LOW: 5,
    HEATMAP_PERCENTILE_HIGH: 95,
};

const GESTURE_ICONS = {
    'none': '✋', 'hold': '🤚', 'push': '👆',
    'pull': '👇', 'tap': '👉', 'wave': '👋'
};

const GESTURE_NAMES = {
    'none': 'Nimic', 'hold': 'Ține', 'push': 'Împinge',
    'pull': 'Trage', 'tap': 'Atingere', 'wave': 'Salut'
};

const GATE_NAMES = {
    'none': '🔴 None', 'hold': '🟡 Hold', 'cnn': '🟢 CNN', 'baseline': '⏳ Baseline'
};

// =============================================================================
// STATE
// =============================================================================
let socket = null;
let radarHistory = [];
let frameCount = 0;
let lastFrameTime = Date.now();
let clientFps = 0;
let isConnected = false;
let showDebug = false;

// Heatmap normalization (rolling)
let heatmapMin = Infinity;
let heatmapMax = -Infinity;
let normalizationBuffer = [];

// Range config (from server)
let rangeStart = 0.20;
let rangeEnd = 0.55;

// =============================================================================
// DOM ELEMENTS
// =============================================================================
const elements = {};

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    initializeCanvas();
    initializeSocket();
    initializeEventListeners();
    startRenderLoop();
});

function initializeElements() {
    elements.canvas = document.getElementById('radarCanvas');
    elements.ctx = elements.canvas.getContext('2d');
    elements.connectionStatus = document.getElementById('connectionStatus');
    elements.startBtn = document.getElementById('startBtn');
    elements.stopBtn = document.getElementById('stopBtn');
    elements.rebaselineBtn = document.getElementById('rebaselineBtn');
    elements.simulateToggle = document.getElementById('simulateToggle');
    elements.debugToggle = document.getElementById('debugToggle');

    // Gesture displays
    elements.gestureOutIcon = document.getElementById('gestureOutIcon');
    elements.gestureOutName = document.getElementById('gestureOutName');
    elements.gestureOutConfidence = document.getElementById('gestureOutConfidence');
    elements.gestureOutCard = document.getElementById('gestureOutCard');

    elements.gestureRawIcon = document.getElementById('gestureRawIcon');
    elements.gestureRawName = document.getElementById('gestureRawName');
    elements.gateIndicator = document.getElementById('gateIndicator');

    // Stats
    elements.fpsValue = document.getElementById('fpsValue');
    elements.modeValue = document.getElementById('modeValue');
    elements.gateValue = document.getElementById('gateValue');

    // Debug panel
    elements.debugPanel = document.getElementById('debugPanel');
    elements.lvlValue = document.getElementById('lvlValue');
    elements.motValue = document.getElementById('motValue');
    elements.levelThValue = document.getElementById('levelThValue');
    elements.holdThValue = document.getElementById('holdThValue');
    elements.motionThValue = document.getElementById('motionThValue');
    elements.motionStateValue = document.getElementById('motionStateValue');
    elements.holdStateValue = document.getElementById('holdStateValue');

    // Top-3 bars
    elements.top3Container = document.getElementById('top3Container');

    // Gesture items
    elements.gestureItems = document.querySelectorAll('.gesture-item');
}

function initializeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = elements.canvas.getBoundingClientRect();
    elements.canvas.width = rect.width * dpr;
    elements.canvas.height = rect.height * dpr;
    elements.ctx.scale(dpr, dpr);
    elements.canvas.style.width = rect.width + 'px';
    elements.canvas.style.height = rect.height + 'px';
    drawRadar([]);
}

function initializeSocket() {
    socket = io({ transports: ['websocket', 'polling'] });

    socket.on('connect', () => console.log('[Socket] Connected'));
    socket.on('disconnect', () => {
        console.log('[Socket] Disconnected');
        updateConnectionStatus(false);
    });

    socket.on('status', (data) => {
        console.log('[Socket] Status:', data);
        updateConnectionStatus(data.connected);
        if (data.mode) {
            elements.modeValue.textContent = data.mode.includes('simulation') ? 'Sim' : 'Live';
        }
    });

    socket.on('radar_data', handleRadarData);
}

function initializeEventListeners() {
    elements.startBtn.addEventListener('click', startRadar);
    elements.stopBtn.addEventListener('click', stopRadar);
    elements.rebaselineBtn.addEventListener('click', rebaseline);
    elements.debugToggle.addEventListener('change', toggleDebug);
    window.addEventListener('resize', initializeCanvas);
}

// =============================================================================
// RADAR CONTROL
// =============================================================================
function startRadar() {
    const simulate = elements.simulateToggle.checked;
    socket.emit('start_radar', { simulate });
    elements.startBtn.disabled = true;
    elements.stopBtn.disabled = false;
    elements.rebaselineBtn.disabled = false;

    // Reset visualization
    radarHistory = [];
    heatmapMin = Infinity;
    heatmapMax = -Infinity;
    normalizationBuffer = [];
}

function stopRadar() {
    socket.emit('stop_radar');
    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.rebaselineBtn.disabled = true;
    updateConnectionStatus(false);
    radarHistory = [];
    drawRadar([]);
}

function rebaseline() {
    socket.emit('rebaseline');
    console.log('[UI] Requested re-baseline');
}

function toggleDebug() {
    showDebug = elements.debugToggle.checked;
    elements.debugPanel.style.display = showDebug ? 'block' : 'none';
}

// =============================================================================
// DATA HANDLING
// =============================================================================
function handleRadarData(data) {
    frameCount++;

    // Client-side FPS
    const now = Date.now();
    if (now - lastFrameTime >= 1000) {
        clientFps = Math.round((frameCount * 1000) / (now - lastFrameTime));
        frameCount = 0;
        lastFrameTime = now;
    }

    // Update range from server
    if (data.range_start !== undefined) rangeStart = data.range_start;
    if (data.range_end !== undefined) rangeEnd = data.range_end;

    // Add frame to history
    if (data.frame && data.frame.length > 0) {
        radarHistory.push(data.frame);
        if (radarHistory.length > CONFIG.HISTORY_LENGTH) {
            radarHistory.shift();
        }

        // Update normalization buffer
        updateNormalization(data.frame);
    }

    // Update UI
    updateConnectionStatus(data.connected);
    updateGestureDisplay(data);
    updateStats(data);
    updateDebugPanel(data);
    updateTop3(data.all_confidences);
    updateGestureGrid(data.gesture_out, data.pmax);
}

function updateNormalization(frame) {
    // Use rolling percentile for better heatmap
    normalizationBuffer.push(...frame);
    if (normalizationBuffer.length > 5000) {
        normalizationBuffer = normalizationBuffer.slice(-3000);
    }

    if (normalizationBuffer.length > 100) {
        const sorted = [...normalizationBuffer].sort((a, b) => a - b);
        const lowIdx = Math.floor(sorted.length * CONFIG.HEATMAP_PERCENTILE_LOW / 100);
        const highIdx = Math.floor(sorted.length * CONFIG.HEATMAP_PERCENTILE_HIGH / 100);
        heatmapMin = sorted[lowIdx];
        heatmapMax = sorted[highIdx];
    }
}

// =============================================================================
// UI UPDATES
// =============================================================================
function updateConnectionStatus(connected) {
    isConnected = connected;
    if (connected) {
        elements.connectionStatus.classList.add('connected');
        elements.connectionStatus.querySelector('.status-text').textContent = 'Conectat';
    } else {
        elements.connectionStatus.classList.remove('connected');
        elements.connectionStatus.querySelector('.status-text').textContent = 'Deconectat';
    }
}

function updateGestureDisplay(data) {
    const gestureOut = data.gesture_out || 'none';
    const gestureRaw = data.gesture_raw;
    const gate = data.gate || 'none';
    const pmax = data.pmax || 0;

    // OUT gesture (final output)
    elements.gestureOutIcon.textContent = GESTURE_ICONS[gestureOut] || '❓';
    elements.gestureOutName.textContent = GESTURE_NAMES[gestureOut] || gestureOut;

    const confPercent = Math.round(pmax * 100);
    elements.gestureOutConfidence.textContent = gate === 'cnn' ? `${confPercent}%` : '-';

    // Active state
    if (gestureOut !== 'none' && (pmax > 0.4 || gate !== 'cnn')) {
        elements.gestureOutCard.classList.add('active');
    } else {
        elements.gestureOutCard.classList.remove('active');
    }

    // RAW gesture (CNN output, only when gate=cnn)
    if (gate === 'cnn' && gestureRaw) {
        elements.gestureRawIcon.textContent = GESTURE_ICONS[gestureRaw] || '❓';
        elements.gestureRawName.textContent = GESTURE_NAMES[gestureRaw] || gestureRaw;
    } else {
        elements.gestureRawIcon.textContent = '—';
        elements.gestureRawName.textContent = 'CNN off';
    }

    // Gate indicator
    elements.gateIndicator.textContent = GATE_NAMES[gate] || gate;
    elements.gateIndicator.className = 'gate-indicator gate-' + gate;
}

function updateStats(data) {
    elements.fpsValue.textContent = data.fps || 0;
    elements.gateValue.textContent = data.gate || 'none';
}

function updateDebugPanel(data) {
    if (!showDebug) return;

    elements.lvlValue.textContent = (data.lvl || 0).toFixed(2);
    elements.motValue.textContent = (data.mot || 0).toFixed(4);
    elements.levelThValue.textContent = (data.level_th || 0).toFixed(2);
    elements.holdThValue.textContent = (data.hold_th || 0).toFixed(2);
    elements.motionThValue.textContent = (data.motion_th || 0).toFixed(4);
    elements.motionStateValue.textContent = data.motion_state ? '🟢 Motion' : '⚪ Still';
    elements.holdStateValue.textContent = data.hold_state ? '🟡 Holding' : '⚪ Empty';
}

function updateTop3(allConfidences) {
    if (!allConfidences || Object.keys(allConfidences).length === 0) {
        elements.top3Container.innerHTML = '<div class="top3-empty">No CNN data</div>';
        return;
    }

    // Sort by confidence
    const sorted = Object.entries(allConfidences)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);

    let html = '';
    sorted.forEach(([label, conf], idx) => {
        const percent = Math.round(conf * 100);
        const barWidth = Math.max(percent, 2);
        const icon = GESTURE_ICONS[label] || '?';
        const name = GESTURE_NAMES[label] || label;

        html += `
            <div class="top3-item ${idx === 0 ? 'top3-winner' : ''}">
                <span class="top3-icon">${icon}</span>
                <span class="top3-label">${name}</span>
                <div class="top3-bar-container">
                    <div class="top3-bar" style="width: ${barWidth}%"></div>
                </div>
                <span class="top3-percent">${percent}%</span>
            </div>
        `;
    });

    elements.top3Container.innerHTML = html;
}

function updateGestureGrid(gesture, confidence) {
    elements.gestureItems.forEach(item => {
        const itemGesture = item.dataset.gesture;
        if (itemGesture === gesture && (confidence > 0.35 || gesture === 'none' || gesture === 'hold')) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}

// =============================================================================
// RADAR VISUALIZATION
// =============================================================================
function startRenderLoop() {
    function render() {
        drawRadar(radarHistory);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
}

function drawRadar(history) {
    const ctx = elements.ctx;
    const width = elements.canvas.clientWidth;
    const height = elements.canvas.clientHeight;

    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, width, height);

    if (history.length === 0) {
        ctx.fillStyle = 'rgba(100, 100, 150, 0.5)';
        ctx.font = '16px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Apasă Start pentru a începe', width / 2, height / 2);
        return;
    }

    drawGrid(ctx, width, height);

    const numFrames = history.length;
    const numBins = history[0].length;
    const cellWidth = width / numFrames;
    const cellHeight = height / numBins;

    // Use percentile-based normalization
    const range = Math.max(heatmapMax - heatmapMin, 1);

    // Find peak in latest frame
    let peakIdx = 0;
    let peakVal = -Infinity;
    const latestFrame = history[history.length - 1];

    // Draw heatmap
    for (let i = 0; i < numFrames; i++) {
        const frame = history[i];
        const x = i * cellWidth;

        for (let j = 0; j < numBins; j++) {
            const y = j * cellHeight;
            const value = frame[j];

            // Track peak in latest frame
            if (i === numFrames - 1 && value > peakVal) {
                peakVal = value;
                peakIdx = j;
            }

            const normalized = (value - heatmapMin) / range;
            const intensity = Math.min(1, Math.max(0, normalized));
            const color = getHeatmapColor(intensity);

            ctx.fillStyle = `rgb(${color.r}, ${color.g}, ${color.b})`;
            ctx.fillRect(x, y, Math.ceil(cellWidth) + 1, Math.ceil(cellHeight) + 1);
        }
    }

    // Draw peak distance line
    if (peakVal > heatmapMin + range * 0.3) {
        const peakY = (peakIdx / numBins) * height;
        const peakDistance = rangeStart + (peakIdx / numBins) * (rangeEnd - rangeStart);

        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(0, peakY);
        ctx.lineTo(width, peakY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#22d3ee';
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${(peakDistance * 100).toFixed(0)} cm`, 8, peakY - 5);
    }

    // Draw distance labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';

    const numLabels = 4;
    for (let i = 0; i <= numLabels; i++) {
        const y = (height / numLabels) * i;
        const distance = rangeStart + (i / numLabels) * (rangeEnd - rangeStart);
        ctx.fillText(`${(distance * 100).toFixed(0)}`, width - 3, y + 10);
    }

    // Time indicator
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.fillRect(width - 2, 0, 2, height);
}

function drawGrid(ctx, width, height) {
    ctx.strokeStyle = 'rgba(100, 100, 150, 0.2)';
    ctx.lineWidth = 1;

    for (let i = 0; i <= 4; i++) {
        const y = (height / 4) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }

    for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }
}

function getHeatmapColor(intensity) {
    if (intensity < 0.25) {
        const t = intensity / 0.25;
        return { r: Math.round(10 + t * 24), g: Math.round(10 + t * 50), b: Math.round(30 + t * 80) };
    } else if (intensity < 0.5) {
        const t = (intensity - 0.25) / 0.25;
        return { r: Math.round(34), g: Math.round(60 + t * 151), b: Math.round(110 + t * 128) };
    } else if (intensity < 0.75) {
        const t = (intensity - 0.5) / 0.25;
        return { r: Math.round(34 + t * 65), g: Math.round(211 - t * 109), b: Math.round(238) };
    } else {
        const t = (intensity - 0.75) / 0.25;
        return { r: Math.round(99 + t * 157), g: Math.round(102 - t * 30), b: Math.round(238 - t * 85) };
    }
}
