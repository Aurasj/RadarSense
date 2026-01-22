/**
 * Follow the Gesture - Mini Game
 * Simon Says style gesture recognition game
 */

// =============================================================================
// CONFIGURATION
// =============================================================================
const GESTURES = [
    { id: 'hold', name: 'Ține', icon: '🤚', gateRequired: 'hold' },
    { id: 'push', name: 'Împinge', icon: '👆', gateRequired: 'cnn' },
    { id: 'pull', name: 'Trage', icon: '👇', gateRequired: 'cnn' },
    { id: 'tap', name: 'Atingere', icon: '👉', gateRequired: 'cnn' },
    { id: 'wave', name: 'Salut', icon: '👋', gateRequired: 'cnn' }
];

const GESTURE_ICONS = {
    'none': '✋', 'hold': '🤚', 'push': '👆',
    'pull': '👇', 'tap': '👉', 'wave': '👋'
};

const SUCCESS_FRAMES_REQUIRED = 10; // ~0.35s at 30fps
const PROGRESS_RING_CIRCUMFERENCE = 2 * Math.PI * 54; // radius 54

// =============================================================================
// STATE
// =============================================================================
let socket = null;
let gameActive = false;
let currentTarget = null;
let matchFrameCount = 0;
let score = 0;
let streak = 0;
let startTime = null;
let timeInterval = null;
let baselineReady = false;
let showDebug = false;

// Configurable thresholds
let minPmax = 0.45;
let minMargin = 0.15;

// =============================================================================
// DOM ELEMENTS
// =============================================================================
const elements = {};

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initElements();
    initSocket();
    initEventListeners();
    updateProgressRing(0);
});

function initElements() {
    elements.connectionStatus = document.getElementById('connectionStatus');
    elements.startBtn = document.getElementById('startBtn');
    elements.stopBtn = document.getElementById('stopBtn');
    elements.rebaselineBtn = document.getElementById('rebaselineBtn');
    elements.simulateToggle = document.getElementById('simulateToggle');
    elements.debugToggle = document.getElementById('debugToggle');
    elements.debugPanel = document.getElementById('debugPanel');

    elements.scoreValue = document.getElementById('scoreValue');
    elements.streakValue = document.getElementById('streakValue');
    elements.timeValue = document.getElementById('timeValue');

    elements.waitingMessage = document.getElementById('waitingMessage');
    elements.baselineMessage = document.getElementById('baselineMessage');
    elements.targetDisplay = document.getElementById('targetDisplay');
    elements.targetIcon = document.getElementById('targetIcon');
    elements.targetName = document.getElementById('targetName');
    elements.progressFill = document.getElementById('progressFill');
    elements.successFlash = document.getElementById('successFlash');

    elements.detectedIcon = document.getElementById('detectedIcon');
    elements.detectedName = document.getElementById('detectedName');
    elements.gateBadge = document.getElementById('gateBadge');
    elements.pmaxBar = document.getElementById('pmaxBar');
    elements.pmaxText = document.getElementById('pmaxText');

    elements.minPmaxSlider = document.getElementById('minPmaxSlider');
    elements.minPmaxValue = document.getElementById('minPmaxValue');
    elements.minMarginSlider = document.getElementById('minMarginSlider');
    elements.minMarginValue = document.getElementById('minMarginValue');

    // Debug
    elements.dbgGesture = document.getElementById('dbgGesture');
    elements.dbgGate = document.getElementById('dbgGate');
    elements.dbgPmax = document.getElementById('dbgPmax');
    elements.dbgMargin = document.getElementById('dbgMargin');
    elements.dbgLvl = document.getElementById('dbgLvl');
    elements.dbgMot = document.getElementById('dbgMot');
}

function initSocket() {
    socket = io({ transports: ['websocket', 'polling'] });

    socket.on('connect', () => console.log('[Game] Socket connected'));
    socket.on('disconnect', () => {
        console.log('[Game] Socket disconnected');
        updateConnectionStatus(false, false);
    });

    socket.on('status', (data) => {
        updateConnectionStatus(data.connected, data.baseline_ready);
        baselineReady = data.baseline_ready || false;
    });

    socket.on('radar_data', handleRadarData);
}

function initEventListeners() {
    elements.startBtn.addEventListener('click', startRadar);
    elements.stopBtn.addEventListener('click', stopRadar);
    elements.rebaselineBtn.addEventListener('click', () => socket.emit('rebaseline'));
    elements.debugToggle.addEventListener('change', toggleDebug);

    elements.minPmaxSlider.addEventListener('input', (e) => {
        minPmax = parseFloat(e.target.value);
        elements.minPmaxValue.textContent = Math.round(minPmax * 100) + '%';
    });

    elements.minMarginSlider.addEventListener('input', (e) => {
        minMargin = parseFloat(e.target.value);
        elements.minMarginValue.textContent = Math.round(minMargin * 100) + '%';
    });
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

    resetGame();
    showWaiting(false);
    showBaseline(true);
}

function stopRadar() {
    socket.emit('stop_radar');

    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.rebaselineBtn.disabled = true;

    stopGame();
}

function toggleDebug() {
    showDebug = elements.debugToggle.checked;
    elements.debugPanel.style.display = showDebug ? 'flex' : 'none';
}

// =============================================================================
// GAME LOGIC
// =============================================================================
function resetGame() {
    score = 0;
    streak = 0;
    matchFrameCount = 0;
    currentTarget = null;
    gameActive = false;
    baselineReady = false;

    updateScoreDisplay();

    if (timeInterval) clearInterval(timeInterval);
    startTime = null;
    elements.timeValue.textContent = '0:00';
}

function startGame() {
    gameActive = true;
    startTime = Date.now();

    timeInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        elements.timeValue.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
    }, 1000);

    pickNewTarget();
    showTarget(true);
}

function stopGame() {
    gameActive = false;
    if (timeInterval) clearInterval(timeInterval);
    showWaiting(true);
    showBaseline(false);
    showTarget(false);
}

function pickNewTarget() {
    // Don't pick the same gesture twice in a row
    let newTarget;
    do {
        newTarget = GESTURES[Math.floor(Math.random() * GESTURES.length)];
    } while (currentTarget && newTarget.id === currentTarget.id);

    currentTarget = newTarget;
    matchFrameCount = 0;

    elements.targetIcon.textContent = currentTarget.icon;
    elements.targetName.textContent = currentTarget.name;
    updateProgressRing(0);
}

function handleSuccess() {
    score++;
    streak++;
    updateScoreDisplay();

    // Flash effect
    elements.successFlash.classList.add('show');
    setTimeout(() => elements.successFlash.classList.remove('show'), 400);

    // Play sound (optional - create audio element)
    try {
        const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YVoGAACAgICAgICAgIB/f35+fX19fHx8e3t6enl5eHh3d3Z2dXV0dHNzc3JycnFxcXBwcG9vb29ub25tbm1tbGxsbGtra2tqampqaWlpaWhoaGhnZ2dnZmdmZmZlZWVlZGRkZGNjY2NiYmJiYWFhYWBgYGBfX19fXl5eXl1dXV1cXFxcW1tbW1paWlpZWVlZWFhYWFdXV1dWVlZWVVVVVVRUVFRTU1NTU1JSUlJSUVFRUVBQUFBPT09PT05OTk5NTU1NTExMTExLS0tLS0pKSkpJSUlJSEhISEdHR0dHRkZGRkVFRUVEREREQ0NDQ0NCQkJCQUFBQUBAQEA/Pz8/Pj4+Pj09PT08PDw8Ozs7Ozs6Ojo6OTk5OTg4ODg3Nzc3NjY2NjU1NTU0NDQ0MzMzMzIyMjIxMTExMDAwMC8vLy8uLi4uLS0tLS0sLCwsKysrKyoqKioqKSkpKSgoKCgnJycnJyYmJiYlJSUlJCQkJCMjIyMjIiIiIiEhISEgICAgHx8fHx8eHh4eHR0dHRwcHBwcGxsbGxoaGhoZGRkZGBgYGBgXFxcXFhYWFhUVFRUUFBQUFBMTExMSEhISERERERERDw8PDw4ODg4NDQ0NDA');
        audio.volume = 0.3;
        audio.play().catch(() => { });
    } catch (e) { }

    // Small delay before next target
    setTimeout(() => {
        if (gameActive) pickNewTarget();
    }, 300);
}

function handleMiss() {
    streak = 0;
    updateScoreDisplay();
}

function updateScoreDisplay() {
    elements.scoreValue.textContent = score;
    elements.streakValue.textContent = streak;
}

// =============================================================================
// DATA HANDLING
// =============================================================================
function handleRadarData(data) {
    // Update connection status
    updateConnectionStatus(data.connected, data.baseline_ready);

    // Handle baseline transition
    if (!baselineReady && data.baseline_ready) {
        baselineReady = true;
        showBaseline(false);
        startGame();
    }

    // Update detection display
    updateDetectionDisplay(data);

    // Update debug
    if (showDebug) updateDebug(data);

    // Game logic
    if (!gameActive || !currentTarget) return;

    const gestureOut = data.gesture_out;
    const gate = data.gate;
    const pmax = data.pmax || 0;
    const margin = data.top2_margin || 0;

    // Check if detected gesture matches target
    const isCorrectGesture = gestureOut === currentTarget.id;
    const isCorrectGate = gate === currentTarget.gateRequired;
    const meetsThresholds = pmax >= minPmax && margin >= minMargin;

    // For hold, we use gate='hold', for others gate='cnn'
    const isValidDetection = isCorrectGesture && isCorrectGate &&
        (currentTarget.gateRequired === 'hold' || meetsThresholds);

    if (isValidDetection) {
        matchFrameCount++;
        updateProgressRing(matchFrameCount / SUCCESS_FRAMES_REQUIRED);

        if (matchFrameCount >= SUCCESS_FRAMES_REQUIRED) {
            handleSuccess();
        }
    } else {
        if (matchFrameCount > 0) {
            matchFrameCount = Math.max(0, matchFrameCount - 2); // Decay slowly
            updateProgressRing(matchFrameCount / SUCCESS_FRAMES_REQUIRED);
        }
    }
}

// =============================================================================
// UI UPDATES
// =============================================================================
function updateConnectionStatus(connected, baseline) {
    if (connected) {
        elements.connectionStatus.classList.add('connected');
        elements.connectionStatus.querySelector('.status-text').textContent =
            baseline ? 'Ready' : 'Conectat';
    } else {
        elements.connectionStatus.classList.remove('connected');
        elements.connectionStatus.querySelector('.status-text').textContent = 'Deconectat';
    }
}

function updateDetectionDisplay(data) {
    const gesture = data.gesture_out || 'none';
    const gate = data.gate || '-';
    const pmax = data.pmax || 0;

    elements.detectedIcon.textContent = GESTURE_ICONS[gesture] || '?';
    elements.detectedName.textContent = gesture;
    elements.gateBadge.textContent = gate;
    elements.gateBadge.className = 'gate-badge gate-' + gate;

    const pmaxPercent = Math.round(pmax * 100);
    elements.pmaxBar.style.width = pmaxPercent + '%';
    elements.pmaxText.textContent = pmaxPercent + '%';
}

function updateDebug(data) {
    elements.dbgGesture.textContent = data.gesture_out || '-';
    elements.dbgGate.textContent = data.gate || '-';
    elements.dbgPmax.textContent = (data.pmax || 0).toFixed(2);
    elements.dbgMargin.textContent = (data.top2_margin || 0).toFixed(2);
    elements.dbgLvl.textContent = (data.lvl || 0).toFixed(1);
    elements.dbgMot.textContent = (data.mot || 0).toFixed(3);
}

function updateProgressRing(progress) {
    const offset = PROGRESS_RING_CIRCUMFERENCE * (1 - Math.min(1, progress));
    elements.progressFill.style.strokeDashoffset = offset;

    // Color based on progress
    if (progress >= 1) {
        elements.progressFill.style.stroke = '#10b981';
    } else if (progress > 0.5) {
        elements.progressFill.style.stroke = '#eab308';
    } else {
        elements.progressFill.style.stroke = '#6366f1';
    }
}

function showWaiting(show) {
    elements.waitingMessage.style.display = show ? 'block' : 'none';
}

function showBaseline(show) {
    elements.baselineMessage.style.display = show ? 'flex' : 'none';
}

function showTarget(show) {
    elements.targetDisplay.style.display = show ? 'flex' : 'none';
}
