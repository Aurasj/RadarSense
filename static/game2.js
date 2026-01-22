/**
 * Balloon Garden - Natural Gesture Game
 * Push=lift, Pull=lower, Wave=sweep, Tap=pop, Hold=slow-mo
 */

// =============================================================================
// CONFIGURATION
// =============================================================================
const CONFIG = {
    GAME_DURATION: 60,
    BASE_SPAWN_RATE: 1500,
    MIN_SPAWN_RATE: 400,
    BALLOON_MIN_SIZE: 30,
    BALLOON_MAX_SIZE: 60,
    PUSH_FORCE: -3,
    PULL_FORCE: 2,
    WAVE_FORCE: 8,
    HOLD_SLOW_FACTOR: 0.2,
    TAP_COOLDOWN: 400,
    WAVE_COOLDOWN: 400,
    POP_ZONE_RADIUS: 100,
    LEVEL_UP_SCORE: 10,
};

const GESTURE_ICONS = {
    'none': '✋', 'hold': '🤚', 'push': '👆',
    'pull': '👇', 'tap': '👉', 'wave': '👋'
};

const BALLOON_COLORS = [
    '#ef4444', '#f97316', '#eab308', '#22c55e',
    '#06b6d4', '#6366f1', '#a855f7', '#ec4899'
];

// =============================================================================
// STATE
// =============================================================================
let socket = null;
let canvas, ctx;
let gameActive = false;
let gamePaused = false;
let balloons = [];
let particles = [];
let score = 0;
let combo = 1;
let timeLeft = CONFIG.GAME_DURATION;
let level = 1;
let lastSpawn = 0;
let lastTap = 0;
let lastWave = 0;
let isHolding = false;
let showDebug = false;
let baselineReady = false;

// Thresholds
let minConf = 0.45;
let minMargin = 0.15;

// Current gesture state
let currentGesture = 'none';
let currentGate = 'none';
let currentPmax = 0;
let currentMargin = 0;

// Animation
let lastFrameTime = 0;
let gameFps = 0;
let frameCount = 0;
let fpsTime = 0;

// =============================================================================
// DOM ELEMENTS
// =============================================================================
const elements = {};

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initElements();
    initCanvas();
    initSocket();
    initEventListeners();
    requestAnimationFrame(gameLoop);
});

function initElements() {
    canvas = document.getElementById('gameCanvas');
    ctx = canvas.getContext('2d');

    elements.connectionStatus = document.getElementById('connectionStatus');
    elements.startBtn = document.getElementById('startBtn');
    elements.stopBtn = document.getElementById('stopBtn');
    elements.rebaselineBtn = document.getElementById('rebaselineBtn');
    elements.simulateToggle = document.getElementById('simulateToggle');
    elements.debugToggle = document.getElementById('debugToggle');
    elements.debugPanel = document.getElementById('debugPanel');

    elements.scoreValue = document.getElementById('scoreValue');
    elements.comboValue = document.getElementById('comboValue');
    elements.timeValue = document.getElementById('timeValue');

    elements.gestureIcon = document.getElementById('gestureIcon');
    elements.gestureName = document.getElementById('gestureName');
    elements.gestureIndicator = document.getElementById('gestureIndicator');

    elements.startOverlay = document.getElementById('startOverlay');
    elements.effectOverlay = document.getElementById('effectOverlay');

    elements.minConfSlider = document.getElementById('minConfSlider');
    elements.minConfValue = document.getElementById('minConfValue');
    elements.minMarginSlider = document.getElementById('minMarginSlider');
    elements.minMarginValue = document.getElementById('minMarginValue');

    elements.dbgGesture = document.getElementById('dbgGesture');
    elements.dbgGate = document.getElementById('dbgGate');
    elements.dbgPmax = document.getElementById('dbgPmax');
    elements.dbgMargin = document.getElementById('dbgMargin');
    elements.dbgFps = document.getElementById('dbgFps');
}

function initCanvas() {
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
}

function resizeCanvas() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
}

function initSocket() {
    socket = io({ transports: ['websocket', 'polling'] });

    socket.on('connect', () => console.log('[Game2] Connected'));
    socket.on('disconnect', () => updateConnectionStatus(false, false));
    socket.on('status', (data) => {
        updateConnectionStatus(data.connected, data.baseline_ready);
        if (data.baseline_ready) baselineReady = true;
    });
    socket.on('radar_data', handleRadarData);
}

function initEventListeners() {
    elements.startBtn.addEventListener('click', startRadar);
    elements.stopBtn.addEventListener('click', stopRadar);
    elements.rebaselineBtn.addEventListener('click', () => socket.emit('rebaseline'));
    elements.debugToggle.addEventListener('change', () => {
        showDebug = elements.debugToggle.checked;
        elements.debugPanel.style.display = showDebug ? 'flex' : 'none';
    });

    elements.minConfSlider.addEventListener('input', (e) => {
        minConf = parseFloat(e.target.value);
        elements.minConfValue.textContent = Math.round(minConf * 100) + '%';
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
    elements.startOverlay.style.display = 'flex';
    elements.startOverlay.innerHTML = '<div class="spinner"></div><p>Calibrare senzor...</p>';
}

function stopRadar() {
    socket.emit('stop_radar');

    elements.startBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.rebaselineBtn.disabled = true;

    gameActive = false;
    showStartOverlay();
}

// =============================================================================
// GAME LOGIC
// =============================================================================
function resetGame() {
    balloons = [];
    particles = [];
    score = 0;
    combo = 1;
    timeLeft = CONFIG.GAME_DURATION;
    level = 1;
    lastSpawn = 0;
    lastTap = 0;
    lastWave = 0;
    isHolding = false;
    gameActive = false;
    baselineReady = false;
    updateHUD();
}

function startGame() {
    gameActive = true;
    elements.startOverlay.style.display = 'none';

    // Start timer
    const timerInterval = setInterval(() => {
        if (!gameActive) {
            clearInterval(timerInterval);
            return;
        }
        timeLeft--;
        elements.timeValue.textContent = timeLeft;

        if (timeLeft <= 0) {
            clearInterval(timerInterval);
            endGame();
        }
    }, 1000);
}

function endGame() {
    gameActive = false;
    elements.startOverlay.style.display = 'flex';
    elements.startOverlay.innerHTML = `
        <h2>🎉 Game Over!</h2>
        <p class="final-score">Score: <b>${score}</b></p>
        <p>Level reached: ${level}</p>
        <p class="start-hint">Apasă Start pentru a juca din nou</p>
    `;
}

function showStartOverlay() {
    elements.startOverlay.style.display = 'flex';
    elements.startOverlay.innerHTML = `
        <h2>🎈 Balloon Garden</h2>
        <p>Folosește gesturi naturale pentru a controla baloanele!</p>
        <div class="gesture-legend">
            <div><span>👆 Push</span> = Ridică</div>
            <div><span>👇 Pull</span> = Coboară</div>
            <div><span>👉 Tap</span> = Sparge</div>
            <div><span>👋 Wave</span> = Împrăștie</div>
            <div><span>🤚 Hold</span> = Slow-mo</div>
        </div>
        <p class="start-hint">Apasă Start și așteaptă calibrarea</p>
    `;
}

// =============================================================================
// BALLOON CLASS
// =============================================================================
class Balloon {
    constructor() {
        this.size = CONFIG.BALLOON_MIN_SIZE + Math.random() * (CONFIG.BALLOON_MAX_SIZE - CONFIG.BALLOON_MIN_SIZE);
        this.x = this.size + Math.random() * (canvas.width - this.size * 2);
        this.y = canvas.height + this.size;
        this.vx = (Math.random() - 0.5) * 1;
        this.vy = -0.5 - Math.random() * 1.5 - level * 0.2;
        this.color = BALLOON_COLORS[Math.floor(Math.random() * BALLOON_COLORS.length)];
        this.wobble = Math.random() * Math.PI * 2;
        this.wobbleSpeed = 0.02 + Math.random() * 0.02;
        this.popped = false;
        this.points = Math.ceil((CONFIG.BALLOON_MAX_SIZE - this.size) / 10) + 1;
    }

    update(dt, slowFactor = 1) {
        const speed = slowFactor;
        this.wobble += this.wobbleSpeed * speed;
        this.x += (this.vx + Math.sin(this.wobble) * 0.5) * speed;
        this.y += this.vy * speed;

        // Bounce off walls
        if (this.x < this.size || this.x > canvas.width - this.size) {
            this.vx *= -0.8;
            this.x = Math.max(this.size, Math.min(canvas.width - this.size, this.x));
        }
    }

    draw(ctx) {
        // Shadow
        ctx.beginPath();
        ctx.ellipse(this.x, this.y + this.size * 0.8, this.size * 0.4, this.size * 0.15, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0,0,0,0.15)';
        ctx.fill();

        // Balloon body
        ctx.beginPath();
        ctx.ellipse(this.x, this.y, this.size * 0.9, this.size, 0, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();

        // Highlight
        ctx.beginPath();
        ctx.ellipse(this.x - this.size * 0.3, this.y - this.size * 0.4, this.size * 0.2, this.size * 0.3, -0.5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255,255,255,0.4)';
        ctx.fill();

        // String
        ctx.beginPath();
        ctx.moveTo(this.x, this.y + this.size);
        ctx.quadraticCurveTo(this.x + Math.sin(this.wobble) * 5, this.y + this.size + 15, this.x, this.y + this.size + 25);
        ctx.strokeStyle = '#666';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    isInPopZone() {
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const dist = Math.hypot(this.x - cx, this.y - cy);
        return dist < CONFIG.POP_ZONE_RADIUS + this.size;
    }

    isOffScreen() {
        return this.y < -this.size * 2 || this.y > canvas.height + this.size * 2;
    }
}

// =============================================================================
// PARTICLE CLASS
// =============================================================================
class Particle {
    constructor(x, y, color) {
        this.x = x;
        this.y = y;
        this.vx = (Math.random() - 0.5) * 8;
        this.vy = (Math.random() - 0.5) * 8;
        this.size = 3 + Math.random() * 5;
        this.color = color;
        this.life = 1;
        this.decay = 0.02 + Math.random() * 0.02;
    }

    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vy += 0.2;
        this.life -= this.decay;
    }

    draw(ctx) {
        ctx.globalAlpha = this.life;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = this.color;
        ctx.fill();
        ctx.globalAlpha = 1;
    }
}

// =============================================================================
// GESTURE HANDLING
// =============================================================================
function handleRadarData(data) {
    updateConnectionStatus(data.connected, data.baseline_ready);

    // Handle baseline transition
    if (!baselineReady && data.baseline_ready) {
        baselineReady = true;
        startGame();
    }

    // Store gesture state
    currentGesture = data.gesture_out || 'none';
    currentGate = data.gate || 'none';
    currentPmax = data.pmax || 0;
    currentMargin = data.top2_margin || 0;

    // Update gesture indicator
    elements.gestureIcon.textContent = GESTURE_ICONS[currentGesture] || '?';
    elements.gestureName.textContent = currentGesture;
    elements.gestureIndicator.className = 'gesture-indicator gesture-' + currentGesture;

    // Update debug
    if (showDebug) {
        elements.dbgGesture.textContent = currentGesture;
        elements.dbgGate.textContent = currentGate;
        elements.dbgPmax.textContent = currentPmax.toFixed(2);
        elements.dbgMargin.textContent = currentMargin.toFixed(2);
    }

    if (!gameActive) return;

    // Process gestures
    processGestures();
}

function processGestures() {
    const now = Date.now();
    const validCNN = currentGate === 'cnn' && currentPmax >= minConf && currentMargin >= minMargin;
    const validHold = currentGate === 'hold' || (currentGesture === 'hold');

    // HOLD = slow motion
    isHolding = validHold && currentGesture === 'hold';

    // PUSH = lift balloons
    if (currentGesture === 'push' && validCNN) {
        applyForceToAll(0, CONFIG.PUSH_FORCE);
        showEffect('push');
    }

    // PULL = lower balloons
    if (currentGesture === 'pull' && validCNN) {
        applyForceToAll(0, CONFIG.PULL_FORCE);
        showEffect('pull');
    }

    // TAP = pop balloon in center (with cooldown)
    if (currentGesture === 'tap' && validCNN && now - lastTap > CONFIG.TAP_COOLDOWN) {
        if (popBalloonInZone()) {
            lastTap = now;
            showEffect('tap');
        }
    }

    // WAVE = sweep all balloons away (with cooldown)
    if (currentGesture === 'wave' && validCNN && now - lastWave > CONFIG.WAVE_COOLDOWN) {
        sweepBalloons();
        lastWave = now;
        showEffect('wave');
    }
}

function applyForceToAll(fx, fy) {
    balloons.forEach(b => {
        b.vy += fy * 0.5;
        b.vx += fx * 0.5;
    });
}

function popBalloonInZone() {
    // Find balloon closest to center that's in pop zone
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    let closest = null;
    let closestDist = Infinity;

    for (const b of balloons) {
        if (b.isInPopZone()) {
            const dist = Math.hypot(b.x - cx, b.y - cy);
            if (dist < closestDist) {
                closestDist = dist;
                closest = b;
            }
        }
    }

    if (closest) {
        popBalloon(closest);
        return true;
    }
    return false;
}

function popBalloon(balloon) {
    // Create particles
    for (let i = 0; i < 12; i++) {
        particles.push(new Particle(balloon.x, balloon.y, balloon.color));
    }

    // Score
    const points = balloon.points * combo;
    score += points;
    combo = Math.min(combo + 1, 10);

    // Level up
    if (score >= level * CONFIG.LEVEL_UP_SCORE) {
        level++;
    }

    // Remove balloon
    balloons = balloons.filter(b => b !== balloon);
    updateHUD();
}

function sweepBalloons() {
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    balloons.forEach(b => {
        const dx = b.x - cx;
        const dy = b.y - cy;
        const dist = Math.hypot(dx, dy) || 1;
        b.vx += (dx / dist) * CONFIG.WAVE_FORCE;
        b.vy += (dy / dist) * CONFIG.WAVE_FORCE * 0.5;
    });
}

function showEffect(type) {
    elements.effectOverlay.className = 'effect-overlay effect-' + type;
    setTimeout(() => {
        elements.effectOverlay.className = 'effect-overlay';
    }, 200);
}

// =============================================================================
// UI UPDATES
// =============================================================================
function updateConnectionStatus(connected, baseline) {
    if (connected) {
        elements.connectionStatus.classList.add('connected');
        elements.connectionStatus.querySelector('.status-text').textContent =
            baseline ? 'Ready' : 'Connected';
    } else {
        elements.connectionStatus.classList.remove('connected');
        elements.connectionStatus.querySelector('.status-text').textContent = 'Offline';
    }
}

function updateHUD() {
    elements.scoreValue.textContent = score;
    elements.comboValue.textContent = 'x' + combo;
}

// =============================================================================
// GAME LOOP
// =============================================================================
function gameLoop(timestamp) {
    const dt = timestamp - lastFrameTime;
    lastFrameTime = timestamp;

    // FPS counter
    frameCount++;
    if (timestamp - fpsTime >= 1000) {
        gameFps = frameCount;
        frameCount = 0;
        fpsTime = timestamp;
        if (showDebug) elements.dbgFps.textContent = gameFps;
    }

    // Clear
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (gameActive) {
        const speedFactor = isHolding ? CONFIG.HOLD_SLOW_FACTOR : 1;

        // Spawn balloons
        const spawnRate = Math.max(CONFIG.MIN_SPAWN_RATE, CONFIG.BASE_SPAWN_RATE - level * 100);
        if (timestamp - lastSpawn > spawnRate / speedFactor) {
            balloons.push(new Balloon());
            lastSpawn = timestamp;
        }

        // Update balloons
        balloons.forEach(b => b.update(dt, speedFactor));

        // Remove off-screen balloons (combo reset)
        const offScreen = balloons.filter(b => b.isOffScreen());
        if (offScreen.length > 0) {
            combo = 1;
            updateHUD();
        }
        balloons = balloons.filter(b => !b.isOffScreen());

        // Update particles
        particles.forEach(p => p.update());
        particles = particles.filter(p => p.life > 0);

        // Draw pop zone indicator
        drawPopZone();

        // Draw hold effect
        if (isHolding) {
            ctx.fillStyle = 'rgba(139, 92, 246, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
    }

    // Draw balloons
    balloons.forEach(b => b.draw(ctx));

    // Draw particles
    particles.forEach(p => p.draw(ctx));

    requestAnimationFrame(gameLoop);
}

function drawPopZone() {
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    ctx.beginPath();
    ctx.arc(cx, cy, CONFIG.POP_ZONE_RADIUS, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.3)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Label
    ctx.fillStyle = 'rgba(99, 102, 241, 0.5)';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('TAP Zone', cx, cy + CONFIG.POP_ZONE_RADIUS + 15);
}
