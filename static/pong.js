/**
 * Radar Pong - Simple & Responsive Pong Game
 * PUSH = Move Left, PULL = Move Right
 */

// =============================================================================
// CONFIG
// =============================================================================
const CONFIG = {
    CANVAS_WIDTH: 800,
    CANVAS_HEIGHT: 500,
    PADDLE_WIDTH: 100,
    PADDLE_HEIGHT: 12,
    PADDLE_SPEED: 12,
    BALL_SIZE: 12,
    BALL_SPEED: 6,
    BOT_SPEED: 5,
    WIN_SCORE: 5
};

// =============================================================================
// STATE
// =============================================================================
const STATE = {
    MENU: 'menu',
    CALIBRATING: 'calibrating',
    PLAYING: 'playing',
    GAME_OVER: 'game_over'
};

let currentState = STATE.MENU;
let playerScore = 0;
let botScore = 0;

// Paddle positions (center x)
let playerPaddleX = CONFIG.CANVAS_WIDTH / 2;
let botPaddleX = CONFIG.CANVAS_WIDTH / 2;

// Ball
let ballX = CONFIG.CANVAS_WIDTH / 2;
let ballY = CONFIG.CANVAS_HEIGHT / 2;
let ballVX = CONFIG.BALL_SPEED;
let ballVY = CONFIG.BALL_SPEED;

// Gesture
let socket = null;
let currentGesture = 'none';

// Canvas
let canvas, ctx;

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initCanvas();
    initSocket();
    initUI();
    initKeyboard();
    enterMenu();
    gameLoop();
});

function initCanvas() {
    canvas = document.getElementById('gameCanvas');
    ctx = canvas.getContext('2d');

    canvas.width = CONFIG.CANVAS_WIDTH;
    canvas.height = CONFIG.CANVAS_HEIGHT;
}

function initSocket() {
    socket = io();
    socket.on('connect', () => updateStatus(true));
    socket.on('disconnect', () => updateStatus(false));
    socket.on('radar_data', handleRadarData);
}

function initUI() {
    document.getElementById('startBtn').addEventListener('click', () => {
        socket.emit('start_radar', { simulate: document.getElementById('simulateToggle').checked });
        enterCalibration();
    });

    document.getElementById('stopBtn').addEventListener('click', () => {
        socket.emit('stop_radar');
        enterMenu();
    });

    document.getElementById('restartBtn').addEventListener('click', () => {
        resetGame();
        enterPlaying();
    });
}

function initKeyboard() {
    // Keyboard controls for testing: A/Left = Push (left), D/Right = Pull (right)
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
            currentGesture = 'push';
        } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
            currentGesture = 'pull';
        }
    });

    document.addEventListener('keyup', (e) => {
        if ((e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') && currentGesture === 'push') {
            currentGesture = 'none';
        } else if ((e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') && currentGesture === 'pull') {
            currentGesture = 'none';
        }
    });

    console.log('[INFO] Keyboard: Left/A = Push (left), Right/D = Pull (right)');
}

// =============================================================================
// GAME FLOW
// =============================================================================
function enterMenu() {
    currentState = STATE.MENU;
    document.getElementById('startOverlay').style.display = 'flex';
    document.getElementById('gameOverOverlay').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'none';
    document.querySelector('.start-hint').innerHTML = "Press <b>Start</b> to begin";

    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
}

function enterCalibration() {
    currentState = STATE.CALIBRATING;
    document.getElementById('loadingSpinner').style.display = 'block';
    document.querySelector('.start-hint').innerHTML = "<b>CALIBRATING...</b>";

    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
}

function enterPlaying() {
    currentState = STATE.PLAYING;
    document.getElementById('startOverlay').style.display = 'none';
    document.getElementById('gameOverOverlay').style.display = 'none';
    resetGame();
}

function enterGameOver(winner) {
    currentState = STATE.GAME_OVER;
    document.getElementById('gameOverOverlay').style.display = 'flex';

    if (winner === 'player') {
        document.getElementById('goTitle').textContent = '🎉 YOU WIN!';
        document.getElementById('goTitle').style.color = '#22c55e';
    } else {
        document.getElementById('goTitle').textContent = '💀 BOT WINS';
        document.getElementById('goTitle').style.color = '#ef4444';
    }
    document.getElementById('goMessage').textContent = `${playerScore} - ${botScore}`;
}

function resetGame() {
    playerScore = 0;
    botScore = 0;
    playerPaddleX = CONFIG.CANVAS_WIDTH / 2;
    botPaddleX = CONFIG.CANVAS_WIDTH / 2;
    resetBall();
    updateScoreUI();
}

function resetBall() {
    ballX = CONFIG.CANVAS_WIDTH / 2;
    ballY = CONFIG.CANVAS_HEIGHT / 2;
    ballVX = (Math.random() > 0.5 ? 1 : -1) * CONFIG.BALL_SPEED;
    ballVY = (Math.random() > 0.5 ? 1 : -1) * CONFIG.BALL_SPEED;
}

// =============================================================================
// DATA HANDLING
// =============================================================================
function handleRadarData(data) {
    if (currentState === STATE.CALIBRATING && data.baseline_ready) {
        enterPlaying();
    }

    const gesture = data.gesture_out || 'none';
    currentGesture = gesture;

    updateDebug(gesture);
    updateGestureIcon(gesture);
}

// =============================================================================
// GAME LOOP
// =============================================================================
function gameLoop() {
    update();
    render();
    requestAnimationFrame(gameLoop);
}

function update() {
    if (currentState !== STATE.PLAYING) return;

    // Player paddle movement based on gesture
    if (currentGesture === 'push') {
        playerPaddleX -= CONFIG.PADDLE_SPEED;
    } else if (currentGesture === 'pull') {
        playerPaddleX += CONFIG.PADDLE_SPEED;
    }

    // Clamp player paddle
    const halfPaddle = CONFIG.PADDLE_WIDTH / 2;
    playerPaddleX = Math.max(halfPaddle, Math.min(CONFIG.CANVAS_WIDTH - halfPaddle, playerPaddleX));

    // Bot AI - follows ball with some lag
    const botTarget = ballX;
    if (botPaddleX < botTarget - 10) {
        botPaddleX += CONFIG.BOT_SPEED;
    } else if (botPaddleX > botTarget + 10) {
        botPaddleX -= CONFIG.BOT_SPEED;
    }
    botPaddleX = Math.max(halfPaddle, Math.min(CONFIG.CANVAS_WIDTH - halfPaddle, botPaddleX));

    // Ball movement
    ballX += ballVX;
    ballY += ballVY;

    // Wall collision (left/right)
    if (ballX <= CONFIG.BALL_SIZE / 2 || ballX >= CONFIG.CANVAS_WIDTH - CONFIG.BALL_SIZE / 2) {
        ballVX *= -1;
        ballX = Math.max(CONFIG.BALL_SIZE / 2, Math.min(CONFIG.CANVAS_WIDTH - CONFIG.BALL_SIZE / 2, ballX));
    }

    // Paddle collision - Player (bottom)
    const playerPaddleY = CONFIG.CANVAS_HEIGHT - 30;
    if (ballY >= playerPaddleY - CONFIG.BALL_SIZE / 2 &&
        ballY <= playerPaddleY + CONFIG.PADDLE_HEIGHT &&
        ballX >= playerPaddleX - halfPaddle &&
        ballX <= playerPaddleX + halfPaddle) {
        ballVY = -Math.abs(ballVY) * 1.05; // Speed up slightly
        ballY = playerPaddleY - CONFIG.BALL_SIZE / 2;

        // Add angle based on hit position
        const hitPos = (ballX - playerPaddleX) / halfPaddle;
        ballVX += hitPos * 2;
    }

    // Paddle collision - Bot (top)
    const botPaddleY = 30;
    if (ballY <= botPaddleY + CONFIG.PADDLE_HEIGHT + CONFIG.BALL_SIZE / 2 &&
        ballY >= botPaddleY &&
        ballX >= botPaddleX - halfPaddle &&
        ballX <= botPaddleX + halfPaddle) {
        ballVY = Math.abs(ballVY) * 1.05;
        ballY = botPaddleY + CONFIG.PADDLE_HEIGHT + CONFIG.BALL_SIZE / 2;

        const hitPos = (ballX - botPaddleX) / halfPaddle;
        ballVX += hitPos * 2;
    }

    // Clamp ball speed
    const maxSpeed = 15;
    ballVX = Math.max(-maxSpeed, Math.min(maxSpeed, ballVX));
    ballVY = Math.max(-maxSpeed, Math.min(maxSpeed, ballVY));

    // Score - ball goes past paddles
    if (ballY < 0) {
        // Player scores
        playerScore++;
        updateScoreUI();
        if (playerScore >= CONFIG.WIN_SCORE) {
            enterGameOver('player');
        } else {
            resetBall();
        }
    } else if (ballY > CONFIG.CANVAS_HEIGHT) {
        // Bot scores
        botScore++;
        updateScoreUI();
        if (botScore >= CONFIG.WIN_SCORE) {
            enterGameOver('bot');
        } else {
            resetBall();
        }
    }
}

function render() {
    // Clear
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT);

    // Center line
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 10]);
    ctx.beginPath();
    ctx.moveTo(0, CONFIG.CANVAS_HEIGHT / 2);
    ctx.lineTo(CONFIG.CANVAS_WIDTH, CONFIG.CANVAS_HEIGHT / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    if (currentState !== STATE.PLAYING && currentState !== STATE.GAME_OVER) return;

    // Player paddle (bottom, green)
    ctx.fillStyle = '#22c55e';
    ctx.shadowColor = '#22c55e';
    ctx.shadowBlur = 15;
    ctx.fillRect(
        playerPaddleX - CONFIG.PADDLE_WIDTH / 2,
        CONFIG.CANVAS_HEIGHT - 30,
        CONFIG.PADDLE_WIDTH,
        CONFIG.PADDLE_HEIGHT
    );

    // Bot paddle (top, red)
    ctx.fillStyle = '#ef4444';
    ctx.shadowColor = '#ef4444';
    ctx.shadowBlur = 15;
    ctx.fillRect(
        botPaddleX - CONFIG.PADDLE_WIDTH / 2,
        30,
        CONFIG.PADDLE_WIDTH,
        CONFIG.PADDLE_HEIGHT
    );

    // Ball
    ctx.fillStyle = '#fbbf24';
    ctx.shadowColor = '#fbbf24';
    ctx.shadowBlur = 20;
    ctx.beginPath();
    ctx.arc(ballX, ballY, CONFIG.BALL_SIZE / 2, 0, Math.PI * 2);
    ctx.fill();

    ctx.shadowBlur = 0;
}

// =============================================================================
// UI
// =============================================================================
function updateScoreUI() {
    document.getElementById('playerScore').textContent = playerScore;
    document.getElementById('botScore').textContent = botScore;
}

function updateGestureIcon(gesture) {
    const icons = {
        'push': '👆',
        'pull': '👇',
        'none': '✋',
        'hold': '🤚',
        'tap': '👊',
        'wave': '👋'
    };
    document.getElementById('gestureIcon').textContent = icons[gesture] || '✋';
    document.getElementById('gestureName').textContent = gesture;

    const indicator = document.getElementById('gestureIndicator');
    indicator.className = 'gesture-indicator gesture-' + gesture;
}

function updateDebug(gesture) {
    document.getElementById('dbgGesture').textContent = gesture;
}

function updateStatus(connected) {
    const el = document.getElementById('connectionStatus');
    if (el) {
        const txt = el.querySelector('.status-text');
        if (connected) {
            el.classList.add('connected');
            txt.textContent = 'Connected';
        } else {
            el.classList.remove('connected');
            txt.textContent = 'Offline';
        }
    }
}
