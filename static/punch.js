/**
 * 3D Punching Bag Game - Three.js Implementation
 * Gestures: Tap (Punch), Push/Pull (Move), Hold (Block), Wave (Reset)
 */

// =============================================================================
// CONFIGURATION
// =============================================================================
const CONFIG = {
    BAG_HP_MAX: 100,
    PLAYER_HP_MAX: 3,
    // Timing
    ATTACK_WARNING_TIME: 2000,
    ATTACK_DURATION: 1500,
    ATTACK_COOLDOWN_MIN: 2500,
    ATTACK_COOLDOWN_MAX: 5000,
    COMBO_RESET_TIME: 2000,

    // Mechanics
    PUNCH_DAMAGE: 1,
    GESTURE_FRAMES_REQ: 2,
    RAPID_FIRE_RATE: 6,

    // Visuals
    BAG_COLOR: 0xdc2626,
    BAG_WARNING_COLOR: 0xff0000,
    BAG_ATTACK_COLOR: 0xffaa00,
    ROOM_COLOR: 0x111111
};

// =============================================================================
// STATE
// =============================================================================
const STATE = {
    MENU: 'menu',
    CALIBRATING: 'calib',
    PLAYING_IDLE: 'idle',
    PLAYING_WARN: 'warn',
    PLAYING_ATK: 'atk',
    KO: 'ko',
    GAME_OVER: 'game_over'
};

let currentState = STATE.MENU;
let bagHp = CONFIG.BAG_HP_MAX;
let playerHp = CONFIG.PLAYER_HP_MAX;

// Combo System
let comboCount = 0;
let comboMultiplier = 1;
let lastHitTime = 0;

// Timers
let nextAttackTime = 0;
let warningStartTime = 0;
let attackStartTime = 0;

// Gestures
let socket = null;
let currentGesture = 'none';
let lastGesture = 'none';
let gestureFrames = 0;
let waveFrames = 0;
let isBlocking = false;

// 3D Scene
let scene, camera, renderer;
let bag, chain, spotLight;
let cameraTargetZ = 5;
let bagAngle = 0;
let bagAngularVelocity = 0;
let impactShake = 0;

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    initSocket();
    initUI();
    enterMenu();
    animate();
});

function initThreeJS() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(CONFIG.ROOM_COLOR);
    // REMOVED FOG for debugging visibility issues

    // Camera
    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 1.2, 5); // Slightly higher
    camera.lookAt(0, 1.5, 0); // Explicitly look at the bag pivot point area

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    spotLight = new THREE.SpotLight(0xffffff, 1.2);
    spotLight.position.set(2, 6, 4);
    spotLight.angle = Math.PI / 4;
    spotLight.penumbra = 0.2;
    spotLight.castShadow = true;
    scene.add(spotLight);

    // Initial Pivot Target Point (optional helper)
    // const target = new THREE.Object3D();
    // target.position.set(0, 1.5, 0);
    // scene.add(target);

    // Bag Group
    createBag();
    createRoom();

    window.addEventListener('resize', onWindowResize, false);
}

function createBag() {
    const bagMaterial = new THREE.MeshStandardMaterial({
        color: CONFIG.BAG_COLOR, roughness: 0.5, metalness: 0.2
    });

    bag = new THREE.Group();
    bag.position.set(0, 3.5, 0);
    scene.add(bag);

    // Chain
    const chainGeo = new THREE.CylinderGeometry(0.04, 0.04, 2, 8);
    const chainMat = new THREE.MeshStandardMaterial({ color: 0x777777 });
    chain = new THREE.Mesh(chainGeo, chainMat);
    chain.position.y = -1;
    chain.castShadow = true;
    bag.add(chain);

    // Bag Body
    const bodyGeo = new THREE.CylinderGeometry(0.55, 0.55, 1.6, 32);
    const bodyMesh = new THREE.Mesh(bodyGeo, bagMaterial);
    bodyMesh.position.y = -2.2;
    bodyMesh.castShadow = true;
    bodyMesh.name = 'bagBody';
    bag.add(bodyMesh);
}

function createRoom() {
    const floorGeo = new THREE.PlaneGeometry(50, 50);
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.9 });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -2.5;
    floor.receiveShadow = true;
    scene.add(floor);
}

function onWindowResize() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
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

    document.getElementById('rebaselineBtn').addEventListener('click', () => {
        socket.emit('rebaseline');
    });

    document.getElementById('restartBtn').addEventListener('click', () => {
        resetGameStats();
        enterPlaying();
    });
}

// =============================================================================
// FLOW CONTROL
// =============================================================================
function enterMenu() {
    currentState = STATE.MENU;
    document.getElementById('startOverlay').style.display = 'flex';
    document.getElementById('gameOverOverlay').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'none';
    document.getElementById('comboDisplay').style.display = 'none';
    document.querySelector('.start-hint').innerHTML = "Press <b>Start</b> to begin calibration.";

    if (bag) {
        bag.rotation.x = 0;
        bagAngle = 0;
        const body = bag.getObjectByName('bagBody');
        if (body) body.material.color.setHex(CONFIG.BAG_COLOR);
    }

    updateButtons(true, false, false);
}

function enterCalibration() {
    currentState = STATE.CALIBRATING;
    const overlay = document.getElementById('startOverlay');
    overlay.style.display = 'flex';
    document.getElementById('loadingSpinner').style.display = 'block';
    document.querySelector('.start-hint').innerHTML = "<b>CALIBRATING...</b><br>Stay still for a moment.";

    updateButtons(false, true, false);
}

function enterPlaying() {
    currentState = STATE.PLAYING_IDLE;
    document.getElementById('startOverlay').style.display = 'none';
    document.getElementById('gameOverOverlay').style.display = 'none';
    document.getElementById('comboDisplay').style.display = 'flex';

    showFeedback("FIGHT!", "hit");
    resetGameStats();
    resetAttackTimer();

    updateButtons(false, true, true);
}

function updateButtons(canStart, canStop, canBaseline) {
    document.getElementById('startBtn').disabled = !canStart;
    document.getElementById('stopBtn').disabled = !canStop;
    document.getElementById('rebaselineBtn').disabled = !canBaseline;
    const sim = document.getElementById('simulateToggle');
    if (sim) sim.disabled = !canStart;
}

function resetGameStats() {
    bagHp = CONFIG.BAG_HP_MAX;
    playerHp = CONFIG.PLAYER_HP_MAX;
    comboCount = 0;
    comboMultiplier = 1;
    updateHealthUI();
    updateComboUI();

    const body = bag.getObjectByName('bagBody');
    if (body) body.material.color.setHex(CONFIG.BAG_COLOR);
    bagAngle = 0;
    bagAngularVelocity = 0;
}

function resetAttackTimer() {
    nextAttackTime = Date.now() + CONFIG.ATTACK_COOLDOWN_MIN + Math.random() * (CONFIG.ATTACK_COOLDOWN_MAX - CONFIG.ATTACK_COOLDOWN_MIN);
}

// =============================================================================
// DATA HANDLING
// =============================================================================
function handleRadarData(data) {
    if (currentState === STATE.CALIBRATING && data.baseline_ready) {
        enterPlaying();
    }

    const newGesture = data.gesture_out || 'none';
    const gate = data.gate || 'none';

    // Wave Reset Logic (Fix: make it more reactive)
    if (newGesture === 'wave') {
        waveFrames++;
    } else {
        waveFrames = 0;
    }

    if (newGesture === lastGesture) gestureFrames++;
    else gestureFrames = 0;
    lastGesture = newGesture;

    if (gestureFrames >= CONFIG.GESTURE_FRAMES_REQ) {
        currentGesture = newGesture;
    }

    updateDebugInfo(currentGesture, gate, currentState);
    updateGestureIcon(currentGesture);

    if (isGameActive()) {
        processGameInput(currentGesture, gate);
    }
    else if (currentState === STATE.KO || currentState === STATE.GAME_OVER) {
        if (waveFrames >= 2) {
            resetGameStats();
            enterPlaying();
        }
    }
}

function isGameActive() {
    return currentState === STATE.PLAYING_IDLE ||
        currentState === STATE.PLAYING_WARN ||
        currentState === STATE.PLAYING_ATK;
}

function updateDebugInfo(gesture, gate, state) {
    const el = document.getElementById('debugPanel');
    if (el) el.innerHTML = `<span>G:<b>${gesture}</b></span> <span>St:<b>${state}</b></span>`;
}

function updateGestureIcon(gesture) {
    const icons = { 'tap': '👊', 'push': '💪', 'pull': '🎯', 'hold': '🛡️', 'wave': '👋', 'none': '✋' };
    document.getElementById('gestureIcon').textContent = icons[gesture] || '✋';
    document.getElementById('gestureName').textContent = gesture;

    const indicator = document.getElementById('gestureIndicator');
    indicator.className = 'gesture-indicator gesture-' + gesture;
}

// =============================================================================
// GAMEPLAY MECHANICS
// =============================================================================
function processGameInput(gesture, gate) {
    const now = Date.now();

    // COMBO TIMEOUT CHECK
    if (comboCount > 0 && now - lastHitTime > CONFIG.COMBO_RESET_TIME) {
        resetCombo();
    }

    isBlocking = (gesture === 'hold');
    if (isBlocking) {
        resetCombo();
    }

    // RAPID FIRE TAP
    const isInitialHit = (gestureFrames === CONFIG.GESTURE_FRAMES_REQ);
    const isRapidFire = (gestureFrames > CONFIG.GESTURE_FRAMES_REQ && gestureFrames % CONFIG.RAPID_FIRE_RATE === 0);

    if (gesture === 'tap' && gate === 'cnn' && (isInitialHit || isRapidFire)) {
        punchBag();
    }

    if (gesture === 'push') cameraTargetZ = Math.min(8, cameraTargetZ + 0.05);
    if (gesture === 'pull') cameraTargetZ = Math.max(3.5, cameraTargetZ - 0.05);
}

function punchBag() {
    const now = Date.now();
    lastHitTime = now;

    comboCount++;
    const tier = Math.floor((comboCount - 1) / 4);
    comboMultiplier = Math.pow(2, Math.min(tier, 5));

    const damage = CONFIG.PUNCH_DAMAGE * comboMultiplier;
    bagHp -= damage;

    impactShake = 0.3 + (comboMultiplier * 0.1);
    bagAngularVelocity += 0.08 + (comboMultiplier * 0.02);

    showFeedback("HIT! x" + comboMultiplier, "hit");
    updateComboUI();

    const body = bag.getObjectByName('bagBody');
    if (body) {
        body.material.emissive.setHex(0x555555);
        setTimeout(() => body.material.emissive.setHex(0x000000), 100);
    }

    if (bagHp <= 0) {
        knockout();
    }
    updateHealthUI();

    if (currentState === STATE.PLAYING_IDLE) {
        nextAttackTime += 500;
    }
}

function resetCombo() {
    if (comboCount > 0) {
        comboCount = 0;
        comboMultiplier = 1;
        updateComboUI();
    }
}

function updateComboUI() {
    const val = document.getElementById('comboValue');
    const mult = document.getElementById('comboMultiplier');

    if (comboCount > 1) {
        val.textContent = comboCount;
        mult.textContent = "x" + comboMultiplier;

        val.classList.remove('pulse');
        void val.offsetWidth;
        val.classList.add('pulse');

        if (comboMultiplier > 1) mult.classList.add('active');
        else mult.classList.remove('active');
    } else {
        val.textContent = comboCount > 0 ? comboCount : "";
        mult.classList.remove('active');
    }
}

function takeDamage() {
    resetCombo();
    playerHp--;
    showFeedback("OUCH!", "miss");
    updateHealthUI();

    document.body.style.backgroundColor = "#500";
    setTimeout(() => document.body.style.backgroundColor = CONFIG.ROOM_COLOR, 500);

    if (playerHp <= 0) {
        gameOver();
    }
}

// =============================================================================
// MAIN LOOP & PHYSICS
// =============================================================================
function animate() {
    requestAnimationFrame(animate);
    const now = Date.now();
    const dt = 0.016;

    if (isGameActive()) {
        updateAI(now);
        updatePhysics(dt);
    } else if (currentState === STATE.KO) {
        updatePhysics(dt);
    }

    if (renderer && scene && camera) {
        // Camera movement
        camera.position.z += (cameraTargetZ - camera.position.z) * 0.1;

        if (impactShake > 0) {
            camera.position.x = (Math.random() - 0.5) * impactShake;
            camera.position.y = 1.2 + (Math.random() - 0.5) * impactShake;
            impactShake *= 0.9;
            if (impactShake < 0.01) impactShake = 0;
        } else {
            camera.position.x = 0;
            camera.position.y = 1.2;
        }

        // Always look at the bag pivot area
        camera.lookAt(0, 1.5, 0);

        renderer.render(scene, camera);
    }
}

function updatePhysics(dt) {
    const gravity = 9.8;
    const length = 2.2;
    const damping = 0.98;

    const angularAccel = -(gravity / length) * Math.sin(bagAngle);
    bagAngularVelocity += angularAccel * dt;
    bagAngularVelocity *= damping;
    bagAngle += bagAngularVelocity * dt;

    if (bag) bag.rotation.x = bagAngle;
}

function updateAI(now) {
    const body = bag.getObjectByName('bagBody');
    if (!body) return;

    if (currentState === STATE.PLAYING_IDLE && now > nextAttackTime) {
        currentState = STATE.PLAYING_WARN;
        warningStartTime = now;
        showFeedback("INCOMING!", "miss");
    }

    if (currentState === STATE.PLAYING_WARN) {
        const progress = (now - warningStartTime);
        const pulse = Math.sin(progress * 0.015) * 0.5 + 0.5;

        body.material.color.setHex(CONFIG.BAG_WARNING_COLOR);
        body.material.emissive.setHex(0x550000);
        body.material.emissiveIntensity = pulse;

        if (progress > CONFIG.ATTACK_WARNING_TIME) {
            currentState = STATE.PLAYING_ATK;
            attackStartTime = now;
            body.material.color.setHex(CONFIG.BAG_ATTACK_COLOR);
            body.material.emissiveIntensity = 0;
            bagAngularVelocity -= 0.2;
        }
    }

    if (currentState === STATE.PLAYING_ATK) {
        if (now > attackStartTime + CONFIG.ATTACK_DURATION) {
            if (isBlocking) {
                showFeedback("BLOCKED!", "block");
            } else {
                takeDamage();
            }

            currentState = STATE.PLAYING_IDLE;
            resetAttackTimer();
            body.material.color.setHex(CONFIG.BAG_COLOR);
        }
    }
}

function knockout() {
    currentState = STATE.KO;
    document.getElementById('gameOverOverlay').style.display = 'flex';
    document.getElementById('goTitle').textContent = "KO! YOU WIN!";
    document.getElementById('goMessage').textContent = "Wave to Restart";
    bagAngularVelocity = 0;
    bagAngle = Math.PI / 2;
}

function gameOver() {
    currentState = STATE.GAME_OVER;
    document.getElementById('gameOverOverlay').style.display = 'flex';
    document.getElementById('goTitle').textContent = "DEFEAT";
    document.getElementById('goMessage').textContent = "Wave to Try Again";
}

function updateHealthUI() {
    const enemyPercent = Math.max(0, (bagHp / CONFIG.BAG_HP_MAX) * 100);
    const bar = document.getElementById('enemyHpBar');
    if (bar) bar.style.width = enemyPercent + '%';

    const hearts = document.getElementById('playerHearts');
    if (hearts) {
        for (let i = 0; i < hearts.children.length; i++) {
            if (i < playerHp) hearts.children[i].classList.remove('lost');
            else hearts.children[i].classList.add('lost');
        }
    }
}

function showFeedback(text, type) {
    const el = document.getElementById('centerFeedback');
    if (!el) return;
    el.textContent = text;
    el.className = 'center-feedback feedback-' + type;

    void el.offsetWidth;
    el.style.opacity = 1;
    setTimeout(() => el.style.opacity = 0, 800);
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
