/**
 * Space Defender 3D - Three.js Game
 * All 5 Gestures: TAP (laser), PUSH (shield bash), PULL (tractor), HOLD (shield), WAVE (special)
 */

// =============================================================================
// CONFIGURATION
// =============================================================================
const CONFIG = {
    // Energy & Resources
    ENERGY_MAX: 100,
    ENERGY_REGEN: 0.15,
    LASER_COST: 5,
    SHIELD_BASH_COST: 15,
    TRACTOR_COST: 0.3,
    SHIELD_COST: 0.5,
    SPECIAL_COOLDOWN: 10000, // 10 seconds

    // Damage
    LASER_DAMAGE: 10,
    SHIELD_BASH_DAMAGE: 25,
    SPECIAL_DAMAGE: 100,
    ASTEROID_DAMAGE: 20,
    ENEMY_DAMAGE: 30,

    // Timing
    GESTURE_FRAMES_REQ: 2,
    RAPID_FIRE_RATE: 4,
    COMBO_RESET_TIME: 2000,

    // Spawning
    SPAWN_INTERVAL_BASE: 2000,
    SPAWN_INTERVAL_MIN: 600,
    WAVE_DURATION: 30000, // 30 seconds per wave

    // Visuals
    STAR_COUNT: 500,
    FOG_COLOR: 0x0a0a1a,
    AMBIENT_COLOR: 0x222244
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
let score = 0;
let wave = 1;
let energy = CONFIG.ENERGY_MAX;
let kills = 0;

// Combo system
let comboCount = 0;
let comboMultiplier = 1;
let lastHitTime = 0;

// Special attack
let lastSpecialTime = 0;
let specialReady = true;

// Gesture
let socket = null;
let currentGesture = 'none';
let lastGesture = 'none';
let gestureFrames = 0;
let waveFrames = 0;

// Shield & effects
let isShieldActive = false;
let isShieldBashing = false;

// Wave system
let waveStartTime = 0;
let nextSpawnTime = 0;

// Manual gesture override (for testing with keyboard)
let manualGestureOverride = null;
let manualGestureFrames = 0;

// 3D Objects
let scene, camera, renderer;
let ship, shield, tractorBeam;
let stars = [];
let asteroids = [];
let enemies = [];
let lasers = [];
let powerups = [];
let particles = [];

// =============================================================================
// INITIALIZATION
// =============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    initSocket();
    initUI();
    initKeyboardControls();
    enterMenu();
    animate();
});

// Keyboard controls for manual gesture testing
function initKeyboardControls() {
    const gestureKeys = {
        '1': 'tap',
        '2': 'push',
        '3': 'pull',
        '4': 'hold',
        '5': 'wave',
        '0': 'none',
        'Escape': 'none'
    };

    document.addEventListener('keydown', (e) => {
        if (gestureKeys[e.key]) {
            manualGestureOverride = gestureKeys[e.key];
            manualGestureFrames = 1;
            console.log('[KEYBOARD] Gesture override:', manualGestureOverride);
        }
    });

    document.addEventListener('keyup', (e) => {
        if (gestureKeys[e.key] && manualGestureOverride === gestureKeys[e.key]) {
            // Keep hold active while key is pressed, reset on release
            if (e.key !== '4') { // Don't auto-reset HOLD on keyup
                // manualGestureOverride = null;
            }
        }
    });

    console.log('[INFO] Keyboard controls active: 1=TAP, 2=PUSH, 3=PULL, 4=HOLD, 5=WAVE, 0=NONE');
}

function initThreeJS() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(CONFIG.FOG_COLOR);
    scene.fog = new THREE.Fog(CONFIG.FOG_COLOR, 30, 80);

    // Camera
    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.set(0, 8, 15);
    camera.lookAt(0, 0, -10);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Lights
    const ambient = new THREE.AmbientLight(CONFIG.AMBIENT_COLOR, 0.6);
    scene.add(ambient);

    const directional = new THREE.DirectionalLight(0xffffff, 0.8);
    directional.position.set(5, 10, 5);
    directional.castShadow = true;
    scene.add(directional);

    // Create game objects
    createStarfield();
    createShip();
    createShield();
    createTractorBeam();

    window.addEventListener('resize', onWindowResize);
}

function createStarfield() {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(CONFIG.STAR_COUNT * 3);

    for (let i = 0; i < CONFIG.STAR_COUNT * 3; i += 3) {
        positions[i] = (Math.random() - 0.5) * 200;
        positions[i + 1] = (Math.random() - 0.5) * 100;
        positions[i + 2] = -Math.random() * 150 - 20;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
        color: 0xffffff,
        size: 0.5,
        transparent: true,
        opacity: 0.8
    });

    const starfield = new THREE.Points(geometry, material);
    scene.add(starfield);
}

function createShip() {
    ship = new THREE.Group();

    // Main body - sleek triangular shape
    const bodyGeo = new THREE.ConeGeometry(0.8, 3, 4);
    const bodyMat = new THREE.MeshStandardMaterial({
        color: 0x06b6d4,
        metalness: 0.8,
        roughness: 0.2,
        emissive: 0x033d4d,
        emissiveIntensity: 0.3
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.rotation.x = Math.PI / 2;
    body.rotation.z = Math.PI / 4;
    ship.add(body);

    // Wings
    const wingGeo = new THREE.BoxGeometry(4, 0.1, 1);
    const wingMat = new THREE.MeshStandardMaterial({
        color: 0x8b5cf6,
        metalness: 0.6,
        roughness: 0.3
    });
    const wings = new THREE.Mesh(wingGeo, wingMat);
    wings.position.z = 0.5;
    ship.add(wings);

    // Engine glow
    const engineGeo = new THREE.SphereGeometry(0.3, 16, 16);
    const engineMat = new THREE.MeshBasicMaterial({
        color: 0xfbbf24,
        transparent: true,
        opacity: 0.9
    });
    const engine = new THREE.Mesh(engineGeo, engineMat);
    engine.position.z = 1.5;
    engine.name = 'engine';
    ship.add(engine);

    ship.position.set(0, 0, 5);
    scene.add(ship);
}

function createShield() {
    const shieldGeo = new THREE.SphereGeometry(2.5, 32, 32);
    const shieldMat = new THREE.MeshBasicMaterial({
        color: 0x06b6d4,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide
    });
    shield = new THREE.Mesh(shieldGeo, shieldMat);
    shield.position.copy(ship.position);
    scene.add(shield);
}

function createTractorBeam() {
    const beamGeo = new THREE.ConeGeometry(3, 20, 16, 1, true);
    const beamMat = new THREE.MeshBasicMaterial({
        color: 0x8b5cf6,
        transparent: true,
        opacity: 0,
        side: THREE.DoubleSide
    });
    tractorBeam = new THREE.Mesh(beamGeo, beamMat);
    tractorBeam.rotation.x = -Math.PI / 2;
    tractorBeam.position.set(0, 0, -5);
    scene.add(tractorBeam);
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

    document.getElementById('restartBtn').addEventListener('click', restartGame);
}

// =============================================================================
// GAME FLOW
// =============================================================================
function enterMenu() {
    currentState = STATE.MENU;
    document.getElementById('startOverlay').style.display = 'flex';
    document.getElementById('gameOverOverlay').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'none';
    document.querySelector('.start-hint').innerHTML = "Press <b>Start</b> to begin calibration";

    updateButtons(true, false, false);
    clearAllObjects();
}

function enterCalibration() {
    currentState = STATE.CALIBRATING;
    document.getElementById('loadingSpinner').style.display = 'block';
    document.querySelector('.start-hint').innerHTML = "<b>CALIBRATING...</b><br>Stay still for a moment";

    updateButtons(false, true, false);
}

function enterPlaying() {
    currentState = STATE.PLAYING;
    document.getElementById('startOverlay').style.display = 'none';
    document.getElementById('gameOverOverlay').style.display = 'none';
    document.getElementById('comboDisplay').style.display = 'flex';

    resetGameStats();
    showFeedback("DEFEND!", "hit");

    updateButtons(false, true, true);
}

function enterGameOver() {
    currentState = STATE.GAME_OVER;

    document.getElementById('gameOverOverlay').style.display = 'flex';
    document.getElementById('goTitle').textContent = 'GAME OVER';
    document.getElementById('goMessage').textContent = `Wave ${wave} reached`;
    document.getElementById('finalScore').textContent = score;
    document.getElementById('finalWave').textContent = wave;
    document.getElementById('finalKills').textContent = kills;
}

function restartGame() {
    clearAllObjects();
    resetGameStats();
    enterPlaying();
}

function resetGameStats() {
    score = 0;
    wave = 1;
    energy = CONFIG.ENERGY_MAX;
    kills = 0;
    comboCount = 0;
    comboMultiplier = 1;
    lastSpecialTime = 0;
    specialReady = true;
    waveStartTime = Date.now();
    nextSpawnTime = Date.now() + 1000;

    updateHUD();
    updateComboUI();
    updateSpecialUI();
}

function updateButtons(canStart, canStop, canBaseline) {
    document.getElementById('startBtn').disabled = !canStart;
    document.getElementById('stopBtn').disabled = !canStop;
    document.getElementById('rebaselineBtn').disabled = !canBaseline;
}

function clearAllObjects() {
    [...asteroids, ...enemies, ...lasers, ...powerups, ...particles].forEach(obj => {
        scene.remove(obj);
    });
    asteroids = [];
    enemies = [];
    lasers = [];
    powerups = [];
    particles = [];
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

    // Wave reset logic
    if (newGesture === 'wave') {
        waveFrames++;
    } else {
        waveFrames = 0;
    }

    // Gesture confirmation - track frames for rapid fire detection
    if (newGesture === lastGesture) {
        gestureFrames++;
    } else {
        gestureFrames = 1; // Start at 1 (this is the first frame of new gesture)
    }
    lastGesture = newGesture;

    // Update current gesture immediately (server already does filtering)
    // Only require confirmation for game actions, not for display
    currentGesture = newGesture;

    updateDebugInfo(currentGesture, gate, currentState);
    updateGestureIcon(currentGesture);

    if (currentState === STATE.PLAYING) {
        processGameInput(currentGesture, gate, gestureFrames);
    } else if (currentState === STATE.GAME_OVER && waveFrames >= 2) {
        restartGame();
    }
}

function processGameInput(gesture, gate, frames) {
    const now = Date.now();

    // Combo timeout
    if (comboCount > 0 && now - lastHitTime > CONFIG.COMBO_RESET_TIME) {
        resetCombo();
    }

    // Energy regeneration
    energy = Math.min(CONFIG.ENERGY_MAX, energy + CONFIG.ENERGY_REGEN);

    // HOLD - Defensive Shield (works with gate='hold' from server)
    if (gesture === 'hold' || gate === 'hold') {
        if (energy > CONFIG.SHIELD_COST) {
            activateShield(true);
            energy -= CONFIG.SHIELD_COST;
        }
    } else {
        activateShield(false);
    }

    // TAP - Fire Laser (rapid fire) - requires CNN gate
    const isInitialHit = (frames === CONFIG.GESTURE_FRAMES_REQ);
    const isRapidFire = (frames > CONFIG.GESTURE_FRAMES_REQ && frames % CONFIG.RAPID_FIRE_RATE === 0);

    if (gesture === 'tap' && (gate === 'cnn' || gate === 'hold') && (isInitialHit || isRapidFire)) {
        if (energy >= CONFIG.LASER_COST) {
            fireLaser();
            energy -= CONFIG.LASER_COST;
        }
    }

    // PUSH - Shield Bash - requires CNN gate
    if (gesture === 'push' && (gate === 'cnn' || gate === 'hold') && frames === CONFIG.GESTURE_FRAMES_REQ) {
        if (energy >= CONFIG.SHIELD_BASH_COST) {
            shieldBash();
            energy -= CONFIG.SHIELD_BASH_COST;
        }
    }

    // PULL - Tractor Beam (works with any gate since it's a continuous action)
    if (gesture === 'pull') {
        if (energy > CONFIG.TRACTOR_COST) {
            activateTractorBeam(true);
            energy -= CONFIG.TRACTOR_COST;
        }
    } else {
        activateTractorBeam(false);
    }

    // WAVE - Special Attack (needs 3+ frames to confirm)
    if (gesture === 'wave' && specialReady && frames >= 3) {
        triggerSpecialAttack();
    }

    updateHUD();
}

// =============================================================================
// GAME ACTIONS
// =============================================================================
function fireLaser() {
    const laserGeo = new THREE.CylinderGeometry(0.08, 0.08, 2, 8);
    const laserMat = new THREE.MeshBasicMaterial({
        color: 0xef4444,
        emissive: 0xef4444
    });
    const laser = new THREE.Mesh(laserGeo, laserMat);
    laser.rotation.x = Math.PI / 2;
    laser.position.copy(ship.position);
    laser.position.z -= 2;
    laser.userData = { speed: 1.5, damage: CONFIG.LASER_DAMAGE };

    scene.add(laser);
    lasers.push(laser);

    // Muzzle flash
    const engine = ship.getObjectByName('engine');
    if (engine) {
        engine.material.color.setHex(0xef4444);
        setTimeout(() => engine.material.color.setHex(0xfbbf24), 100);
    }
}

function shieldBash() {
    isShieldBashing = true;
    shield.material.color.setHex(0x22c55e);
    shield.material.opacity = 0.5;

    showFeedback("BASH!", "shield");

    // Damage nearby asteroids/enemies
    const bashRadius = 5;
    [...asteroids, ...enemies].forEach(obj => {
        const dist = obj.position.distanceTo(ship.position);
        if (dist < bashRadius) {
            damageObject(obj, CONFIG.SHIELD_BASH_DAMAGE);
            // Push back
            const dir = obj.position.clone().sub(ship.position).normalize();
            obj.position.add(dir.multiplyScalar(3));
        }
    });

    setTimeout(() => {
        isShieldBashing = false;
        shield.material.color.setHex(0x06b6d4);
        shield.material.opacity = 0;
    }, 300);
}

function activateShield(active) {
    isShieldActive = active;

    if (active && !isShieldBashing) {
        shield.material.opacity = 0.3;
        shield.scale.setScalar(1 + Math.sin(Date.now() * 0.01) * 0.05);
        document.getElementById('shieldStatus').classList.add('active');
    } else if (!isShieldBashing) {
        shield.material.opacity = 0;
        document.getElementById('shieldStatus').classList.remove('active');
    }

    shield.position.copy(ship.position);
}

function activateTractorBeam(active) {
    if (active) {
        tractorBeam.material.opacity = 0.3;
        tractorBeam.position.set(ship.position.x, ship.position.y, ship.position.z - 10);

        // Pull powerups toward ship
        powerups.forEach(p => {
            const dir = ship.position.clone().sub(p.position).normalize();
            p.position.add(dir.multiplyScalar(0.3));
        });
    } else {
        tractorBeam.material.opacity = 0;
    }
}

function triggerSpecialAttack() {
    specialReady = false;
    lastSpecialTime = Date.now();

    showFeedback("ENERGY WAVE!", "special");

    // Visual effect - expanding ring
    const ringGeo = new THREE.RingGeometry(1, 2, 32);
    const ringMat = new THREE.MeshBasicMaterial({
        color: 0xfbbf24,
        transparent: true,
        opacity: 1,
        side: THREE.DoubleSide
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.copy(ship.position);
    ring.rotation.x = -Math.PI / 2;
    ring.userData = { type: 'wave', age: 0 };
    scene.add(ring);
    particles.push(ring);

    // Damage all enemies and asteroids
    [...asteroids, ...enemies].forEach(obj => {
        damageObject(obj, CONFIG.SPECIAL_DAMAGE);
        createExplosion(obj.position);
    });

    // Add score bonus
    const bonus = asteroids.length * 50 + enemies.length * 100;
    score += bonus;
    updateHUD();

    setTimeout(() => {
        specialReady = true;
        updateSpecialUI();
    }, CONFIG.SPECIAL_COOLDOWN);

    updateSpecialUI();
}

function damageObject(obj, damage) {
    obj.userData.hp = (obj.userData.hp || 50) - damage;

    if (obj.userData.hp <= 0) {
        destroyObject(obj);
    } else {
        // Flash effect
        const originalColor = obj.material.color.getHex();
        obj.material.color.setHex(0xffffff);
        setTimeout(() => obj.material.color.setHex(originalColor), 100);
    }
}

function destroyObject(obj) {
    createExplosion(obj.position);
    scene.remove(obj);

    const idx1 = asteroids.indexOf(obj);
    if (idx1 > -1) {
        asteroids.splice(idx1, 1);
        incrementCombo();
        score += 10 * comboMultiplier;
    }

    const idx2 = enemies.indexOf(obj);
    if (idx2 > -1) {
        enemies.splice(idx2, 1);
        incrementCombo();
        score += 25 * comboMultiplier;

        // Chance to spawn powerup
        if (Math.random() < 0.3) {
            spawnPowerup(obj.position.clone());
        }
    }

    kills++;
    lastHitTime = Date.now();
    updateHUD();
    updateComboUI();
}

function createExplosion(position) {
    for (let i = 0; i < 8; i++) {
        const geo = new THREE.SphereGeometry(0.2, 8, 8);
        const mat = new THREE.MeshBasicMaterial({
            color: Math.random() > 0.5 ? 0xfbbf24 : 0xef4444,
            transparent: true,
            opacity: 1
        });
        const particle = new THREE.Mesh(geo, mat);
        particle.position.copy(position);
        particle.userData = {
            velocity: new THREE.Vector3(
                (Math.random() - 0.5) * 0.5,
                (Math.random() - 0.5) * 0.5,
                (Math.random() - 0.5) * 0.5
            ),
            age: 0
        };
        scene.add(particle);
        particles.push(particle);
    }
}

function incrementCombo() {
    comboCount++;
    const tier = Math.floor((comboCount - 1) / 5);
    comboMultiplier = Math.pow(2, Math.min(tier, 4));
}

function resetCombo() {
    if (comboCount > 0) {
        comboCount = 0;
        comboMultiplier = 1;
        updateComboUI();
    }
}

// =============================================================================
// SPAWNING
// =============================================================================
function spawnAsteroid() {
    const size = 0.5 + Math.random() * 1.5;
    const geo = new THREE.DodecahedronGeometry(size, 0);
    const mat = new THREE.MeshStandardMaterial({
        color: 0x555566,
        roughness: 0.9,
        metalness: 0.1
    });
    const asteroid = new THREE.Mesh(geo, mat);

    asteroid.position.set(
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 10,
        -50 - Math.random() * 20
    );
    asteroid.userData = {
        speed: 0.1 + Math.random() * 0.1 + wave * 0.02,
        rotationSpeed: (Math.random() - 0.5) * 0.05,
        hp: 20 + size * 20
    };

    scene.add(asteroid);
    asteroids.push(asteroid);
}

function spawnEnemy() {
    const enemyGroup = new THREE.Group();

    // Body
    const bodyGeo = new THREE.OctahedronGeometry(1, 0);
    const bodyMat = new THREE.MeshStandardMaterial({
        color: 0xef4444,
        emissive: 0x550000,
        emissiveIntensity: 0.5,
        metalness: 0.7,
        roughness: 0.3
    });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    enemyGroup.add(body);

    // Glow
    const glowGeo = new THREE.SphereGeometry(1.3, 16, 16);
    const glowMat = new THREE.MeshBasicMaterial({
        color: 0xef4444,
        transparent: true,
        opacity: 0.2
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    enemyGroup.add(glow);

    enemyGroup.position.set(
        (Math.random() - 0.5) * 15,
        (Math.random() - 0.5) * 8,
        -50
    );
    enemyGroup.userData = {
        speed: 0.08 + wave * 0.015,
        hp: 50 + wave * 10,
        isEnemy: true
    };

    scene.add(enemyGroup);
    enemies.push(enemyGroup);
}

function spawnPowerup(position) {
    const geo = new THREE.TorusGeometry(0.5, 0.15, 8, 16);
    const mat = new THREE.MeshBasicMaterial({
        color: 0x22c55e,
        transparent: true,
        opacity: 0.9
    });
    const powerup = new THREE.Mesh(geo, mat);
    powerup.position.copy(position);
    powerup.userData = {
        type: Math.random() > 0.5 ? 'energy' : 'score',
        rotationSpeed: 0.05
    };

    scene.add(powerup);
    powerups.push(powerup);
}

// =============================================================================
// GAME LOOP
// =============================================================================
function animate() {
    requestAnimationFrame(animate);

    if (currentState === STATE.PLAYING) {
        updateGame();
    }

    // Render
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

function updateGame() {
    const now = Date.now();
    const dt = 0.016;

    // Wave progression
    if (now - waveStartTime > CONFIG.WAVE_DURATION) {
        wave++;
        waveStartTime = now;
        showFeedback(`WAVE ${wave}!`, "special");
        updateHUD();
    }

    // Spawning
    if (now > nextSpawnTime) {
        const spawnInterval = Math.max(
            CONFIG.SPAWN_INTERVAL_MIN,
            CONFIG.SPAWN_INTERVAL_BASE - wave * 150
        );
        nextSpawnTime = now + spawnInterval;

        if (Math.random() > 0.3) {
            spawnAsteroid();
        } else if (wave >= 2) {
            spawnEnemy();
        }
    }

    // Update objects
    updateAsteroids(dt);
    updateEnemies(dt);
    updateLasers(dt);
    updatePowerups(dt);
    updateParticles(dt);
    checkCollisions();

    // Update special cooldown
    if (!specialReady) {
        updateSpecialUI();
    }
}

function updateAsteroids(dt) {
    for (let i = asteroids.length - 1; i >= 0; i--) {
        const a = asteroids[i];
        a.position.z += a.userData.speed;
        a.rotation.x += a.userData.rotationSpeed;
        a.rotation.y += a.userData.rotationSpeed * 0.7;

        // Remove if past camera
        if (a.position.z > 20) {
            scene.remove(a);
            asteroids.splice(i, 1);
        }
    }
}

function updateEnemies(dt) {
    for (let i = enemies.length - 1; i >= 0; i--) {
        const e = enemies[i];
        e.position.z += e.userData.speed;
        e.rotation.y += 0.02;

        if (e.position.z > 20) {
            scene.remove(e);
            enemies.splice(i, 1);
        }
    }
}

function updateLasers(dt) {
    for (let i = lasers.length - 1; i >= 0; i--) {
        const l = lasers[i];
        l.position.z -= l.userData.speed;

        if (l.position.z < -60) {
            scene.remove(l);
            lasers.splice(i, 1);
        }
    }
}

function updatePowerups(dt) {
    for (let i = powerups.length - 1; i >= 0; i--) {
        const p = powerups[i];
        p.rotation.y += p.userData.rotationSpeed;
        p.rotation.x += 0.02;
        p.position.z += 0.02; // Slowly drift

        // Check collection
        if (p.position.distanceTo(ship.position) < 2) {
            collectPowerup(p);
            scene.remove(p);
            powerups.splice(i, 1);
        } else if (p.position.z > 15) {
            scene.remove(p);
            powerups.splice(i, 1);
        }
    }
}

function collectPowerup(p) {
    if (p.userData.type === 'energy') {
        energy = Math.min(CONFIG.ENERGY_MAX, energy + 30);
        showFeedback("+30 ENERGY", "shield");
    } else {
        score += 100;
        showFeedback("+100 SCORE", "hit");
    }
    updateHUD();
}

function updateParticles(dt) {
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.userData.age++;

        if (p.userData.type === 'wave') {
            // Expanding wave ring
            p.scale.setScalar(1 + p.userData.age * 0.5);
            p.material.opacity = 1 - p.userData.age * 0.05;
        } else {
            // Explosion particles
            p.position.add(p.userData.velocity);
            p.material.opacity -= 0.05;
            p.scale.multiplyScalar(0.95);
        }

        if (p.material.opacity <= 0 || p.userData.age > 30) {
            scene.remove(p);
            particles.splice(i, 1);
        }
    }
}

function checkCollisions() {
    // Laser vs Asteroids/Enemies
    for (let i = lasers.length - 1; i >= 0; i--) {
        const laser = lasers[i];

        for (const asteroid of asteroids) {
            if (laser.position.distanceTo(asteroid.position) < 2) {
                damageObject(asteroid, laser.userData.damage);
                scene.remove(laser);
                lasers.splice(i, 1);
                break;
            }
        }

        if (lasers[i]) {
            for (const enemy of enemies) {
                if (laser.position.distanceTo(enemy.position) < 1.5) {
                    damageObject(enemy, laser.userData.damage);
                    scene.remove(laser);
                    lasers.splice(i, 1);
                    break;
                }
            }
        }
    }

    // Objects vs Ship
    if (!isShieldActive) {
        for (const asteroid of asteroids) {
            if (asteroid.position.distanceTo(ship.position) < 2) {
                takeDamage(CONFIG.ASTEROID_DAMAGE);
                destroyObject(asteroid);
                break;
            }
        }

        for (const enemy of enemies) {
            if (enemy.position.distanceTo(ship.position) < 2) {
                takeDamage(CONFIG.ENEMY_DAMAGE);
                destroyObject(enemy);
                break;
            }
        }
    }
}

function takeDamage(amount) {
    energy -= amount;
    resetCombo();
    showFeedback("HIT!", "miss");

    // Screen shake effect
    camera.position.x += (Math.random() - 0.5) * 0.5;
    camera.position.y += (Math.random() - 0.5) * 0.5;
    setTimeout(() => {
        camera.position.set(0, 8, 15);
    }, 200);

    if (energy <= 0) {
        enterGameOver();
    }

    updateHUD();
}

// =============================================================================
// UI UPDATES
// =============================================================================
function updateHUD() {
    document.getElementById('scoreValue').textContent = score;
    document.getElementById('waveValue').textContent = wave;

    const energyPercent = Math.max(0, (energy / CONFIG.ENERGY_MAX) * 100);
    document.getElementById('energyBar').style.width = energyPercent + '%';
}

function updateComboUI() {
    const display = document.getElementById('comboDisplay');
    const val = document.getElementById('comboValue');
    const mult = document.getElementById('comboMultiplier');

    if (comboCount > 1) {
        display.style.display = 'flex';
        val.textContent = comboCount;
        mult.textContent = 'x' + comboMultiplier;

        val.classList.remove('pulse');
        void val.offsetWidth;
        val.classList.add('pulse');

        mult.classList.toggle('active', comboMultiplier > 1);
    } else {
        val.textContent = comboCount > 0 ? comboCount : '';
        mult.classList.remove('active');
    }
}

function updateSpecialUI() {
    const indicator = document.getElementById('specialIndicator');
    const text = document.getElementById('specialText');

    if (specialReady) {
        indicator.classList.remove('cooldown');
        text.textContent = 'READY';
    } else {
        indicator.classList.add('cooldown');
        const remaining = Math.ceil((CONFIG.SPECIAL_COOLDOWN - (Date.now() - lastSpecialTime)) / 1000);
        text.textContent = remaining + 's';
    }
}

function showFeedback(text, type) {
    const el = document.getElementById('centerFeedback');
    if (!el) return;

    el.textContent = text;
    el.className = 'center-feedback feedback-' + type;
    el.style.opacity = 1;

    setTimeout(() => el.style.opacity = 0, 800);
}

function updateGestureIcon(gesture) {
    const icons = {
        'tap': '👊',
        'push': '👆',
        'pull': '👇',
        'hold': '🤚',
        'wave': '👋',
        'none': '✋'
    };
    document.getElementById('gestureIcon').textContent = icons[gesture] || '✋';
    document.getElementById('gestureName').textContent = gesture;

    const indicator = document.getElementById('gestureIndicator');
    indicator.className = 'gesture-indicator gesture-' + gesture;
}

function updateDebugInfo(gesture, gate, state) {
    const el = document.getElementById('debugPanel');
    if (el) {
        el.innerHTML = `<span>G: <b>${gesture}</b></span><span>Gate: <b>${gate}</b></span><span>S: <b>${state}</b></span>`;
    }
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
