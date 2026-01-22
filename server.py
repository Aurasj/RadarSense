"""
Gesture Detection Server - Production Pipeline
Acconeer A111 with exploration server protocol + motion gating + TinyCNN

Author: Senior Engineer implementation for thesis demo
"""

# IMPORTANT: Eventlet monkey patch MUST be first!
import eventlet
eventlet.monkey_patch()

import os
import json
import threading
import time
import numpy as np
import onnxruntime as ort
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List

# =============================================================================
# CONFIGURATION
# =============================================================================
PI_HOST = os.environ.get("PI_HOST", "192.168.252.108")
PI_PORT = int(os.environ.get("PI_PORT", 6110))
UPDATE_RATE = int(os.environ.get("UPDATE_RATE", 30))
RANGE_START = float(os.environ.get("RANGE_START", 0.20))
RANGE_END = float(os.environ.get("RANGE_END", 0.55))
SERVER_PORT = int(os.environ.get("SERVER_PORT", 5002))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "out_model", "gesture_cnn.onnx")
META_PATH = os.path.join(os.path.dirname(__file__), "out_model", "gesture_cnn_meta.json")

# Gating parameters
BASELINE_DURATION = 2.0  # seconds
K_LEVEL = 3.0
K_MOTION = 3.0
HOLD_FACTOR = 1.5
HYSTERESIS_ON = 1.6
HYSTERESIS_OFF = 0.9
HOLD_OFF_FACTOR = 0.75
SMOOTHING_WINDOW = 5  # frames for CNN probability smoothing
TAIL_DURATION = 0.7  # seconds for level/motion calculation

# =============================================================================
# FLASK APP
# =============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'gesture_production_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# =============================================================================
# LOAD MODEL METADATA
# =============================================================================
with open(META_PATH, 'r') as f:
    meta = json.load(f)
    LABELS = meta['labels']
    T = meta['T']  # 60 frames
    R = meta['R']  # 128 range bins
    MEAN = meta['mean']
    STD = meta['std']

print(f"[INFO] Model config: T={T}, R={R}, labels={LABELS}")
print(f"[INFO] Normalization: mean={MEAN:.4f}, std={STD:.4f}")

# =============================================================================
# LOAD ONNX MODEL
# =============================================================================
ort_session = None
try:
    ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("[INFO] ONNX model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")


# =============================================================================
# GATING STATE
# =============================================================================
@dataclass
class GatingState:
    """State for motion gating pipeline"""
    # Baseline
    baseline: Optional[np.ndarray] = None
    baseline_peaks: List[float] = field(default_factory=list)
    baseline_motions: List[float] = field(default_factory=list)
    baseline_ready: bool = False
    
    # Thresholds (auto-calculated from baseline)
    level_th: float = 0.0
    motion_th: float = 0.0
    hold_th: float = 0.0
    motion_on: float = 0.0
    motion_off: float = 0.0
    hold_off: float = 0.0
    
    # Current state
    motion_state: int = 0  # 0=no motion, 1=motion detected
    hold_state: int = 0    # 0=not holding, 1=holding
    
    # Current measurements
    lvl: float = 0.0
    mot: float = 0.0
    
    # Buffers
    frame_buffer: deque = field(default_factory=lambda: deque(maxlen=T + 30))
    prev_frame: Optional[np.ndarray] = None
    
    # CNN smoothing
    prob_history: deque = field(default_factory=lambda: deque(maxlen=SMOOTHING_WINDOW))
    
    # Output
    gesture_out: str = "none"
    gesture_raw: Optional[str] = None
    all_confidences: Dict[str, float] = field(default_factory=dict)
    pmax: float = 0.0
    top2_margin: float = 0.0
    gate: str = "none"
    
    def reset(self):
        """Reset all state to initial values"""
        self.baseline = None
        self.baseline_peaks = []
        self.baseline_motions = []
        self.baseline_ready = False
        self.level_th = 0.0
        self.motion_th = 0.0
        self.hold_th = 0.0
        self.motion_on = 0.0
        self.motion_off = 0.0
        self.hold_off = 0.0
        self.motion_state = 0
        self.hold_state = 0
        self.lvl = 0.0
        self.mot = 0.0
        self.frame_buffer.clear()
        self.prev_frame = None
        self.prob_history.clear()
        self.gesture_out = "none"
        self.gesture_raw = None
        self.all_confidences = {}
        self.pmax = 0.0
        self.top2_margin = 0.0
        self.gate = "none"
        print("[INFO] State reset complete")


state = GatingState()

# Frame rate tracking
frame_count = 0
last_fps_time = time.time()
current_fps = 0.0

# Radar connection state
radar_connected = False
radar_client = None
data_length = 0


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================
def resample_to_r(frame: np.ndarray, target_r: int = R) -> np.ndarray:
    """Resample frame to target R bins using linear interpolation"""
    if len(frame) == target_r:
        return frame
    x_old = np.linspace(0, 1, len(frame))
    x_new = np.linspace(0, 1, target_r)
    return np.interp(x_new, x_old, frame).astype(np.float32)


def preprocess_for_cnn(frames: np.ndarray) -> np.ndarray:
    """
    Preprocess frames for CNN inference - MUST match training!
    Input: (T, data_length) raw frames
    Output: (1, 1, T, R) normalized tensor
    """
    # Step 1: Resample each frame to R bins
    resampled = np.array([resample_to_r(f, R) for f in frames])
    
    # Step 2: log1p(max(x, 0)) - clip negative values
    data = np.log1p(np.maximum(resampled, 0))
    
    # Step 3: Normalize with training mean/std
    data = (data - MEAN) / STD
    
    # Step 4: Reshape for CNN (batch, channels, T, R)
    return data.astype(np.float32).reshape(1, 1, T, R)


def calculate_mad(values: np.ndarray) -> float:
    """Calculate Median Absolute Deviation"""
    median = np.median(values)
    return np.median(np.abs(values - median))


# =============================================================================
# BASELINE FUNCTIONS
# =============================================================================
def reset_baseline():
    """Reset baseline state for recalibration"""
    global state
    state.baseline = None
    state.baseline_peaks = []
    state.baseline_motions = []
    state.baseline_ready = False
    state.motion_state = 0
    state.hold_state = 0
    state.gesture_out = "none"
    state.gesture_raw = None
    state.prob_history.clear()
    print("[INFO] Baseline reset - collecting new baseline...")


def update_baseline(frames: List[np.ndarray]):
    """Calculate baseline from collected frames"""
    global state
    
    if len(frames) < 10:
        return
    
    # Calculate median baseline per bin
    frames_arr = np.array(frames)
    state.baseline = np.median(frames_arr, axis=0)
    
    # Calculate peak levels for each frame (after baseline subtraction)
    peaks = []
    motions = []
    prev = None
    
    for frame in frames:
        subtracted = np.clip(frame - state.baseline, 0, None)
        peaks.append(np.max(subtracted))
        
        if prev is not None:
            delta = np.abs(frame - prev)
            motions.append(np.mean(delta))
        prev = frame
    
    peaks = np.array(peaks)
    motions = np.array(motions) if motions else np.array([0.0])
    
    # Calculate thresholds
    peak_median = np.median(peaks)
    peak_mad = calculate_mad(peaks)
    motion_median = np.median(motions)
    motion_mad = calculate_mad(motions)
    
    state.level_th = peak_median + K_LEVEL * peak_mad
    state.motion_th = motion_median + K_MOTION * motion_mad
    state.hold_th = max(state.level_th * HOLD_FACTOR, state.level_th + 6 * peak_mad)
    
    # Hysteresis thresholds
    state.motion_on = state.motion_th * HYSTERESIS_ON
    state.motion_off = state.motion_th * HYSTERESIS_OFF
    state.hold_off = state.hold_th * HOLD_OFF_FACTOR
    
    state.baseline_ready = True
    
    print(f"[INFO] Baseline calculated:")
    print(f"       level_th={state.level_th:.2f}, hold_th={state.hold_th:.2f}")
    print(f"       motion_th={state.motion_th:.4f}, on={state.motion_on:.4f}, off={state.motion_off:.4f}")


# =============================================================================
# GATING STATE MACHINE
# =============================================================================
def run_gating(frame: np.ndarray, tail_frames: List[np.ndarray]) -> str:
    """
    Run motion gating state machine.
    Returns: "none", "hold", or "cnn"
    """
    global state
    
    if not state.baseline_ready or state.baseline is None:
        return "none"
    
    # Calculate current level and motion from tail
    if len(tail_frames) < 3:
        return "none"
    
    tail_arr = np.array(tail_frames)
    subtracted = np.clip(tail_arr - state.baseline, 0, None)
    
    # Level: median of max peaks in tail
    peaks = np.max(subtracted, axis=1)
    state.lvl = np.median(peaks)
    
    # Motion: median of mean absolute frame deltas in tail
    deltas = np.abs(np.diff(tail_arr, axis=0))
    motions = np.mean(deltas, axis=1)
    state.mot = np.median(motions) if len(motions) > 0 else 0.0
    
    # State machine transitions
    prev_motion_state = state.motion_state
    
    # Motion detection with hysteresis
    if state.motion_state == 0:
        if state.mot > state.motion_on:
            state.motion_state = 1
    else:
        if state.mot < state.motion_off:
            state.motion_state = 0
    
    # Hold detection
    if state.lvl > state.hold_th:
        state.hold_state = 1
    elif state.lvl < state.hold_off:
        state.hold_state = 0
    
    # Determine gate output
    if state.motion_state == 0:
        # No motion - only none or hold
        if state.hold_state == 1:
            state.gate = "hold"
            return "hold"
        else:
            state.gate = "none"
            return "none"
    else:
        # Motion detected - run CNN
        state.gate = "cnn"
        return "cnn"


# =============================================================================
# CNN INFERENCE
# =============================================================================
def run_cnn_inference(frames: np.ndarray) -> tuple:
    """
    Run CNN inference on frames.
    Returns: (gesture, confidence, all_probs, top2_margin)
    """
    if ort_session is None or len(frames) < T:
        return None, 0.0, {}, 0.0
    
    try:
        # Take last T frames
        input_frames = frames[-T:]
        
        # Preprocess for CNN (NO baseline subtraction here!)
        tensor = preprocess_for_cnn(input_frames)
        
        # Run inference
        outputs = ort_session.run(None, {'x': tensor})
        logits = outputs[0][0]
        
        # Softmax (temperature=1.0, no scaling)
        e_x = np.exp(logits - np.max(logits))
        probs = e_x / e_x.sum()
        
        return probs, logits
        
    except Exception as e:
        print(f"[ERROR] CNN inference failed: {e}")
        return None, None


def apply_smoothing(probs: np.ndarray) -> tuple:
    """Apply temporal smoothing to CNN probabilities"""
    global state
    
    state.prob_history.append(probs)
    
    if len(state.prob_history) >= 2:
        avg_probs = np.mean(list(state.prob_history), axis=0)
    else:
        avg_probs = probs
    
    # Get prediction
    pred_idx = np.argmax(avg_probs)
    gesture = LABELS[pred_idx]
    pmax = float(avg_probs[pred_idx])
    
    # Calculate top-2 margin
    sorted_probs = np.sort(avg_probs)[::-1]
    top2_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else pmax
    
    # All confidences
    all_conf = {LABELS[i]: float(avg_probs[i]) for i in range(len(LABELS))}
    
    return gesture, pmax, all_conf, top2_margin


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def process_frame(raw_frame: np.ndarray):
    """Process a single radar frame through the full pipeline"""
    global state, frame_count, last_fps_time, current_fps
    
    frame_count += 1
    
    # FPS calculation
    now = time.time()
    if now - last_fps_time >= 1.0:
        current_fps = frame_count / (now - last_fps_time)
        frame_count = 0
        last_fps_time = now
    
    # Ensure float32
    frame = raw_frame.astype(np.float32)
    
    # Add to buffer
    state.frame_buffer.append(frame)
    
    # Baseline collection phase
    if not state.baseline_ready:
        baseline_frames = int(BASELINE_DURATION * UPDATE_RATE)
        if len(state.frame_buffer) >= baseline_frames:
            update_baseline(list(state.frame_buffer)[-baseline_frames:])
        state.gesture_out = "none"
        state.gesture_raw = None
        state.gate = "baseline"
        return
    
    # Get tail frames for gating
    tail_frames_count = int(TAIL_DURATION * UPDATE_RATE)
    tail_frames = list(state.frame_buffer)[-tail_frames_count:] if len(state.frame_buffer) >= tail_frames_count else list(state.frame_buffer)
    
    # Run gating
    gate = run_gating(frame, tail_frames)
    
    if gate == "none":
        state.gesture_out = "none"
        state.gesture_raw = None
        state.pmax = 0.0
        state.top2_margin = 0.0
        state.all_confidences = {label: 0.0 for label in LABELS}
        state.prob_history.clear()
        
    elif gate == "hold":
        state.gesture_out = "hold"
        state.gesture_raw = None
        state.pmax = 0.0
        state.top2_margin = 0.0
        state.all_confidences = {label: 0.0 for label in LABELS}
        state.prob_history.clear()
        
    else:  # gate == "cnn"
        # Run CNN on raw frames (not baseline-subtracted!)
        if len(state.frame_buffer) >= T:
            raw_frames = np.array(list(state.frame_buffer)[-T:])
            result = run_cnn_inference(raw_frames)
            
            if result[0] is not None:
                probs = result[0]
                gesture, pmax, all_conf, margin = apply_smoothing(probs)
                
                state.gesture_raw = gesture
                state.gesture_out = gesture
                state.pmax = pmax
                state.top2_margin = margin
                state.all_confidences = all_conf
    
    # Store previous frame for motion calculation
    state.prev_frame = frame


# =============================================================================
# EMIT DATA TO FRONTEND
# =============================================================================
def emit_data(frame: np.ndarray):
    """Emit rich data to frontend via SocketIO"""
    # Resample frame for display (R=128)
    frame_display = resample_to_r(frame, R).tolist()
    
    # Helper to convert numpy types to Python native
    def to_float(val):
        if isinstance(val, (np.floating, np.float32, np.float64)):
            return float(val)
        return val
    
    data = {
        # Frame data
        'frame': frame_display,
        
        # Gesture outputs
        'gesture_out': state.gesture_out,
        'gesture_raw': state.gesture_raw,
        'gate': state.gate,
        
        # Confidences (convert numpy floats)
        'pmax': to_float(state.pmax),
        'top2_margin': to_float(state.top2_margin),
        'all_confidences': {k: to_float(v) for k, v in state.all_confidences.items()},
        
        # Gating state
        'motion_state': int(state.motion_state),
        'hold_state': int(state.hold_state),
        'lvl': to_float(state.lvl),
        'mot': to_float(state.mot),
        
        # Thresholds
        'level_th': to_float(state.level_th),
        'hold_th': to_float(state.hold_th),
        'motion_th': to_float(state.motion_th),
        'motion_on': to_float(state.motion_on),
        'motion_off': to_float(state.motion_off),
        'hold_off': to_float(state.hold_off),
        
        # Status
        'connected': radar_connected,
        'fps': round(current_fps, 1),
        'baseline_ready': state.baseline_ready,
        
        # Config for frontend
        'range_start': RANGE_START,
        'range_end': RANGE_END,
    }
    
    socketio.emit('radar_data', data)


# =============================================================================
# RADAR CLIENT (EXPLORATION PROTOCOL)
# =============================================================================
class AcconeerExplorationClient:
    """Acconeer A111 client using exploration server protocol"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.client = None
        self.running = False
        self.thread = None
        self.data_length = 0
    
    def connect_and_start(self) -> bool:
        """Connect to exploration server and start streaming"""
        global radar_connected, data_length
        
        try:
            import acconeer.exptool as et
            
            print(f"[INFO] Connecting to {self.host}:{self.port} (exploration protocol)...")
            
            # Create client with explicit exploration protocol
            self.client = et.a111.Client(
                protocol="exploration",
                link="socket",
                host=self.host,
                port=self.port
            )
            
            # Connect
            self.client.connect()
            print("[INFO] Connected to exploration server!")
            
            # Configure envelope service
            cfg = et.a111.EnvelopeServiceConfig()
            cfg.sensor = [1]
            cfg.range_interval = [RANGE_START, RANGE_END]
            cfg.update_rate = UPDATE_RATE
            # Do NOT set profile (leave as default/unset)
            
            print(f"[INFO] Configuring: range=[{RANGE_START}, {RANGE_END}], rate={UPDATE_RATE}Hz")
            
            # Setup session
            session_info = self.client.setup_session(cfg)
            self.data_length = session_info.get("data_length", 0)
            data_length = self.data_length
            print(f"[INFO] Session configured, data_length={self.data_length}")
            
            # Start streaming
            self.client.start_session()
            print("[INFO] Streaming started!")
            
            radar_connected = True
            self.running = True
            
            # Start receive thread
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to connect: {e}")
            import traceback
            traceback.print_exc()
            radar_connected = False
            return False
    
    def stop(self):
        """Stop streaming and disconnect"""
        global radar_connected
        self.running = False
        
        if self.client:
            try:
                self.client.stop_session()
                self.client.disconnect()
            except:
                pass
        
        if self.thread:
            self.thread.join(timeout=2)
        
        radar_connected = False
        self.client = None
        print("[INFO] Radar disconnected")
    
    def _receive_loop(self):
        """Main receive loop"""
        global radar_connected
        errors = 0
        
        while self.running and radar_connected:
            try:
                data_info, data = self.client.get_next()
                
                if data is not None:
                    # Get frame from sensor 1
                    frame = data[0] if len(data.shape) > 1 else data
                    
                    # Process through pipeline
                    process_frame(frame)
                    
                    # Emit to frontend
                    emit_data(frame)
                    
                    errors = 0
                    
            except Exception as e:
                errors += 1
                print(f"[ERROR] Receive error: {e}")
                if errors > 10:
                    print("[ERROR] Too many errors, stopping...")
                    break
                time.sleep(0.05)
        
        radar_connected = False


# =============================================================================
# SIMULATOR (Same pipeline)
# =============================================================================
class RadarSimulator:
    """Simulates radar data through the same pipeline"""
    
    def __init__(self):
        self.running = False
        self.thread = None
        self.frame_count = 0
        self.patterns = ['none', 'hold', 'push', 'pull', 'tap', 'wave']
        self.pattern_idx = 0
        self.sim_data_length = 724  # Simulate realistic data length
    
    def start(self):
        global radar_connected, data_length
        self.running = True
        radar_connected = True
        data_length = self.sim_data_length
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[INFO] Simulator started")
    
    def stop(self):
        global radar_connected
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        radar_connected = False
    
    def _gen_frame(self, t: int, pattern: str) -> np.ndarray:
        """Generate a simulated radar frame"""
        n = self.sim_data_length
        base = np.random.randn(n) * 2 + 50  # Base noise around 50
        
        if pattern == 'none':
            return base
        
        # Add signal at different positions based on gesture
        if pattern == 'hold':
            center = n // 2
            signal = 100 * np.exp(-((np.arange(n) - center) ** 2) / (2 * 50**2))
            return base + signal
        
        elif pattern == 'push':
            # Moving toward sensor
            center = int(n * 0.7 - (t % 60) * 2)
            center = max(n // 4, center)
            signal = 150 * np.exp(-((np.arange(n) - center) ** 2) / (2 * 40**2))
            return base + signal
        
        elif pattern == 'pull':
            # Moving away from sensor
            center = int(n * 0.3 + (t % 60) * 2)
            center = min(3 * n // 4, center)
            signal = 150 * np.exp(-((np.arange(n) - center) ** 2) / (2 * 40**2))
            return base + signal
        
        elif pattern == 'tap':
            # Quick in-out movement
            phase = (t % 30) / 30
            center = int(n // 2 - 100 * np.sin(phase * np.pi))
            intensity = 200 * np.sin(phase * np.pi)
            signal = intensity * np.exp(-((np.arange(n) - center) ** 2) / (2 * 30**2))
            return base + signal
        
        elif pattern == 'wave':
            # Side-to-side oscillation (intensity changes)
            center = n // 2
            intensity = 80 + 60 * np.sin(t * 0.2)
            signal = intensity * np.exp(-((np.arange(n) - center) ** 2) / (2 * 60**2))
            return base + signal
        
        return base
    
    def _loop(self):
        global radar_connected
        
        # First 2 seconds: generate baseline (no gesture)
        baseline_frames = int(BASELINE_DURATION * UPDATE_RATE)
        
        while self.running:
            self.frame_count += 1
            
            # Change pattern every 3 seconds (after baseline)
            if self.frame_count > baseline_frames:
                if (self.frame_count - baseline_frames) % (3 * UPDATE_RATE) == 0:
                    self.pattern_idx = (self.pattern_idx + 1) % len(self.patterns)
            
            pattern = 'none' if self.frame_count <= baseline_frames else self.patterns[self.pattern_idx]
            frame = self._gen_frame(self.frame_count, pattern).astype(np.float32)
            
            # Process through same pipeline as real radar
            process_frame(frame)
            emit_data(frame)
            
            time.sleep(1.0 / UPDATE_RATE)
        
        radar_connected = False


# =============================================================================
# GLOBAL RADAR INSTANCE
# =============================================================================
radar = None


# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/game')
def game():
    return render_template('game.html')


@app.route('/game2')
def game2():
    return render_template('game2.html')


@app.route('/games')
def games_hub():
    return render_template('games.html')


@app.route('/punch')
def punch_game():
    return render_template('punch.html')


@app.route('/space')
def space_game():
    return render_template('space.html')


@app.route('/pong')
def pong_game():
    return render_template('pong.html')


@app.route('/status')
def status():
    return jsonify({
        'connected': radar_connected,
        'baseline_ready': state.baseline_ready,
        'gesture_out': state.gesture_out,
        'fps': current_fps
    })


@app.route('/config')
def config():
    return jsonify({
        'labels': LABELS,
        'T': T,
        'R': R,
        'range_start': RANGE_START,
        'range_end': RANGE_END,
        'update_rate': UPDATE_RATE
    })


# =============================================================================
# SOCKETIO EVENTS
# =============================================================================
@socketio.on('connect')
def on_connect():
    print("[INFO] Browser client connected")
    emit('status', {
        'connected': radar_connected,
        'baseline_ready': state.baseline_ready
    })


@socketio.on('disconnect')
def on_disconnect():
    print("[INFO] Browser client disconnected")


@socketio.on('start_radar')
def on_start(data=None):
    global radar, state
    
    simulate = data.get('simulate', False) if data else False
    
    # Stop existing
    if radar:
        radar.stop()
        time.sleep(0.3)
    
    # Reset state (don't create new object - threads need same reference)
    state.reset()
    
    if simulate:
        print("[INFO] Starting simulator...")
        radar = RadarSimulator()
        radar.start()
        emit('status', {'connected': True, 'mode': 'simulation'})
    else:
        print("[INFO] Connecting to A111 exploration server...")
        radar = AcconeerExplorationClient(PI_HOST, PI_PORT)
        if radar.connect_and_start():
            emit('status', {'connected': True, 'mode': 'live'})
        else:
            print("[WARN] Falling back to simulator")
            radar = RadarSimulator()
            radar.start()
            emit('status', {'connected': True, 'mode': 'simulation (fallback)'})


@socketio.on('stop_radar')
def on_stop():
    global radar, state
    if radar:
        radar.stop()
        radar = None
    state = GatingState()
    emit('status', {'connected': False})


@socketio.on('rebaseline')
def on_rebaseline():
    """Re-capture baseline"""
    reset_baseline()
    emit('status', {'baseline_ready': False, 'message': 'Recapturing baseline...'})


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print(" GESTURE RADAR - Production Pipeline")
    print("=" * 60)
    print(f" Pi Server: {PI_HOST}:{PI_PORT}")
    print(f" Range: [{RANGE_START}, {RANGE_END}] m @ {UPDATE_RATE} Hz")
    print(f" Model: {T}x{R}, labels={LABELS}")
    print(f" Server: http://localhost:{SERVER_PORT}")
    print("=" * 60)
    
    socketio.run(app, host='0.0.0.0', port=SERVER_PORT, debug=False)
