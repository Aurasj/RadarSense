"""
serverV2.py  —  The Central Hub
=================================
Strict contract: NO DSP, NO AI math.
This file only wires hardware → engine → web → Unity.

Responsibilities
─────────────────
  1. Connect to the Acconeer A111 radar via TCP (Raspberry Pi).
  2. Feed raw frames through GestureEngine (all ML lives there).
  3. Stream results to the browser dashboard over Flask-SocketIO.
  4. Forward confirmed gesture events to Unity and the optional event monitor over UDP.

Data flow (all in one background thread)
─────────────────────────────────────────
  AcconeerExplorationClient.get_next_frame()
        │  raw float32 (N_bins,)
        ▼
  frame_buffer  deque(maxlen=60)
        │
        ▼
  GestureEngine.predict(frame_buffer)
        │  gesture, confidence, probs
        ├──► SocketIO  "radar_data"    → browser waterfall
        ├──► SocketIO  "prediction"    → browser gesture monitor / bars
        └──► UDP  5005 (Unity)  +  5006 (Event Monitor)   [only if != 'none']
"""

import logging
import os
import socket
import sys
import threading
import time
from collections import deque

import numpy as np
from flask import Flask, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from gesture_engine import GestureEngine

# ──────────────────────────────────────────────────────────────────────────────
# Silence Flask's noisy request logger
# ──────────────────────────────────────────────────────────────────────────────
logging.getLogger("werkzeug").setLevel(logging.ERROR)
log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (override via environment variables)
# ══════════════════════════════════════════════════════════════════════════════

PI_HOST     = os.environ.get("PI_HOST",     "192.168.0.106")
PI_PORT     = int(os.environ.get("PI_PORT",       6110))
SERVER_PORT = int(os.environ.get("SERVER_PORT",   5002))
RANGE_START = float(os.environ.get("RANGE_START",  0.20))   # metres
RANGE_END   = float(os.environ.get("RANGE_END",    0.55))   # metres
UPDATE_RATE = int(os.environ.get("UPDATE_RATE",      30))   # fps

UNITY_IP      = os.environ.get("UNITY_IP",    "127.0.0.1")
UNITY_PORT    = int(os.environ.get("UNITY_PORT",    5005))
MONITOR_PORT  = int(os.environ.get("MONITOR_PORT",  5006))

# Production terminal hygiene:
# 0 = quiet terminal (recommended for demo), 1 = print every confirmed gesture event.
VERBOSE_GESTURES = os.environ.get("VERBOSE_GESTURES", "0") == "1"

# Sliding window depth — matches GestureEngine's TARGET_FRAMES (60)
WINDOW_SIZE = 60

# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION SETUP
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "antiradar_production_secret")
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)

# ── GestureEngine — the ONLY source of inference in this codebase ─────────────
engine = GestureEngine()
print(f"[Server] GestureEngine ready  |  labels={engine.labels}")

# ── Sliding window (thread-safe for single-producer / single-consumer) ────────
frame_buffer: deque = deque(maxlen=WINDOW_SIZE)

# ── UDP socket — for Unity and local monitor ──────────────────────────────────
_udp: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ── Thread-control globals ────────────────────────────────────────────────────
_radar_connected  = False
_connected_clients = 0
_active_client    = None          # AcconeerExplorationClient instance
_radar_thread     = None          # threading.Thread
_stop_event       = threading.Event()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _udp_send(gesture: str) -> None:
    """
    Send a gesture string to Unity and the optional local event monitor.
    Best-effort UDP — a failed send is logged but never raises.
    """
    payload = gesture.encode()
    for target in ((UNITY_IP, UNITY_PORT), (UNITY_IP, MONITOR_PORT)):
        try:
            _udp.sendto(payload, target)
        except OSError as exc:
            log.warning("UDP send to %s failed: %s", target, exc)


def _emit_status(connected: bool, mode: str = "") -> None:
    """Broadcast radar connection status to all browser clients."""
    socketio.emit("radar_status", {
        "connected": connected,
        "mode":      mode or ("live" if connected else "offline"),
    })


# ══════════════════════════════════════════════════════════════════════════════
# ACCONEER CLIENT  (hardware abstraction — no AI logic here)
# ══════════════════════════════════════════════════════════════════════════════

class AcconeerExplorationClient:
    """
    Thin wrapper around the Acconeer exptool A111 SDK.

    Provides two methods the radar thread cares about:
      connect_and_start() → bool
      get_next_frame()    → np.ndarray | None
      stop()
    """

    def __init__(self) -> None:
        self._client = None
        self.running  = False

    def connect_and_start(self) -> bool:
        """Single connection + session start attempt.  Returns True on success."""
        try:
            import acconeer.exptool as et

            self._client = et.a111.Client(
                protocol="exploration",
                link="socket",
                host=PI_HOST,
                port=PI_PORT,
            )
            self._client.connect()

            cfg                = et.a111.EnvelopeServiceConfig()
            cfg.sensor         = [1]
            cfg.range_interval = [RANGE_START, RANGE_END]
            cfg.update_rate    = UPDATE_RATE
            self._client.setup_session(cfg)
            self._client.start_session()

            self.running = True
            print(f"[Server] Radar connected  →  {PI_HOST}:{PI_PORT}")
            return True

        except Exception as exc:
            print(f"[Server] Connection failed: {exc}")
            self._client = None
            self.running  = False
            return False

    def get_next_frame(self) -> "np.ndarray | None":
        """
        Fetch one envelope frame.

        Returns a 1-D float32 array of shape (N_bins,), or None on error.
        """
        try:
            _, data = self._client.get_next()
            if data is None:
                return None
            frame = data[0] if data.ndim > 1 else data
            return np.asarray(frame).ravel().astype(np.float32)
        except Exception:
            return None

    def stop(self) -> None:
        """Gracefully stop the session and disconnect."""
        self.running = False
        if self._client is None:
            return
        for fn in (self._client.stop_session, self._client.disconnect):
            try:
                fn()
                time.sleep(0.5)
            except Exception as exc:
                log.debug("Radar stop/disconnect warning: %s", exc)
        self._client = None


# ══════════════════════════════════════════════════════════════════════════════
# RADAR BACKGROUND THREAD
# ══════════════════════════════════════════════════════════════════════════════

def _radar_loop() -> None:
    """
    Main acquisition thread.

    One iteration = one radar frame:
      1. fetch raw frame from radar
      2. append to sliding window
      3. emit 'radar_data'  → browser waveform / waterfall
      4. call engine.predict()
      5. emit 'prediction'  → browser gesture monitor
      6. if gesture != 'none': UDP to Unity + Event Monitor
    """
    global _radar_connected, _active_client

    client = AcconeerExplorationClient()
    _active_client = client

    # ── Retry connect up to 3 times ───────────────────────────────────────────
    connected = False
    for attempt in range(1, 4):
        if _stop_event.is_set():
            return
        _emit_status(False, f"connecting ({attempt}/3)")
        if client.connect_and_start():
            connected = True
            break
        print(f"[Server] Attempt {attempt}/3 failed — retrying in 1.5 s …")
        time.sleep(1.5)

    if not connected:
        print("[Server] Could not connect after 3 attempts.")
        _emit_status(False, "error")
        _active_client = None
        return

    _radar_connected = True
    _emit_status(True, "live")

    errors = 0   # consecutive null-frame counter
    last_frame_ts = time.time()
    fps_ema = 0.0

    while client.running and not _stop_event.is_set():

        # ── 1. Acquire frame ──────────────────────────────────────────────────
        raw = client.get_next_frame()

        if raw is None:
            errors += 1
            if errors > 20:
                print("[Server] Too many consecutive errors — stopping.")
                break
            time.sleep(0.01)
            continue
        errors = 0

        # ── 2. Buffer frame ───────────────────────────────────────────────────
        frame_buffer.append(raw)

        now_ts = time.time()
        inst_fps = 1.0 / max(now_ts - last_frame_ts, 1e-6)
        last_frame_ts = now_ts
        fps_ema = inst_fps if fps_ema <= 0 else (0.85 * fps_ema + 0.15 * inst_fps)

        # ── 3. Emit raw waveform to browser ───────────────────────────────────
        # The frontend uses this for the live waterfall / envelope chart.
        socketio.emit("radar_data", {
            "data":        raw.tolist(),
            "range_start": RANGE_START,
            "range_end":   RANGE_END,
            "connected":   True,
        })

        # Wait for the window to fill before running inference
        if len(frame_buffer) < WINDOW_SIZE:
            continue

        # ── 4. Inference (all logic inside GestureEngine) ─────────────────────
        # Returns: (display_gesture, confidence, probs, is_event)
        #  - display_gesture: FSM-filtered, visually persisted for the UI
        #  - is_event: True ONLY on the frame the gesture fires (for UDP pulse)
        gesture, confidence, probs, is_event = engine.predict(frame_buffer)

        # ── 5. Emit prediction to browser ─────────────────────────────────────
        # fsm_state includes raw_top and raw_conf — what the model ACTUALLY
        # sees right now (pre-FSM).  The dashboard can display both:
        #   - Gesture Monitor: shows 'gesture' (FSM-filtered, persistent)
        #   - Confidence Bars: shows 'probabilities' (EMA-smoothed)
        #   - Raw indicator:   shows fsm.raw_top / fsm.raw_conf
        fsm = engine.fsm_state
        socketio.emit("prediction", {
            "gesture":          gesture,
            "confidence":       round(confidence, 4),
            "probabilities":    probs,
            "raw_probabilities": probs,
            "fsm":              fsm,
            "debug":            fsm,
            "fps":              round(fps_ema, 1),
            "is_event":         bool(is_event),
        })

        # ── 6. UDP bridge to Unity + Event Monitor ──────────────────────────────────
        # is_event is True on EXACTLY ONE frame per gesture.
        # This guarantees Unity receives a single clean pulse.
        if is_event and gesture != "none":
            if VERBOSE_GESTURES:
                print(f"[Server] Event → {gesture.upper()}  conf={confidence:.3f}")
            _udp_send(gesture)

        # Yield to avoid 100% CPU; the SDK's blocking get_next() paces the loop
        time.sleep(0)

    # ── Teardown ──────────────────────────────────────────────────────────────
    _radar_connected = False
    _active_client   = None
    client.stop()
    _emit_status(False, "offline")
    print("[Server] Radar thread stopped.")


def _stop_radar() -> None:
    """Signal the radar thread to exit and wait for it to join."""
    global _radar_thread, _active_client, _radar_connected

    _stop_event.set()

    if _active_client:
        _active_client.stop()
        _active_client = None

    if _radar_thread and _radar_thread.is_alive():
        _radar_thread.join(timeout=2.0)

    _radar_connected = False
    _radar_thread    = None


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def api_status():
    return jsonify({
        "connected":  _radar_connected,
        "labels":     engine.labels,
        "fsm":        engine.fsm_state,
        "port":       SERVER_PORT,
    })


# ══════════════════════════════════════════════════════════════════════════════
# SOCKET.IO EVENTS
# ══════════════════════════════════════════════════════════════════════════════

@socketio.on("connect")
def on_connect():
    global _connected_clients
    _connected_clients += 1
    print(f"[Server] Browser connected  (total: {_connected_clients})")
    # Immediately tell the new client the current radar state
    emit("radar_status", {
        "connected": _radar_connected,
        "mode":      "live" if _radar_connected else "offline",
    })


@socketio.on("disconnect")
def on_disconnect(reason=None):
    global _connected_clients
    _connected_clients = max(0, _connected_clients - 1)
    print(f"[Server] Browser disconnected  (total: {_connected_clients})")
    if _connected_clients == 0:
        _stop_radar()


@socketio.on("get_status")
def handle_get_status():
    emit("radar_status", {
        "connected": _radar_connected,
        "mode":      "live" if _radar_connected else "offline",
    })


@socketio.on("start_radar")
def handle_start_radar():
    global _radar_thread

    # Clean up any existing session before starting a fresh one
    _stop_radar()
    frame_buffer.clear()
    engine.reset()

    _stop_event.clear()
    _radar_thread = threading.Thread(target=_radar_loop, daemon=True)
    _radar_thread.start()

    emit("radar_status", {"connected": False, "mode": "starting…"})


@socketio.on("stop_radar")
def handle_stop_radar():
    _stop_radar()
    frame_buffer.clear()
    engine.reset()
    # Brief pause so the Pi hardware releases its session before the next connect
    time.sleep(2.5)
    emit("radar_status", {"connected": False, "mode": "offline"})


@socketio.on("rebase")
@socketio.on("rebaseline")
def handle_rebase():
    """
    Reset the engine FSM and EMA without stopping the radar thread.
    Use after repositioning the sensor or when spurious detections occur.
    """
    engine.reset()
    frame_buffer.clear()
    print("[Server] Engine reset (rebase).")
    emit("radar_status", {"connected": _radar_connected, "mode": "live"})


@socketio.on_error()
def on_error(exc):
    log.error("SocketIO error: %s", exc)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import webbrowser

    url = f"http://localhost:{SERVER_PORT}"

    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        print("=" * 65)
        print("  ANTRADAR  —  Production Control Hub")
        print(f"  Dashboard  →  {url}")
        print(f"  Radar Pi   →  {PI_HOST}:{PI_PORT}")
        print(f"  Unity UDP  →  {UNITY_IP}:{UNITY_PORT}")
        print(f"  Event Mon. →  {UNITY_IP}:{MONITOR_PORT}")
        print(f"  Labels     →  {engine.labels}")
        print(f"  Verbose    →  {'on' if VERBOSE_GESTURES else 'off'}")
        print("=" * 65)
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        app.jinja_env.auto_reload           = True
        app.config["TEMPLATES_AUTO_RELOAD"] = True
        socketio.run(
            app,
            host="0.0.0.0",
            port=SERVER_PORT,
            debug=False,
            allow_unsafe_werkzeug=True,
        )
    except KeyboardInterrupt:
        print("\n[Server] Shutting down …")
        _stop_radar()
        sys.exit(0)