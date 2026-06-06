"""
Main RadarSense server.

Connects the radar, dashboard and gesture engine.
Streams radar data to the UI and publishes confirmed gesture events.
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

# Silence Flask's per-request log spam so the terminal stays readable.
logging.getLogger("werkzeug").setLevel(logging.ERROR)
log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ── Configuration ─────────────────────────────────────────────────────────────
# All of these can be overridden with environment variables.

PI_HOST     = os.environ.get("PI_HOST",     "auraspi")
PI_PORT     = int(os.environ.get("PI_PORT",       6110))
SERVER_PORT = int(os.environ.get("SERVER_PORT",   5002))
RANGE_START = float(os.environ.get("RANGE_START",  0.20))   # metres
RANGE_END   = float(os.environ.get("RANGE_END",    0.55))   # metres
UPDATE_RATE = int(os.environ.get("UPDATE_RATE",      30))   # fps

UNITY_IP      = os.environ.get("UNITY_IP",    "127.0.0.1")
UNITY_PORT    = int(os.environ.get("UNITY_PORT",    5005))
MONITOR_PORT  = int(os.environ.get("MONITOR_PORT",  5006))

# Print confirmed gestures to the terminal when enabled.
VERBOSE_GESTURES = os.environ.get("VERBOSE_GESTURES", "0") == "1"

# If set to 1, the radar stops automatically when all browser tabs close.
AUTO_STOP_ON_NO_CLIENTS = os.environ.get("AUTO_STOP_ON_NO_CLIENTS", "0") == "1"

# The dashboard gets a downsampled 128-bin version of each frame.
# The ML engine always receives the full raw frame.
DASHBOARD_BINS = int(os.environ.get("DASHBOARD_BINS", "128"))

# Must match GestureEngine's T=60 (sliding window size).
WINDOW_SIZE = 60

# ── Flask + SocketIO setup ────────────────────────────────────────────────────

_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WEB_DIR = os.path.join(_BASE_DIR, "web")

app = Flask(
    __name__,
    template_folder=os.path.join(_WEB_DIR, "templates"),
    static_folder=os.path.join(_WEB_DIR, "static"),
    static_url_path="/static",
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "radarsense_production_secret")
CORS(app)
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)

# GestureEngine is the only place where inference happens.
engine = GestureEngine()
print(f"[Server] GestureEngine ready  |  labels={engine.labels}")

# Holds the last 60 radar frames for CNN inference.
# deque.append and iteration are safe across threads in CPython.
frame_buffer: deque = deque(maxlen=WINDOW_SIZE)

# Single UDP socket used for both Unity and the local event monitor.
_udp: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Thread control globals
_radar_connected   = False
_connected_clients = 0
_active_client     = None   # current AcconeerExplorationClient instance
_radar_thread      = None   # background acquisition thread
_stop_event        = threading.Event()
_clients_lock      = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _resample_1d(x: np.ndarray, out_bins: int) -> np.ndarray:
    """Downsample a radar frame to out_bins for the dashboard. The ML engine always gets the raw frame."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if out_bins <= 0 or x.size == out_bins:
        return x
    src = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, out_bins, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def _udp_send(gesture: str) -> None:
    """Send the gesture label to both UDP targets. Errors are logged, not raised."""
    payload = gesture.encode()
    for target in ((UNITY_IP, UNITY_PORT), (UNITY_IP, MONITOR_PORT)):
        try:
            _udp.sendto(payload, target)
        except OSError as exc:
            log.warning("UDP send to %s failed: %s", target, exc)


def _emit_status(connected: bool, mode: str = "") -> None:
    """Push the current radar connection state to all browser clients."""
    socketio.emit("radar_status", {
        "connected": connected,
        "mode":      mode or ("live" if connected else "offline"),
    })


# ── Acconeer client ───────────────────────────────────────────────────────────

class AcconeerExplorationClient:
    """
    Thin wrapper around the Acconeer exptool A111 SDK.
    Used by the radar thread via three methods:
      connect_and_start() -> bool
      get_next_frame()    -> ndarray | None
      stop()
    """

    def __init__(self) -> None:
        self._client = None
        self.running  = False

    def connect_and_start(self) -> bool:
        """Try to connect and start a session. Returns True on success."""
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

            if self._client is not None:
                try:
                    self._client.disconnect()
                except Exception:
                    pass

            self._client = None
            self.running = False
            return False

    def get_next_frame(self) -> "np.ndarray | None":
        """
        Fetch one envelope frame from the radar.
        Returns a 1D float32 array of shape (N_bins,), or None on error.
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
        """Stop the session and disconnect from the Pi."""
        self.running = False
        if self._client is None:
            return
        # Sleep runs even if the Pi already closed its side.
        try:
            self._client.stop_session()
        except Exception as exc:
            log.debug("stop_session warning: %s", exc)
        finally:
            time.sleep(0.6)
        try:
            self._client.disconnect()
        except Exception as exc:
            log.debug("disconnect warning: %s", exc)
        finally:
            time.sleep(0.6)   # give the Pi time to fully close port 6110 before the next connect
        self._client = None


# ── Radar background thread ───────────────────────────────────────────────────

def _radar_loop() -> None:
    """
    Background radar acquisition loop.
    Reads frames from the Pi, runs inference, and pushes results to the browser
    and UDP targets. Stops when _stop_event is set.
    """
    global _radar_connected, _active_client

    client = AcconeerExplorationClient()
    _active_client = client

    # Try to connect up to 3 times before giving up.
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

    errors = 0
    last_frame_ts = time.time()
    fps_ema = float(UPDATE_RATE)  # start at the expected rate so the first frame is smoothed normally

    while client.running and not _stop_event.is_set():

        # Fetch the next frame.
        raw = client.get_next_frame()

        if raw is None:
            errors += 1
            if errors > 20:
                print("[Server] Too many consecutive errors — stopping.")
                break
            time.sleep(0.01)
            continue

        errors = 0

        # Append frame to the sliding window and compute FPS.
        frame_buffer.append(raw)

        now_ts = time.time()
        delta = now_ts - last_frame_ts

        # Clamp inst_fps to 2× UPDATE_RATE before feeding into EMA.
        # Buffered TCP frames can produce near-zero deltas, which would otherwise
        # contaminate the EMA and show unrealistic values in the dashboard for ~20+ frames.
        inst_fps = min(1.0 / max(delta, 1e-6), float(UPDATE_RATE * 2))
        last_frame_ts = now_ts
        fps_ema = 0.85 * fps_ema + 0.15 * inst_fps

        # Send the waveform to the browser for the waterfall display.
        vis = _resample_1d(raw, DASHBOARD_BINS)
        socketio.emit("radar_data", {
            "data":        vis.tolist(),
            "bins":        int(vis.size),
            "range_start": RANGE_START,
            "range_end":   RANGE_END,
            "connected":   True,
        })

        # Wait until we have a full window before running inference.
        if len(frame_buffer) < WINDOW_SIZE:
            continue

        # Run inference (GestureEngine handles all the ML logic).
        gesture, confidence, probs, is_event = engine.predict(frame_buffer)

        # Send the prediction to the browser.
        fsm = engine.fsm_state
        socketio.emit("prediction", {
            "gesture":           gesture,
            "confidence":        round(confidence, 4),
            "probabilities":     probs,
            "raw_probabilities": probs,
            "fsm":               fsm,
            "debug":             fsm,
            "fps":               round(fps_ema, 1),
            "is_event":          bool(is_event),
        })

        # Forward confirmed gesture over UDP.
        # is_event is True on exactly one frame per gesture.
        if is_event and gesture != "none":
            if VERBOSE_GESTURES:
                print(f"[Server] Event → {gesture.upper()}  conf={confidence:.3f}")
            _udp_send(gesture)

        # Yield so we don't pin the CPU. The SDK's get_next() is already blocking.
        time.sleep(0)

    # Teardown
    _radar_connected = False
    _active_client   = None
    client.stop()
    _emit_status(False, "offline")
    print("[Server] Radar thread stopped.")


def _stop_radar() -> None:
    """Signal the radar thread to stop and wait for it to finish."""
    global _radar_thread, _active_client, _radar_connected

    _stop_event.set()

    if _active_client:
        _active_client.stop()
        _active_client = None

    if _radar_thread and _radar_thread.is_alive():
        _radar_thread.join(timeout=2.0)

    _radar_connected = False
    _radar_thread    = None


# ── Flask routes ──────────────────────────────────────────────────────────────

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


# ── Socket.IO events ──────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    global _connected_clients

    with _clients_lock:
        _connected_clients += 1
        total_clients = _connected_clients

    print(f"[Server] Browser connected  (total: {total_clients})")

    # Send the current radar status immediately so the browser UI updates.
    emit("radar_status", {
        "connected": _radar_connected,
        "mode":      "live" if _radar_connected else "offline",
    })


@socketio.on("disconnect")
def on_disconnect(reason=None):
    global _connected_clients

    with _clients_lock:
        _connected_clients = max(0, _connected_clients - 1)
        total_clients = _connected_clients

    print(f"[Server] Browser disconnected  (total: {total_clients})")

    if total_clients == 0 and AUTO_STOP_ON_NO_CLIENTS:
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

    # Stop any existing session. The delay gives the Pi time to release the port.
    _stop_radar()
    time.sleep(1.5)

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
    # Tell the browser it's offline right away, before the hardware pause.
    emit("radar_status", {"connected": False, "mode": "offline"})
    # The Pi needs time to fully close its TCP session (TIME_WAIT state).
    # 3.5s is enough clearance before a new connect attempt can succeed.
    time.sleep(3.5)


@socketio.on("rebase")
@socketio.on("rebaseline")
def handle_rebase():
    """
    Reset the engine's FSM and EMA without stopping the radar.
    Useful if you reposition the sensor or get a lot of false positives.
    """
    engine.reset()
    frame_buffer.clear()
    print("[Server] Engine reset (rebase).")
    emit("radar_status", {"connected": _radar_connected, "mode": "live"})


@socketio.on_error()
def on_error(exc):
    log.error("SocketIO error: %s", exc)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser

    url = f"http://localhost:{SERVER_PORT}"

    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        print("=" * 65)
        print("  RadarSense  —  Production Control Hub")
        print(f"  Dashboard  →  {url}")
        print(f"  Radar Pi   →  {PI_HOST}:{PI_PORT}")
        print(f"  Unity UDP  →  {UNITY_IP}:{UNITY_PORT}")
        print(f"  Event Mon. →  {UNITY_IP}:{MONITOR_PORT}")
        print(f"  Labels     →  {engine.labels}")
        print(f"  Verbose    →  {'on' if VERBOSE_GESTURES else 'off'}")
        print(f"  Auto-stop  →  {'on' if AUTO_STOP_ON_NO_CLIENTS else 'off'}")
        print(f"  Dash bins  →  {DASHBOARD_BINS}")
        print("=" * 65)
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        app.jinja_env.auto_reload           = os.environ.get("TEMPLATE_RELOAD", "0") == "1"
        app.config["TEMPLATES_AUTO_RELOAD"] = app.jinja_env.auto_reload
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
        try:
            _udp.close()
        except OSError:
            pass
        sys.exit(0)