"""
radar_terminal_v2.py  –  Real-time gesture recognition on Acconeer A111
Conectare: 192.168.0.145:6110
Model:     gesture_cnn.pt  (TinyCNN, labels: none/hold/push/pull/tap/wave)

Pipeline:
  Phase 1 – Noise Gate  : ignora cadre unde range(raw) < 0.15
  Phase 2 – Temporal FSM: EMA(alpha=0.4) + debounce class-aware (Hold=8, rest=3)
  Phase 3 – Cooldown    : 25 cadre de pauza dupa orice gest (except Hold)

Afisaj terminal – o singura linie care se updateaza:
  [Gate: OPEN/CLOSED] [Top: Wave 88%] [FSM: Debounce 2/3] [Cool: 0]
Cand un gest e detectat:
  🚀 GESTURE: PUSH 🚀
"""

import sys, time, queue, threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import acconeer.exptool as et

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
HOST         = "192.168.0.145"
PORT         = 6110
MODEL_PATH   = "gesture_cnn.pt"

UPDATE_RATE  = 30.0   # Hz – radar frame rate
RANGE_START  = 0.20   # m
RANGE_END    = 0.60   # m
INFER_HZ     = 15.0   # predictii/sec

NOISE_GATE   = 0.15   # sub acest range raw → CLOSED
EMA_ALPHA    = 0.4    # smoothing probabilitati
MIN_CONF     = 0.55   # confidenta minima pentru a considera o clasa

DEBOUNCE     = {      # cadre consecutive necesare
    "hold": 8,
    "default": 3,
}
COOLDOWN_MAX = 25     # cadre de ignorat dupa gest

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
class TinyCNN(nn.Module):
    def __init__(self, n_classes: int = 6):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.h = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.0), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.h(self.f(x))


# ─────────────────────────────────────────────
#  PREPROCESSING  (identic cu training)
# ─────────────────────────────────────────────
def fix_T(X: np.ndarray, T: int) -> np.ndarray:
    if X.shape[0] == T:
        return X
    if X.shape[0] > T:
        return X[:T]
    pad = np.repeat(X[-1:], repeats=(T - X.shape[0]), axis=0)
    return np.concatenate([X, pad], axis=0)


def resample_range(X: np.ndarray, out_bins: int) -> np.ndarray:
    T, R = X.shape
    x_old = np.linspace(0.0, 1.0, R, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, out_bins, dtype=np.float32)
    Y = np.empty((T, out_bins), dtype=np.float32)
    for t in range(T):
        Y[t] = np.interp(x_new, x_old, X[t].astype(np.float32))
    return Y


def preprocess(X: np.ndarray, T: int, R: int, mean: float, std: float) -> np.ndarray:
    X = fix_T(X, T=T)
    X = resample_range(X, out_bins=R)
    X = np.log1p(np.maximum(X, 0.0).astype(np.float32))
    X = (X - mean) / (std + 1e-6)
    return X.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


# ─────────────────────────────────────────────
#  RADAR READER (thread separat)
# ─────────────────────────────────────────────
class RadarReader(threading.Thread):
    def __init__(self, client, q: queue.Queue, stop_evt: threading.Event):
        super().__init__(daemon=True)
        self.client    = client
        self.q         = q
        self.stop_evt  = stop_evt
        self.err       = None

    def run(self):
        try:
            while not self.stop_evt.is_set():
                _info, data = self.client.get_next()
                frame = np.asarray(data).squeeze().astype(np.float32).reshape(-1)
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                    self.q.put(frame)
        except Exception as exc:
            self.err = f"{type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────
GESTURE_ICONS = {
    "hold": "✋",
    "push": "👉",
    "pull": "👈",
    "tap":  "👆",
    "wave": "👋",
    "none": "—",
}

def bar(prob: float, width: int = 10) -> str:
    filled = int(round(prob * width))
    return "█" * filled + "░" * (width - filled)


def print_status(gate_open: bool, top_label: str, top_prob: float,
                 fsm_label: str, count: int, needed: int, cooldown: int):
    gate_str  = "\033[92mOPEN  \033[0m" if gate_open else "\033[91mCLOSED\033[0m"
    icon      = GESTURE_ICONS.get(top_label, "?")
    conf_str  = f"{icon} {top_label.upper():<5} {top_prob*100:5.1f}%  {bar(top_prob)}"

    if cooldown > 0:
        fsm_str = f"\033[93mCooldown {cooldown:2d}/{COOLDOWN_MAX}\033[0m     "
    elif fsm_label == "none" or not gate_open:
        fsm_str = "Idle                 "
    else:
        needed_disp = DEBOUNCE.get(fsm_label, DEBOUNCE["default"])
        fsm_str = f"\033[96mDebounce {count}/{needed_disp}\033[0m  [{fsm_label.upper():<5}]"

    line = (
        f"\r [Gate: {gate_str}]  "
        f"[Top: {conf_str}]  "
        f"[FSM: {fsm_str}]  "
        f"[Cool: {cooldown:2d}]   "
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def print_gesture(label: str):
    icon = GESTURE_ICONS.get(label, "🎯")
    sys.stdout.write(
        f"\n\033[1;92m  🚀 GESTURE: {label.upper()} {icon}  🚀\033[0m\n"
    )
    sys.stdout.flush()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    # ── Incarcare model ──────────────────────
    print(f"[INFO] Incarcare model: {MODEL_PATH}")
    pack   = torch.load(MODEL_PATH, map_location="cpu")
    labels = pack["labels"]         # ex: ['none','hold','push','pull','tap','wave']
    T      = int(pack["T"])         # 60
    R      = int(pack["R"])         # 128
    mean   = float(pack["mean"])
    std    = float(pack["std"])

    none_idx = labels.index("none") if "none" in labels else -1

    model = TinyCNN(n_classes=len(labels))
    model.load_state_dict(pack["state_dict"])
    model.eval()
    print(f"[INFO] Labels: {labels}  |  T={T}  R={R}  mean={mean:.3f}  std={std:.3f}")

    # ── Configurare radar ────────────────────
    print(f"[INFO] Conectare la {HOST}:{PORT} ...")
    cfg              = et.a111.EnvelopeServiceConfig()
    cfg.update_rate  = UPDATE_RATE
    cfg.range_interval = [RANGE_START, RANGE_END]

    client = et.a111.Client(protocol="exploration", link="socket", host=HOST)
    client.connect()
    session_info = client.setup_session(cfg)
    client.start_session()
    data_length = int(session_info["data_length"])
    print(f"[INFO] Sesiune activa | data_length={data_length} | range=[{RANGE_START},{RANGE_END}]m")
    print("[INFO] Ctrl+C pentru stop.\n")

    # ── Threading ────────────────────────────
    q        = queue.Queue(maxsize=500)
    stop_evt = threading.Event()
    reader   = RadarReader(client, q, stop_evt)
    reader.start()

    # ── Stare FSM ────────────────────────────
    buf          = deque(maxlen=T)
    ema_probs    = np.ones(len(labels), dtype=np.float32) / len(labels)

    gate_open    = False
    fsm_label    = "none"     # clasa pe care o debunsam curent
    fsm_count    = 0          # cadre consecutive cu aceasta clasa
    cooldown     = 0          # cadre ramase in cooldown

    infer_period = 1.0 / max(1e-6, INFER_HZ)
    last_infer   = 0.0

    try:
        while True:
            # ── Citire cadre ─────────────────
            while True:
                try:
                    frame = q.get_nowait()
                except queue.Empty:
                    break
                if frame.size != data_length:
                    continue
                buf.append(frame.copy())

            if reader.err:
                sys.stdout.write(f"\n[READER ERROR] {reader.err}\n")
                sys.stdout.flush()
                break

            now = time.time()
            if len(buf) < T or (now - last_infer) < infer_period:
                time.sleep(0.003)
                continue
            last_infer = now

            # ──────────────────────────────────
            # PHASE 1 – NOISE GATE
            # ──────────────────────────────────
            X_raw = np.stack(list(buf), axis=0)  # (T, data_length)
            raw_range = float(X_raw.max() - X_raw.min())
            gate_open = raw_range >= NOISE_GATE

            if not gate_open:
                # Reseteaza FSM, nu emite gest
                ema_probs[:] = 1.0 / len(labels)
                fsm_label = "none"
                fsm_count = 0
                if cooldown > 0:
                    cooldown -= 1
                print_status(False, "none", 0.0, "none", 0, 0, cooldown)
                time.sleep(0.003)
                continue

            # ──────────────────────────────────
            # PHASE 2 – INFERENTA + EMA SMOOTHING
            # ──────────────────────────────────
            Xp     = preprocess(X_raw, T=T, R=R, mean=mean, std=std)
            xt     = torch.from_numpy(Xp[None, None, :, :])

            with torch.no_grad():
                logits = model(xt).numpy().reshape(-1)

            raw_probs  = softmax(logits)
            # EMA smoothing pe probabilitati
            ema_probs  = EMA_ALPHA * raw_probs + (1.0 - EMA_ALPHA) * ema_probs

            top_idx    = int(np.argmax(ema_probs))
            top_prob   = float(ema_probs[top_idx])
            top_label  = labels[top_idx]

            # Daca suntem in cooldown – nu procesam gestul
            if cooldown > 0:
                cooldown -= 1
                print_status(True, top_label, top_prob, top_label, fsm_count,
                             DEBOUNCE.get(top_label, DEBOUNCE["default"]), cooldown)
                time.sleep(0.003)
                continue

            # ──────────────────────────────────
            # PHASE 2b – DEBOUNCE DINAMIC
            # ──────────────────────────────────
            # Ignora daca confidenta e prea mica sau e 'none'
            if top_prob < MIN_CONF or top_label == "none":
                fsm_label = "none"
                fsm_count = 0
                print_status(True, top_label, top_prob, "none", 0, 0, 0)
                time.sleep(0.003)
                continue

            needed = DEBOUNCE.get(top_label, DEBOUNCE["default"])

            if top_label == fsm_label:
                fsm_count += 1
            else:
                # Clasa s-a schimbat – reset counter
                fsm_label = top_label
                fsm_count  = 1

            print_status(True, top_label, top_prob, fsm_label, fsm_count, needed, 0)

            # ──────────────────────────────────
            # PHASE 3 – TRIGGER + COOLDOWN
            # ──────────────────────────────────
            if fsm_count >= needed:
                print_gesture(fsm_label)

                # Hold nu intra in cooldown ca sa poata fi tinut continuu
                if fsm_label != "hold":
                    cooldown  = COOLDOWN_MAX

                # Reseteaza FSM
                fsm_label = "none"
                fsm_count  = 0
                ema_probs[:]= 1.0 / len(labels)  # flush EMA dupa gest

            time.sleep(0.003)

    except KeyboardInterrupt:
        sys.stdout.write("\n[STOP] Deconectare...\n")
        sys.stdout.flush()
    finally:
        stop_evt.set()
        try:
            client.stop_session()
        except Exception:
            pass
        try:
            client.disconnect()
        except Exception:
            pass
        print("[INFO] Gata. La revedere!")


if __name__ == "__main__":
    main()
