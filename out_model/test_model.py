import sys
import time
import queue
import threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import acconeer.exptool as et

# --- CONFIGURATIE ---
HOST         = "192.168.0.145"
PORT         = 6110
MODEL_PATH   = "gesture_cnn_boss.pt"

UPDATE_RATE  = 30.0   
RANGE_START  = 0.20   
RANGE_END    = 0.60   
INFER_HZ     = 15.0   

NOISE_GATE   = 0.15   
EMA_ALPHA    = 0.6    # Sensibilitate mai mare la miscari bruste (push/pull/wave)
#MIN_CONF     = 0.50   # Incredere putin mai mica pt a prinde varful gesturilor rapide
MIN_CONF     = {
    "push": 0.40,     # Foarte sensibil: prinde impulsul instant
    "pull": 0.40,     # Foarte sensibil
    "tap":  0.70,     # Foarte strict: trebuie sa fie un Tap clar ca sa nu se confunde
    "wave": 0.40,
    "hold": 0.60,
    "default": 0.50
}

# DEBOUNCE - REGLAT PENTRU FIECARE GEST IN PARTE
DEBOUNCE     = {
    "hold": 5,    # Gest continuu - asteptam 5 cadre ca sa fim siguri
    "tap":  3,    # Gest rapid dar scurt
    "wave": 2,    # Miscare stanga-dreapta
    "push": 1,    # Gest IMPULS - tragem concluzia instant
    "pull": 1,    # Gest IMPULS - tragem concluzia instant
    "default": 2,
}
COOLDOWN_MAX = 15     # Cate cadre stam pe pauza dupa gest

# --- DEFINITIE MODEL ---
class TinyCNN(nn.Module):
    def __init__(self, n_classes):
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
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.h(self.f(x))

# --- PREPROCESARE ---
def fix_T(X, target_T):
    if X.shape[0] == target_T: return X
    if X.shape[0] > target_T: return X[:target_T]
    pad = np.repeat(X[-1:], target_T - X.shape[0], axis=0)
    return np.concatenate([X, pad], axis=0)

def resample_range(X, out_bins):
    T_curr, R_curr = X.shape
    x_old = np.linspace(0.0, 1.0, R_curr, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, out_bins, dtype=np.float32)
    Y = np.empty((T_curr, out_bins), dtype=np.float32)
    for t in range(T_curr):
        Y[t] = np.interp(x_new, x_old, X[t].astype(np.float32))
    return Y

def preprocess(X, T, R, mean, std):
    X = fix_T(X, T)
    X = resample_range(X, R)
    X = np.log1p(np.maximum(X, 0.0).astype(np.float32))
    X = (X - mean) / std
    return X.astype(np.float32)

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

# --- THREAD PENTRU CITIRE SENZOR ---
class RadarReader(threading.Thread):
    def __init__(self, client, q, stop_evt):
        super().__init__(daemon=True)
        self.client = client
        self.q = q
        self.stop_evt = stop_evt
        self.err = None

    def run(self):
        try:
            while not self.stop_evt.is_set():
                _info, data = self.client.get_next()
                frame = np.asarray(data).squeeze().astype(np.float32).reshape(-1)
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    try: self.q.get_nowait()
                    except queue.Empty: pass
                    self.q.put(frame)
        except Exception as exc:
            self.err = str(exc)

# --- UI TERMINAL ---
ICONS = {"hold": "✋", "push": "👉", "pull": "👈", "tap": "👆", "wave": "👋", "none": "🫥"}

def print_status(gate_open, top_label, top_prob, fsm_label, count, needed, cooldown):
    # Stabilim ce afisam daca e "none" sau poarta e inchisa
    if not gate_open or top_label == "none":
        display_label = "none"
        display_prob = 1.0 if not gate_open else top_prob
        fsm_str = "\033[90m[ Stare: NICIUN GEST ]\033[0m          "
    else:
        display_label = top_label
        display_prob = top_prob
        if cooldown > 0:
            fsm_str = f"\033[93m[ Stare: Pauza {cooldown:2d}/{COOLDOWN_MAX} ]\033[0m      "
        else:
            fsm_str = f"\033[96m[ Stare: Verificare {count}/{needed} ]\033[0m"

    icon = ICONS.get(display_label, "?")
    width = 10
    filled = int(round(display_prob * width))
    bar = "█" * filled + "░" * (width - filled)
    
    # \033[K sterge continutul vechi de pe linie ca sa nu lase artefacte vizuale
    line = (
        f"\r\033[K"
        f"Radar: {'🟢' if gate_open else '🔴'} | "
        f"Detectie: {icon} {display_label.upper():<4} {display_prob*100:5.1f}% {bar} | "
        f"{fsm_str}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()

def print_gesture(label):
    icon = ICONS.get(label, "🎯")
    sys.stdout.write(f"\n\n\033[1;92m >>> GEST CONFIRMAT: {label.upper()} {icon} <<< \033[0m\n\n")
    sys.stdout.flush()

# --- MAIN LOOP ---
def main():
    print(f"🚀 Incarcam modelul BOSS: {MODEL_PATH}")
    pack = torch.load(MODEL_PATH, map_location="cpu")
    labels = pack["labels"]
    T_val, R_val = pack["T"], pack["R"]
    mean, std = pack["mean"], pack["std"]

    model = TinyCNN(n_classes=len(labels))
    model.load_state_dict(pack["state_dict"])
    model.eval() 
    print(f"✅ Model incarcat! Asteapta parametri: T={T_val}, R={R_val}")

    print(f"📡 Ne conectam la radar pe {HOST}:{PORT} ...")
    cfg = et.a111.EnvelopeServiceConfig()
    cfg.update_rate = UPDATE_RATE
    cfg.range_interval = [RANGE_START, RANGE_END]

    client = et.a111.Client(protocol="exploration", link="socket", host=HOST)
    client.connect()
    session_info = client.setup_session(cfg)
    client.start_session()
    data_length = int(session_info["data_length"])
    print("✅ Conectat! Incepem sa citim gesturi. (Apasa Ctrl+C ca sa opresti)\n")

    q = queue.Queue(maxsize=500)
    stop_evt = threading.Event()
    reader = RadarReader(client, q, stop_evt)
    reader.start()

    buf = deque(maxlen=T_val)
    ema_probs = np.ones(len(labels), dtype=np.float32) / len(labels)

    fsm_label = "none"
    fsm_count = 0
    cooldown = 0
    last_infer = 0.0
    infer_period = 1.0 / INFER_HZ

    # --- VARIABILE NOI PENTRU WARMUP ---
    gate_was_open = False
    warmup_frames = 0
    WARMUP_MAX = 10  # Asteapta 10 cadre (~0.3s) cand bagi mana, ca sa nu dea "Push" din greseala

    try:
        while True:
            while True:
                try: frame = q.get_nowait()
                except queue.Empty: break
                if frame.size == data_length:
                    buf.append(frame)

            if reader.err:
                print(f"\n[Eroare Senzor] {reader.err}")
                break

            now = time.time()
            if len(buf) < T_val or (now - last_infer) < infer_period:
                time.sleep(0.005)
                continue
            last_infer = now

            # FAZA 1: Noise Gate & Warmup
            X_raw = np.stack(list(buf), axis=0)
            gate_open = float(X_raw.max() - X_raw.min()) >= NOISE_GATE

            if not gate_open:
                ema_probs[:] = 1.0 / len(labels)
                fsm_label = "none"
                fsm_count = 0
                gate_was_open = False
                if cooldown > 0: cooldown -= 1
                print_status(False, "none", 0.0, "none", 0, 0, cooldown)
                continue

            # Daca poarta abia s-a deschis (ai bagat mana in cadru), activam Warmup
            if not gate_was_open:
                gate_was_open = True
                warmup_frames = WARMUP_MAX

            # FAZA 2: Inferenta model
            Xp = preprocess(X_raw, T_val, R_val, mean, std)
            xt = torch.from_numpy(Xp[None, None, :, :])

            with torch.no_grad():
                logits = model(xt).numpy().reshape(-1)

            raw_probs = softmax(logits)
            ema_probs = EMA_ALPHA * raw_probs + (1.0 - EMA_ALPHA) * ema_probs

            # --- SISTEMUL DE FAST-TRACK (PRIORITATE MAXIMA) ---
            raw_top_idx = int(np.argmax(raw_probs))
            raw_top_label = labels[raw_top_idx]
            raw_top_prob = float(raw_probs[raw_top_idx])

            # Daca vede clar un Push sau Pull, ii da override inerției de la Hold
            if raw_top_label in ["push", "pull"] and raw_top_prob > 0.65:
                top_label = raw_top_label
                top_prob = raw_top_prob
                # Rescriem EMA ca sa nu traga inapoi la Hold
                ema_probs[:] = 1.0 / len(labels)
                ema_probs[raw_top_idx] = raw_top_prob
            else:
                top_idx = int(np.argmax(ema_probs))
                top_prob = float(ema_probs[top_idx])
                top_label = labels[top_idx]

            # Blocare gesturi in timpul pauzei sau a warmup-ului
            if cooldown > 0:
                cooldown -= 1
                print_status(True, top_label, top_prob, top_label, 0, 0, cooldown)
                continue

            if warmup_frames > 0:
                warmup_frames -= 1
                sys.stdout.write(f"\r\033[K Radar: 🟢 | \033[93m[ Stare: Initializare mana... {warmup_frames} ]\033[0m")
                sys.stdout.flush()
                # Acceptam doar Hold in timpul warmup-ului, ca sa poti sa incepi rotatia dronei instant
                if top_label != "hold":
                    continue

            # FAZA 3: Confirmarea gestului
            req_conf = MIN_CONF.get(top_label, MIN_CONF["default"])
            
            if top_prob < req_conf or top_label == "none":
                fsm_label = "none"
                fsm_count = 0
                print_status(True, top_label, top_prob, "none", 0, 0, 0)
                continue

            needed = DEBOUNCE.get(top_label, DEBOUNCE["default"])

            if top_label == fsm_label:
                fsm_count += 1
            else:
                fsm_label = top_label
                fsm_count = 1

            print_status(True, top_label, top_prob, fsm_label, fsm_count, needed, 0)

            # FAZA 4: Declansarea vizuala a gestului
            if fsm_count >= needed:
                print_gesture(fsm_label)
                
                # --- LOGICA NOUA DE COOLDOWN PENTRU DRONA ---
                if fsm_label in ["tap", "wave"]:
                    # Gesturi unice (Meniu, miscare brusca fata). Pauza mare.
                    cooldown = COOLDOWN_MAX  
                elif fsm_label in ["push", "pull"]:
                    # Mod RAFALA pentru altitudine. Trimite comenzi de 3-4 ori pe secunda
                    # cat timp continui miscarea de push/pull.
                    cooldown = 4  
                else:
                    # HOLD: Nicio pauza. Roteste drona continuu, la fel de lin ca joystick-ul.
                    cooldown = 0  
                
                fsm_label = "none"
                fsm_count = 0
                
                # Resetam mediile pt a nu influenta cadrele urmatoare
                ema_probs[:] = 1.0 / len(labels)

    except KeyboardInterrupt:
        sys.stdout.write("\n\n🛑 Oprim treaba...\n")
        sys.stdout.flush()
    finally:
        stop_evt.set()
        try: client.stop_session()
        except: pass
        try: client.disconnect()
        except: pass
        print("✅ Gata. Iesire curata!")

if __name__ == "__main__":
    main()