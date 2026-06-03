import argparse, time, json, queue, threading
from collections import deque
import numpy as np
import acconeer.exptool as et
import torch
import torch.nn as nn

# --- Functii Preprocesare (Ramase neschimbate) ---
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

def preprocess(X: np.ndarray, T: int, out_bins: int, mean: float, std: float) -> np.ndarray:
    X = fix_T(X, T=T)
    X = resample_range(X, out_bins=out_bins)
    X = np.log1p(np.maximum(X, 0.0).astype(np.float32))
    X = (X - mean) / (std + 1e-6)
    return X.astype(np.float32)

# --- Reader pentru datele de la senzor (Threading) ---
class Reader(threading.Thread):
    def __init__(self, client, q, stop_evt):
        super().__init__(daemon=True)
        self.client = client
        self.q = q
        self.stop_evt = stop_evt
        self.err = None

    def run(self):
        try:
            while not self.stop_evt.is_set():
                _i, data = self.client.get_next()
                frame = np.asarray(data).squeeze().astype(np.float32).reshape(-1)
                try:
                    self.q.put_nowait(frame)
                except queue.Full:
                    try: self.q.get_nowait()
                    except queue.Empty: pass
                    self.q.put(frame)
        except Exception as e:
            self.err = f"{type(e).__name__}: {e}"

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

# --- Definirea Clasei TinyCNN (Adaptata pentru modelul tau) ---
class TinyCNN(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        # Schimbat in "f" si "h" conform state_dict-ului din gesture_cnn.pt
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,5), padding=(1,2)),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=(3,5), padding=(1,2)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.h = nn.Sequential(nn.Flatten(), nn.Dropout(0.0), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.h(self.f(x))

# --- Main Pipeline ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="gesture_cnn.pt")
    ap.add_argument("--update_rate", type=float, default=30.0)
    ap.add_argument("--range_start", type=float, default=0.20)
    ap.add_argument("--range_end", type=float, default=0.55)
    ap.add_argument("--infer_hz", type=float, default=10.0)
    ap.add_argument("--smooth", type=int, default=7)
    ap.add_argument("--min_prob", type=float, default=0.55)
    args = ap.parse_args()

    # Incarcare pack si model
    print(f"[INFO] Incarcare model: {args.model}")
    pack = torch.load(args.model, map_location="cpu")
    labels = pack["labels"]
    T = int(pack["T"])
    R = int(pack["R"])
    mean = float(pack["mean"])
    std = float(pack["std"])

    model = TinyCNN(n_classes=len(labels))
    model.load_state_dict(pack["state_dict"])
    model.eval()

    # Configurare Radar
    cfg = et.a111.EnvelopeServiceConfig()
    cfg.update_rate = float(args.update_rate)
    cfg.range_interval = [float(args.range_start), float(args.range_end)]

    client = et.a111.Client(protocol="exploration", link="socket", host=args.host)
    client.connect()
    session_info = client.setup_session(cfg)
    client.start_session()
    data_length = int(session_info["data_length"])

    q = queue.Queue(maxsize=400)
    stop_evt = threading.Event()
    rd = Reader(client, q, stop_evt)
    rd.start()

    buf = deque(maxlen=T)
    pred_hist = deque(maxlen=int(args.smooth))
    last_infer = 0.0
    infer_period = 1.0 / max(1e-6, float(args.infer_hz))

    print("\n[LIVE] Sistem pornit! q=Ctrl+C pentru stop.")
    
    try:
        while True:
            # Preluare cadre din coada
            while True:
                try:
                    f = q.get_nowait()
                except queue.Empty:
                    break
                if f.size != data_length:
                    continue
                buf.append(f.copy())

            if rd.err:
                print("\nREADER ERROR:", rd.err)
                break

            now = time.time()
            if len(buf) >= T and (now - last_infer) >= infer_period:
                last_infer = now

                X = np.stack(list(buf), axis=0)
                Xp = preprocess(X, T=T, out_bins=R, mean=mean, std=std)
                xt = torch.from_numpy(Xp[None, None, :, :])

                with torch.no_grad():
                    logits = model(xt).numpy().reshape(-1)
                
                probs = softmax(logits)
                pmax = float(np.max(probs))
                pred = int(np.argmax(probs))

                if pmax < float(args.min_prob):
                    pred = labels.index("none")

                pred_hist.append(pred)
                vals, cnts = np.unique(np.array(pred_hist), return_counts=True)
                maj = int(vals[np.argmax(cnts)])

                # Afisare rezultate
                s = f"\rPRED={labels[maj]:>5s} | Prob: {pmax:0.2f} | Raw: {labels[int(np.argmax(probs))]:>5s}  "
                print(s, end="", flush=True)

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[STOP]")
    finally:
        stop_evt.set()
        try: client.stop_session()
        except: pass
        try: client.disconnect()
        except: pass

if __name__ == "__main__":
    main()