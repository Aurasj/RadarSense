import os
import time
import queue
import threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import acconeer.exptool as et


# ================= CONFIG =================

HOST = "auraspi"
PORT = 6110

MODEL_PATH = "out_model/gesture_cnn_boss.pt"

RANGE_START = 0.20
RANGE_END = 0.55
UPDATE_RATE = 30

WINDOW_SIZE = 60

NOISE_GATE = 0.15
EMA_ALPHA = 0.70

MIN_CONF = {
    "push": 0.40,
    "pull": 0.40,
    "tap": 0.45,
    "wave": 0.40,
    "hold": 0.45,
    "default": 0.50,
}


# ================= MODEL =================

class TinyCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.h = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.h(self.f(x))


# ================= PREPROCESS =================

def fix_T(X, target_T):
    if X.shape[0] == target_T:
        return X
    if X.shape[0] > target_T:
        return X[:target_T]
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
    X = np.expand_dims(X, axis=(0, 1)).astype(np.float32)
    return torch.tensor(X, dtype=torch.float32)


# ================= MAIN =================

def main():
    print("\n=== LIVE MODEL TEST V3 ===\n")

    pack = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    labels = pack["labels"]
    T = pack["T"]
    R = pack["R"]
    mean = float(pack["mean"])
    std = float(pack["std"])

    model = TinyCNN(len(labels))
    model.load_state_dict(pack["state_dict"])
    model.eval()

    print(f"Model loaded: {MODEL_PATH}")
    print(f"labels={labels}")
    print(f"T={T}, R={R}, mean={mean:.4f}, std={std:.4f}")

    client = et.a111.Client(
        protocol="exploration",
        link="socket",
        host=HOST,
        port=PORT,
    )

    cfg = et.a111.EnvelopeServiceConfig()
    cfg.sensor = [1]
    cfg.range_interval = [RANGE_START, RANGE_END]
    cfg.update_rate = UPDATE_RATE

    print("\nConnecting to radar...")
    client.connect()
    client.setup_session(cfg)
    client.start_session()
    print("Connected. Perform gestures in front of the sensor. Ctrl+C to stop.\n")

    frame_buffer = deque(maxlen=WINDOW_SIZE)
    ema_probs = np.ones(len(labels), dtype=np.float32) / len(labels)

    try:
        while True:
            _, data = client.get_next()

            frame = data[0] if data.ndim > 1 else data
            frame = np.asarray(frame).ravel().astype(np.float32)

            frame_buffer.append(frame)

            if len(frame_buffer) < WINDOW_SIZE:
                print(f"\rWarmup frames: {len(frame_buffer)}/{WINDOW_SIZE}", end="")
                continue

            X = np.stack(frame_buffer, axis=0)

            raw_range = float(np.max(X) - np.min(X))
            gate_open = raw_range >= NOISE_GATE

            if not gate_open:
                gesture = "none"
                confidence = 1.0
            else:
                inp = preprocess(X, T, R, mean, std)

                with torch.no_grad():
                    logits = model(inp)
                    probs = torch.softmax(logits, dim=1).numpy()[0]

                raw_idx = int(np.argmax(probs))
                raw_gesture = labels[raw_idx]
                raw_conf = float(probs[raw_idx])

                if raw_gesture == "tap" and raw_conf >= 0.55:
                    gesture = "tap"
                    confidence = raw_conf
                else:
                    ema_probs = EMA_ALPHA * probs + (1.0 - EMA_ALPHA) * ema_probs

                    idx = int(np.argmax(ema_probs))
                    gesture = labels[idx]
                    confidence = float(ema_probs[idx])

                    min_conf = MIN_CONF.get(gesture, MIN_CONF["default"])

                    if confidence < min_conf:
                        gesture = "none"

            top = sorted(
                [(labels[i], float(ema_probs[i])) for i in range(len(labels))],
                key=lambda x: x[1],
                reverse=True,
            )[:3]

            top_str = " | ".join([f"{g}:{p*100:5.1f}%" for g, p in top])

            print(
                f"\rGesture: {gesture.upper():<6} "
                f"conf={confidence*100:5.1f}% | {top_str}          ",
                end="",
            )

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        print("\n\nStopping radar session...")

        try:
            client.stop_session()
            time.sleep(1.0)
        except Exception as e:
            print(f"stop_session warning: {repr(e)}")

        try:
            client.disconnect()
            time.sleep(1.0)
        except Exception as e:
            print(f"disconnect warning: {repr(e)}")

        print("Done.")


if __name__ == "__main__":
    main()