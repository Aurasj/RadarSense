"""
Gesture inference engine used by RadarSense.

Loads the trained TinyCNN model, preprocesses radar frames and
returns stabilized gesture predictions using the FSM pipeline.

Model:
    cfg/gesture_cnn_boss.pt
"""

import os

import numpy as np
import torch
import torch.nn as nn

# ── Model path ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

_MODEL_PATH = os.path.join(_ROOT, "cfg", "gesture_cnn_boss.pt")

# ── FSM parameters ────────────────────────────────────────────────────────────
# These were tuned for the hardware. Don't change them unless you retrain.

NOISE_GATE = 0.15   # min-max envelope range below which we assume no hand is present

EMA_ALPHA = 0.70    # smoothing factor for probability averaging (higher = more reactive)

# Minimum confidence required before a gesture is even considered.
MIN_CONF = {
    "push":    0.60,
    "pull":    0.60,
    "tap":     0.45,
    "wave":    0.55,   # stricter threshold so wave doesn't steal tap or hold
    "hold":    0.45,   # hand hold reflects less than a phone, so lower threshold
    "default": 0.50,
}

# How many consecutive frames must agree before the gesture fires.
DEBOUNCE = {
    "hold":    5,
    "tap":     2,
    "wave":    2,
    "push":    1,
    "pull":    1,
    "default": 2,
}

COOLDOWN_MAX   = 15   # frames to wait after tap or wave
COOLDOWN_BURST = 4    # frames to wait after push or pull
COOLDOWN_HOLD  = 25   # frames to wait after hold (avoids spamming UDP)
WARMUP_MAX     = 10   # frames to skip after the hand enters the range (except hold)

# If the raw CNN is this confident on a single frame, skip EMA and fire immediately.
FAST_TRACK_THRESH      = 0.65
WAVE_FAST_TRACK_THRESH = 0.70
TAP_FAST_TRACK_THRESH  = 0.58


# ── Model definition ──────────────────────────────────────────────────────────
# Architecture must match train_V3.py exactly, otherwise loading the weights will fail.

class TinyCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        # Feature extractor: 3 conv blocks with batch norm and pooling.
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
        # Classifier head with dropout.
        self.h = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(self.f(x))


# ── Preprocessing ─────────────────────────────────────────────────────────────
# Pipeline must match train_V3.py exactly: fix_T -> resample -> log1p -> z-score.

def _fix_T(X: np.ndarray, target_T: int) -> np.ndarray:
    """Crop or pad the frame sequence to exactly target_T frames."""
    if X.shape[0] == target_T:
        return X
    if X.shape[0] > target_T:
        return X[:target_T]
    # Pad by repeating the last frame.
    pad = np.repeat(X[-1:], target_T - X.shape[0], axis=0)
    return np.concatenate([X, pad], axis=0)


def _resample_range(X: np.ndarray, out_bins: int) -> np.ndarray:
    """Resample the range dimension to out_bins using linear interpolation."""
    T_curr, R_curr = X.shape
    x_old = np.linspace(0.0, 1.0, R_curr, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, out_bins, dtype=np.float32)
    Y = np.empty((T_curr, out_bins), dtype=np.float32)
    for t in range(T_curr):
        Y[t] = np.interp(x_new, x_old, X[t].astype(np.float32))
    return Y


def _preprocess(X: np.ndarray, T: int, R: int,
                mean: float, std: float) -> np.ndarray:
    """Apply the full preprocessing pipeline to a raw frame window."""
    X = _fix_T(X, T)
    X = _resample_range(X, R)
    X = np.log1p(np.maximum(X, 0.0).astype(np.float32))
    X = (X - mean) / std
    return X.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


# ── Gesture Engine ────────────────────────────────────────────────────────────

class GestureEngine:
    """
    Loads the TinyCNN checkpoint and runs gesture inference on each frame window.
    Handles noise gate, EMA smoothing, fast-track, debounce FSM, and cooldown.
    Sets is_event=True on exactly one frame per confirmed gesture.
    """

    def __init__(self) -> None:
        # Load the model checkpoint saved by train_V3.py.
        pack = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
        self.labels: list[str] = pack["labels"]
        self._T: int      = pack["T"]
        self._R: int      = pack["R"]
        self._mean: float = float(pack["mean"])
        self._std:  float = float(pack["std"])

        self._model = TinyCNN(n_classes=len(self.labels))
        self._model.load_state_dict(pack["state_dict"])
        self._model.eval()

        print(f"[GestureEngine] Model loaded  T={self._T} R={self._R} "
              f"labels={self.labels}")

        # FSM state variables.
        self._ema_probs:     np.ndarray = np.ones(len(self.labels),
                                                   dtype=np.float32) / len(self.labels)
        self._fsm_label:     str  = "none"
        self._fsm_count:     int  = 0
        self._cooldown:      int  = 0
        self._gate_was_open: bool = False
        self._warmup_frames: int  = 0

        # Keep the last confirmed gesture visible for a few frames after it fires.
        self._display_gesture:    str   = "none"
        self._display_confidence: float = 0.0
        self._display_ttl:        int   = 0
        self._DISPLAY_TTL               = 30   # number of frames to keep the label visible

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all FSM state and EMA. Called by server on stop or recalibrate."""
        self._ema_probs[:] = 1.0 / len(self.labels)
        self._fsm_label    = "none"
        self._fsm_count    = 0
        self._cooldown     = 0
        self._gate_was_open = False
        self._warmup_frames = 0
        self._display_gesture    = "none"
        self._display_confidence = 0.0
        self._display_ttl        = 0

    @property
    def fsm_state(self) -> dict:
        """Current FSM diagnostics, sent to the browser and /status endpoint."""
        top_idx = int(np.argmax(self._ema_probs))
        return {
            "label":     self._fsm_label,
            "count":     self._fsm_count,
            "cooldown":  self._cooldown,
            "warmup":    self._warmup_frames,
            "gate_open": self._gate_was_open,
            "raw_top":   self.labels[top_idx],
            "raw_conf":  float(self._ema_probs[top_idx]),
            "display":   self._display_gesture,
        }

    def predict(
        self,
        frame_buffer,  # deque of np.ndarray (N_bins,)
    ) -> tuple[str, float, dict, bool]:
        """
        Run one inference cycle over the current sliding window.

        Returns
        -------
        gesture     : str   - FSM-confirmed gesture (or the last one if still in display TTL)
        confidence  : float - confidence score for that gesture
        probs       : dict  - {label: probability} for all classes (EMA-smoothed)
        is_event    : bool  - True only on the single frame when the gesture fires
        """
        X_raw = np.stack(list(frame_buffer), axis=0)  # (T, N_bins)

        # Phase 1: Noise gate.
        # If the envelope variation is too small, there's probably no hand present.
        gate_open = float(X_raw.max() - X_raw.min()) >= NOISE_GATE

        if not gate_open:
            self._ema_probs[:] = 1.0 / len(self.labels)
            self._fsm_label    = "none"
            self._fsm_count    = 0
            self._gate_was_open = False
            if self._cooldown > 0:
                self._cooldown -= 1
            # Count down the display TTL so the label fades naturally.
            if self._display_ttl > 0:
                self._display_ttl -= 1
            else:
                self._display_gesture    = "none"
                self._display_confidence = 0.0

            probs = self._probs_dict()
            return self._display_gesture, self._display_confidence, probs, False

        # Start warmup when the hand first enters the detection zone.
        if not self._gate_was_open:
            self._gate_was_open  = True
            self._warmup_frames  = WARMUP_MAX

        # Phase 2: Run the CNN on the preprocessed window.
        Xp = _preprocess(X_raw, self._T, self._R, self._mean, self._std)
        xt = torch.from_numpy(Xp[None, None, :, :])  # shape: (1, 1, T, R)

        with torch.inference_mode():
            logits = self._model(xt).numpy().reshape(-1)

        raw_probs = _softmax(logits)

        # Blend new raw probabilities with the running EMA.
        self._ema_probs = (
            EMA_ALPHA * raw_probs + (1.0 - EMA_ALPHA) * self._ema_probs
        )

        # Fast-track: if the CNN is very confident on a single frame,
        # bypass the EMA and use the raw result directly.
        raw_top_idx   = int(np.argmax(raw_probs))
        raw_top_label = self.labels[raw_top_idx]
        raw_top_prob  = float(raw_probs[raw_top_idx])

        if raw_top_label in ("push", "pull") and raw_top_prob > FAST_TRACK_THRESH:
            top_label = raw_top_label
            top_prob  = raw_top_prob
            self._set_fasttrack_probs(raw_top_idx, raw_top_prob)

        elif raw_top_label == "tap" and raw_top_prob > TAP_FAST_TRACK_THRESH:
            top_label = "tap"
            top_prob  = raw_top_prob
            self._set_fasttrack_probs(raw_top_idx, raw_top_prob)

        elif raw_top_label == "wave" and raw_top_prob > WAVE_FAST_TRACK_THRESH:
            top_label = "wave"
            top_prob  = raw_top_prob
            self._set_fasttrack_probs(raw_top_idx, raw_top_prob)

        else:
            top_idx   = int(np.argmax(self._ema_probs))
            top_prob  = float(self._ema_probs[top_idx])
            top_label = self.labels[top_idx]

        probs = self._probs_dict()

        # Cooldown: block new gestures for a while after one fires.
        if self._cooldown > 0:
            self._cooldown -= 1
            if self._display_ttl > 0:
                self._display_ttl -= 1
            return (
                self._display_gesture,
                self._display_confidence,
                probs,
                False,
            )

        # Warmup: ignore new gestures for a few frames after the hand enters.
        # Hold is the exception — it can fire immediately.
        if self._warmup_frames > 0:
            self._warmup_frames -= 1
            if top_label != "hold":
                return self._display_gesture, self._display_confidence, probs, False

        # Phase 3: Check minimum confidence threshold.
        req_conf = MIN_CONF.get(top_label, MIN_CONF["default"])

        if top_prob < req_conf or top_label == "none":
            self._fsm_label = "none"
            self._fsm_count = 0
            if self._display_ttl > 0:
                self._display_ttl -= 1
            else:
                self._display_gesture    = "none"
                self._display_confidence = 0.0
            return self._display_gesture, self._display_confidence, probs, False

        # Phase 4: Debounce FSM.
        # Count consecutive frames that agree on the same gesture.
        needed = DEBOUNCE.get(top_label, DEBOUNCE["default"])

        if top_label == self._fsm_label:
            self._fsm_count += 1
        else:
            self._fsm_label = top_label
            self._fsm_count = 1

        # Phase 5: Fire the gesture if the debounce count is reached.
        if self._fsm_count >= needed:
            fired_gesture    = self._fsm_label
            fired_confidence = top_prob

            # Set cooldown based on gesture type.
            if fired_gesture in ("tap", "wave"):
                self._cooldown = COOLDOWN_MAX
            elif fired_gesture in ("push", "pull"):
                self._cooldown = COOLDOWN_BURST
            else:  # hold
                self._cooldown = COOLDOWN_HOLD

            # Reset the FSM for the next gesture.
            self._fsm_label = "none"
            self._fsm_count = 0
            self._ema_probs[:] = 1.0 / len(self.labels)

            # Keep the gesture visible on the dashboard for a bit.
            self._display_gesture    = fired_gesture
            self._display_confidence = fired_confidence
            self._display_ttl        = self._DISPLAY_TTL

            return fired_gesture, fired_confidence, probs, True  # is_event=True

        # Still counting frames — show the last confirmed gesture for now.
        if self._display_ttl > 0:
            self._display_ttl -= 1
        return self._display_gesture, self._display_confidence, probs, False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _set_fasttrack_probs(self, top_idx: int, top_prob: float) -> None:
        """
        Override EMA with a fast-track distribution.
        Winning class gets top_prob; the rest share the remainder equally.
        """
        n = len(self.labels)
        top_prob = float(np.clip(top_prob, 0.0, 1.0))
        rest = (1.0 - top_prob) / max(n - 1, 1)
        self._ema_probs[:] = rest
        self._ema_probs[top_idx] = top_prob

    def _probs_dict(self) -> dict[str, float]:
        """Return EMA probabilities as a {label: value} dict for the browser."""
        return {
            label: round(float(self._ema_probs[i]), 4)
            for i, label in enumerate(self.labels)
        }
