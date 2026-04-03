"""
gesture_engine.py  —  All ML / DSP lives here.
=================================================
Direct port of the proven test_model.py pipeline:
  • TinyCNN identical architecture
  • fix_T + resample_range + log1p + z-score preprocessing
  • Noise gate (range-based energy)
  • EMA probability smoothing  (alpha=0.6)
  • Fast-track override for Push/Pull impulses
  • Per-gesture debounce + cooldown FSM
  • Warmup mask on hand entry

Loaded from:  out_model/gesture_cnn_boss.pt
Meta:         out_model/gesture_cnn_meta.json  (labels, T, R, mean, std)

Public API
──────────
  engine = GestureEngine()
  engine.predict(frame_buffer)  →  (gesture, confidence, probs_dict, is_event)
  engine.reset()
  engine.labels        → list[str]
  engine.fsm_state     → dict  (for SocketIO / /status)
"""

import json
import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────
# Paths  (relative to this file's directory)
# ─────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_HERE, "out_model", "gesture_cnn_boss.pt")
_META_PATH  = os.path.join(_HERE, "out_model", "gesture_cnn_meta.json")

# ─────────────────────────────────────────────────────────────
# FSM parameters  (identical to test_model.py)
# ─────────────────────────────────────────────────────────────
NOISE_GATE   = 0.15          # min-max range threshold

EMA_ALPHA    = 0.6           # higher → more reactive to impulse gestures

MIN_CONF = {
    "push":    0.40,
    "pull":    0.40,
    "tap":     0.70,
    "wave":    0.40,
    "hold":    0.60,
    "default": 0.50,
}

DEBOUNCE = {
    "hold":    5,
    "tap":     3,
    "wave":    2,
    "push":    1,
    "pull":    1,
    "default": 2,
}

COOLDOWN_MAX = 15            # frames after tap / wave
COOLDOWN_BURST = 4           # frames after push / pull  (burst mode)
WARMUP_MAX   = 10            # frames to ignore on hand entry (except hold)

# Fast-track threshold for push / pull  (overrides EMA inertia)
FAST_TRACK_THRESH = 0.65


# ══════════════════════════════════════════════════════════════
# MODEL DEFINITION  (must match training code exactly)
# ══════════════════════════════════════════════════════════════

class TinyCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
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
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.h(self.f(x))


# ══════════════════════════════════════════════════════════════
# PREPROCESSING  (identical to test_model.py)
# ══════════════════════════════════════════════════════════════

def _fix_T(X: np.ndarray, target_T: int) -> np.ndarray:
    if X.shape[0] == target_T:
        return X
    if X.shape[0] > target_T:
        return X[:target_T]
    pad = np.repeat(X[-1:], target_T - X.shape[0], axis=0)
    return np.concatenate([X, pad], axis=0)


def _resample_range(X: np.ndarray, out_bins: int) -> np.ndarray:
    T_curr, R_curr = X.shape
    x_old = np.linspace(0.0, 1.0, R_curr, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, out_bins,  dtype=np.float32)
    Y = np.empty((T_curr, out_bins), dtype=np.float32)
    for t in range(T_curr):
        Y[t] = np.interp(x_new, x_old, X[t].astype(np.float32))
    return Y


def _preprocess(X: np.ndarray, T: int, R: int,
                mean: float, std: float) -> np.ndarray:
    X = _fix_T(X, T)
    X = _resample_range(X, R)
    X = np.log1p(np.maximum(X, 0.0).astype(np.float32))
    X = (X - mean) / std
    return X.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


# ══════════════════════════════════════════════════════════════
# GESTURE ENGINE
# ══════════════════════════════════════════════════════════════

class GestureEngine:
    """
    Drop-in replacement for the old GestureEngine.

    serverV2.py calls:
        gesture, confidence, probs, is_event = engine.predict(frame_buffer)
        engine.reset()
        engine.labels
        engine.fsm_state
    """

    def __init__(self) -> None:
        # ── Load model checkpoint ────────────────────────────────────────
        pack = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
        self.labels: list[str] = pack["labels"]
        self._T: int   = pack["T"]
        self._R: int   = pack["R"]
        self._mean: float = float(pack["mean"])
        self._std:  float = float(pack["std"])

        self._model = TinyCNN(n_classes=len(self.labels))
        self._model.load_state_dict(pack["state_dict"])
        self._model.eval()

        print(f"[GestureEngine] Model loaded  T={self._T} R={self._R} "
              f"labels={self.labels}")

        # ── FSM state ────────────────────────────────────────────────────
        self._ema_probs:    np.ndarray = np.ones(len(self.labels),
                                                 dtype=np.float32) / len(self.labels)
        self._fsm_label:    str  = "none"
        self._fsm_count:    int  = 0
        self._cooldown:     int  = 0
        self._gate_was_open: bool = False
        self._warmup_frames: int  = 0

        # ── Display persistence (shows last gesture for N frames) ────────
        self._display_gesture:    str   = "none"
        self._display_confidence: float = 0.0
        self._display_ttl:        int   = 0
        self._DISPLAY_TTL         = 30   # frames to keep last gesture visible

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset EMA and FSM (called by serverV2 on stop/rebase)."""
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
        """Live FSM diagnostics — exposed to /status and SocketIO payloads."""
        top_idx = int(np.argmax(self._ema_probs))
        return {
            "label":     self._fsm_label,
            "count":     self._fsm_count,
            "cooldown":  self._cooldown,
            "warmup":    self._warmup_frames,
            "gate_open": self._gate_was_open,
            "raw_top":   self.labels[top_idx],
            "raw_conf":  float(self._ema_probs[top_idx]),
        }

    def predict(
        self,
        frame_buffer,  # deque of np.ndarray (N_bins,)
    ) -> tuple[str, float, dict, bool]:
        """
        Run one inference cycle over the current sliding window.

        Returns
        -------
        gesture     : str   — FSM-confirmed gesture (or display-persisted)
        confidence  : float — confidence of that gesture
        probs       : dict  — {label: probability} for all classes (EMA-smoothed)
        is_event    : bool  — True ONLY on the frame the gesture fires (UDP pulse)
        """
        X_raw = np.stack(list(frame_buffer), axis=0)  # (T, N_bins)

        # ── Phase 1: Noise Gate ───────────────────────────────────────────
        gate_open = float(X_raw.max() - X_raw.min()) >= NOISE_GATE

        if not gate_open:
            self._ema_probs[:] = 1.0 / len(self.labels)
            self._fsm_label    = "none"
            self._fsm_count    = 0
            self._gate_was_open = False
            if self._cooldown > 0:
                self._cooldown -= 1
            # Decay display
            if self._display_ttl > 0:
                self._display_ttl -= 1
            else:
                self._display_gesture    = "none"
                self._display_confidence = 0.0

            probs = self._probs_dict()
            return self._display_gesture, self._display_confidence, probs, False

        # Warmup on hand entry
        if not self._gate_was_open:
            self._gate_was_open  = True
            self._warmup_frames  = WARMUP_MAX

        # ── Phase 2: Inference ────────────────────────────────────────────
        Xp = _preprocess(X_raw, self._T, self._R, self._mean, self._std)
        xt = torch.from_numpy(Xp[None, None, :, :])  # (1,1,T,R)

        with torch.no_grad():
            logits = self._model(xt).numpy().reshape(-1)

        raw_probs = _softmax(logits)

        # EMA smoothing
        self._ema_probs = (
            EMA_ALPHA * raw_probs + (1.0 - EMA_ALPHA) * self._ema_probs
        )

        # ── Fast-track for impulse gestures ──────────────────────────────
        raw_top_idx   = int(np.argmax(raw_probs))
        raw_top_label = self.labels[raw_top_idx]
        raw_top_prob  = float(raw_probs[raw_top_idx])

        if raw_top_label in ("push", "pull") and raw_top_prob > FAST_TRACK_THRESH:
            top_label = raw_top_label
            top_prob  = raw_top_prob
            # Reset EMA so EMA inertia doesn't pull back towards hold
            self._ema_probs[:] = 1.0 / len(self.labels)
            self._ema_probs[raw_top_idx] = raw_top_prob
        else:
            top_idx   = int(np.argmax(self._ema_probs))
            top_prob  = float(self._ema_probs[top_idx])
            top_label = self.labels[top_idx]

        probs = self._probs_dict()

        # ── Cooldown ──────────────────────────────────────────────────────
        if self._cooldown > 0:
            self._cooldown -= 1
            # Decay display TTL
            if self._display_ttl > 0:
                self._display_ttl -= 1
            return (
                self._display_gesture,
                self._display_confidence,
                probs,
                False,
            )

        # ── Warmup ────────────────────────────────────────────────────────
        if self._warmup_frames > 0:
            self._warmup_frames -= 1
            if top_label != "hold":
                return self._display_gesture, self._display_confidence, probs, False

        # ── Phase 3: Confidence gate ──────────────────────────────────────
        req_conf = MIN_CONF.get(top_label, MIN_CONF["default"])

        if top_prob < req_conf or top_label == "none":
            self._fsm_label = "none"
            self._fsm_count = 0
            # Decay display
            if self._display_ttl > 0:
                self._display_ttl -= 1
            else:
                self._display_gesture    = "none"
                self._display_confidence = 0.0
            return self._display_gesture, self._display_confidence, probs, False

        # ── Phase 4: Debounce FSM ─────────────────────────────────────────
        needed = DEBOUNCE.get(top_label, DEBOUNCE["default"])

        if top_label == self._fsm_label:
            self._fsm_count += 1
        else:
            self._fsm_label = top_label
            self._fsm_count = 1

        # ── Phase 5: Fire gesture ─────────────────────────────────────────
        if self._fsm_count >= needed:
            fired_gesture    = self._fsm_label
            fired_confidence = top_prob

            # Cooldown per gesture type
            if fired_gesture in ("tap", "wave"):
                self._cooldown = COOLDOWN_MAX
            elif fired_gesture in ("push", "pull"):
                self._cooldown = COOLDOWN_BURST
            else:  # hold
                self._cooldown = 0

            # Reset FSM
            self._fsm_label = "none"
            self._fsm_count = 0
            self._ema_probs[:] = 1.0 / len(self.labels)

            # Persist display
            self._display_gesture    = fired_gesture
            self._display_confidence = fired_confidence
            self._display_ttl        = self._DISPLAY_TTL

            return fired_gesture, fired_confidence, probs, True  # is_event=True

        # Still counting — return persisted display
        if self._display_ttl > 0:
            self._display_ttl -= 1
        return self._display_gesture, self._display_confidence, probs, False

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _probs_dict(self) -> dict[str, float]:
        return {
            label: round(float(self._ema_probs[i]), 4)
            for i, label in enumerate(self.labels)
        }
