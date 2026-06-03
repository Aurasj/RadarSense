import os
import glob
import json
import numpy as np

DATA_DIR = "data"
CLASSES = ["none", "hold", "push", "pull", "tap", "wave"]

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    if "X" in d:
        return d["X"]
    return d[list(d.keys())[0]]

def metrics(X):
    X = np.asarray(X, dtype=np.float32)
    peak_per_t = X.max(axis=1)
    peak_bins = X.argmax(axis=1)

    return {
        "shape": X.shape,
        "max": float(X.max()),
        "mean": float(X.mean()),
        "std": float(X.std()),
        "energy": float(np.mean(np.abs(X))),
        "temporal_motion": float(np.std(peak_bins)),
        "amp_motion": float(np.std(peak_per_t)),
        "range_span": int(peak_bins.max() - peak_bins.min()),
    }

def flag(label, m):
    reasons = []

    if m["shape"][0] < 20:
        reasons.append("too_short")
    if m["max"] < 20:
        reasons.append("weak_signal")
    if m["max"] > 5000:
        reasons.append("possible_saturation")
    if label == "hold" and m["range_span"] > 35:
        reasons.append("hold_too_much_movement")
    if label == "tap" and m["shape"][0] >= 45 and m["range_span"] > 250:
    	reasons.append("old_tap_too_large_like_push_pull")
    if label == "none" and m["range_span"] > 80 and m["amp_motion"] > 80:
        reasons.append("none_suspicious_gesture_like")
    if label in ["push", "pull"] and m["range_span"] < 8:
        reasons.append("gesture_too_small")
    if label == "wave" and m["amp_motion"] < 5 and m["range_span"] < 5:
        reasons.append("wave_too_static_like_hold")

    return reasons

def main():
    print("\n=== DATASET AUDIT ===\n")

    all_suspicious = []

    for label in CLASSES:
        files = sorted(glob.glob(os.path.join(DATA_DIR, label, "*.npz")))
        print(f"\n[{label.upper()}] samples: {len(files)}")

        if not files:
            continue

        vals = []
        suspicious = []

        for path in files:
            try:
                X = load_npz(path)
                m = metrics(X)
                vals.append(m)
                reasons = flag(label, m)
                if reasons:
                    suspicious.append((path, reasons, m))
            except Exception as e:
                suspicious.append((path, [f"load_error: {e}"], {}))

        def avg(k):
            return np.mean([v[k] for v in vals]) if vals else 0

        print(f"  avg max:          {avg('max'):.2f}")
        print(f"  avg energy:       {avg('energy'):.2f}")
        print(f"  avg range_span:   {avg('range_span'):.2f}")
        print(f"  avg amp_motion:   {avg('amp_motion'):.2f}")
        print(f"  suspicious:       {len(suspicious)}")

        for path, reasons, m in suspicious[:15]:
            print(f"    - {path} -> {', '.join(reasons)}")
            if m:
                print(f"      shape={m['shape']} max={m['max']:.1f} span={m['range_span']} amp_motion={m['amp_motion']:.1f}")

        all_suspicious.extend(suspicious)

    print("\n=== SUMMARY ===")
    print(f"Total suspicious files: {len(all_suspicious)}")
    print("Note: suspicious does NOT mean bad automatically. It means: check visually or review.")

if __name__ == "__main__":
    main()
