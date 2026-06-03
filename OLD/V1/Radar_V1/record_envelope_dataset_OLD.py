#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import numpy as np
from acconeer.exptool.a111 import Client, EnvelopeServiceConfig


def json_safe(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


def record_one(label, seconds, host, outdir, range_interval, update_rate):
    outdir.mkdir(parents=True, exist_ok=True)

    client = Client(host=host)

    cfg = EnvelopeServiceConfig()
    cfg.sensor = [1]
    cfg.range_interval = list(range_interval)
    cfg.update_rate = float(update_rate)

    client.connect()
    session_info = client.setup_session(cfg)
    client.start_session()

    frames = []
    t0 = time.time()
    try:
        while time.time() - t0 < seconds:
            _, data = client.get_next()
            frames.append(np.asarray(data, dtype=np.float32))
    finally:
        client.stop_session()
        client.disconnect()

    X = np.stack(frames, axis=0)  # (T, R)

    meta = {
        "label": label,
        "seconds": float(seconds),
        "update_rate": float(update_rate),
        "range_interval_m": list(range_interval),
        "session_info": session_info,
        "frames": int(X.shape[0]),
        "bins": int(X.shape[1]),
        "recorded_unix": time.time(),
    }

    out_name = outdir / f"envelope_{label}_{int(time.time())}.npz"
    np.savez_compressed(out_name, X=X, meta=json.dumps(meta, default=json_safe))

    print(f"Saved: {out_name}")
    print(f"X shape: {X.shape}")
    print(f"Meta frames={meta['frames']} bins={meta['bins']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--range_start", type=float, default=0.20)
    ap.add_argument("--range_end", type=float, default=0.55)
    ap.add_argument("--update_rate", type=float, default=30.0)
    args = ap.parse_args()

    label = args.label.strip()
    outdir = Path(args.outdir) / label

    record_one(
        label=label,
        seconds=args.seconds,
        host=args.host,
        outdir=outdir,
        range_interval=(args.range_start, args.range_end),
        update_rate=args.update_rate,
    )


if __name__ == "__main__":
    main()
