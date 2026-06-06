#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UDP monitor for confirmed RadarSense gesture events.

Receives gesture labels from server.py and displays them
in the terminal with basic session statistics.
"""

import datetime as _dt
import socket
import time
from collections import Counter, deque

PORT            = 5006
LISTEN_IP       = "0.0.0.0"
HISTORY_LEN     = 12
RATE_WINDOW_SEC = 60

# ANSI color codes.
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
HDR   = "\033[38;5;51m"
MUTED = "\033[38;5;245m"

# Color, emoji, and label for each gesture.
GESTURE_STYLE: dict[str, tuple[str, str, str]] = {
    "push": ("\033[38;5;39m",  "👉", "PUSH"),
    "pull": ("\033[38;5;208m", "👈", "PULL"),
    "tap":  ("\033[38;5;226m", "👆", "TAP"),
    "wave": ("\033[38;5;129m", "👋", "WAVE"),
    "hold": ("\033[38;5;46m",  "✋", "HOLD"),
}
UNKNOWN_STYLE = ("\033[97m", "◆", "????")


def _style(gesture: str) -> tuple[str, str, str]:
    return GESTURE_STYLE.get(gesture.lower(), UNKNOWN_STYLE)


def _now() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S")


def _bar(value: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "░" * width
    filled = max(0, min(width, int(round((value / total) * width))))
    return "█" * filled + "░" * (width - filled)


def _rate(timestamps: deque, start_time: float) -> float:
    """Events per minute over the last 60 seconds."""
    now    = time.time()
    cutoff = now - RATE_WINDOW_SEC
    while timestamps and timestamps[0] < cutoff:
        timestamps.popleft()
    elapsed = min(now - start_time, RATE_WINDOW_SEC)
    return 0.0 if elapsed < 1.0 else len(timestamps) / elapsed * 60.0


def main() -> None:
    # Bind socket — exit with a clear message if the port is already in use.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.5)
    try:
        sock.bind((LISTEN_IP, PORT))
    except OSError as exc:
        print(f"[ERROR] Cannot bind to port {PORT}: {exc}")
        print(f"        Is another process already using it?")
        raise SystemExit(1)

    total      = 0
    counters   = Counter()
    history    = deque(maxlen=HISTORY_LEN)
    timestamps = deque()
    start_time = time.time()

    # Header
    print(f"\n{HDR}RadarSense Gesture Monitor{RESET}")
    print(f"Listening on UDP {LISTEN_IP}:{PORT}")
    print(f"{DIM}Press Ctrl+C to stop{RESET}\n")

    try:
        while True:
            try:
                data, _ = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break

            gesture = data.decode(errors="ignore").strip().lower()
            if not gesture or gesture == "none":
                continue

            total += 1
            counters[gesture] += 1
            history.append((_now(), gesture))
            timestamps.append(time.time())

            color, icon, label = _style(gesture)
            rate  = _rate(timestamps, start_time)
            count = counters[gesture]

            print(
                f"  {MUTED}[{_now()}]{RESET}  "
                f"{color}{BOLD}{icon} {label:<4}{RESET}  "
                f"#{total:<4}  "
                f"{DIM}total={count}  rate={rate:.1f}/min{RESET}"
            )

    except KeyboardInterrupt:
        pass
    finally:
        # Summary
        elapsed = time.time() - start_time
        print(f"\n{HDR}Session summary:{RESET}")
        print(f"  Total events : {BOLD}{total}{RESET}")
        print(f"  Duration     : {BOLD}{str(_dt.timedelta(seconds=int(elapsed)))}{RESET}")
        if total > 0 and elapsed > 0:
            print(f"  Average rate : {BOLD}{total / elapsed * 60:.2f}/min{RESET}")

        if counters:
            print(f"\n  {DIM}Distribution:{RESET}")
            for g, c in sorted(counters.items(), key=lambda x: (-x[1], x[0])):
                color, icon, label = _style(g)
                pct = c / total * 100
                print(f"  {color}{icon} {label:<4}{RESET}  {_bar(c, total)}  {c:>3} ({pct:.1f}%)")

        if history:
            print(f"\n  {DIM}Recent events:{RESET}")
            for ts, g in reversed(history):
                color, icon, label = _style(g)
                print(f"  {MUTED}{ts}{RESET}  {color}{icon} {label}{RESET}")

        print()
        sock.close()


if __name__ == "__main__":
    main()
