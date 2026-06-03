#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gesture_event_monitor.py — UDP monitor for confirmed gesture events
===================================================================

Production-friendly companion monitor for AntRadar/serverV2.py.
It listens on the monitor UDP port and displays only confirmed gesture
pulses sent by the server. The main server terminal stays clean; this
monitor is used only when you want a terminal view of gesture events.

Usage:
    python gesture_event_monitor.py
    python gesture_event_monitor.py --port 5006
    python gesture_event_monitor.py --log gestures.log
    python gesture_event_monitor.py --compact
"""

import argparse
import datetime as _dt
import socket
import sys
import time
from collections import Counter, deque

DEFAULT_PORT = 5006
LISTEN_IP = "0.0.0.0"
HISTORY_LEN = 12
RATE_WINDOW_SEC = 60

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
HDR = "\033[38;5;51m"
MUTED = "\033[38;5;245m"
OK = "\033[38;5;46m"

GESTURE_STYLE = {
    "push": ("\033[38;5;39m", "👉", "PUSH"),
    "pull": ("\033[38;5;208m", "👈", "PULL"),
    "tap": ("\033[38;5;226m", "👆", "TAP"),
    "wave": ("\033[38;5;129m", "👋", "WAVE"),
    "hold": ("\033[38;5;46m", "✋", "HOLD"),
    "none": ("\033[90m", "·", "NONE"),
}
DEFAULT_STYLE = ("\033[97m", "◆", "EVENT")


def _style(gesture: str):
    return GESTURE_STYLE.get(gesture.lower(), DEFAULT_STYLE)


def _now() -> str:
    return _dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _safe_bar(value: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "░" * width
    filled = int(round((value / total) * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


class GestureEventMonitor:
    def __init__(self, port: int, log_path: str | None, compact: bool = False):
        self.port = port
        self.log_path = log_path
        self.compact = compact
        self.log_file = None

        self.total = 0
        self.counters = Counter()
        self.history = deque(maxlen=HISTORY_LEN)
        self.timestamps = deque()
        self.start_time = time.time()
        self.running = True

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(0.5)
        self.sock.bind((LISTEN_IP, self.port))

    def open_log(self):
        if not self.log_path:
            return
        self.log_file = open(self.log_path, "a", encoding="utf-8")
        self.log_file.write(f"\n--- AntRadar event monitor started {_dt.datetime.now().isoformat()} ---\n")
        self.log_file.flush()

    def close_log(self):
        if not self.log_file:
            return
        self.log_file.write(f"--- session ended {_dt.datetime.now().isoformat()} | total={self.total} ---\n")
        self.log_file.close()

    def write_log(self, ts: str, gesture: str, addr):
        if not self.log_file:
            return
        self.log_file.write(f"[{ts}] {gesture.upper()} from {addr[0]}:{addr[1]}\n")
        self.log_file.flush()

    def rate_per_minute(self) -> float:
        now = time.time()
        cutoff = now - RATE_WINDOW_SEC
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        elapsed = min(now - self.start_time, RATE_WINDOW_SEC)
        if elapsed < 1.0:
            return 0.0
        return len(self.timestamps) / elapsed * 60.0

    def header(self):
        print(f"\n{HDR}{'═' * 72}{RESET}")
        print(f"{HDR}{' ANTRADAR GESTURE EVENT MONITOR ':^72}{RESET}")
        print(f"{HDR}{' UDP ' + LISTEN_IP + ':' + str(self.port) + '  |  Ctrl+C to stop ':^72}{RESET}")
        print(f"{HDR}{'═' * 72}{RESET}\n")
        print(f"  {DIM}Main server terminal stays clean. Confirmed events appear here.\n{RESET}")

    def print_event(self, ts: str, gesture: str):
        color, icon, label = _style(gesture)
        rate = self.rate_per_minute()
        count = self.counters[gesture]
        if self.compact:
            print(f"[{ts}] {label:<5} #{self.total} rate={rate:.1f}/min")
            return
        print(
            f"  {MUTED}[{ts}]{RESET}  "
            f"{color}{BOLD}{icon} {label:<5}{RESET}  "
            f"{DIM}event #{self.total:<4}{RESET}  "
            f"{OK}{count:>3}x{RESET}  "
            f"{MUTED}rate {rate:4.1f}/min{RESET}"
        )

    def summary(self):
        elapsed = time.time() - self.start_time
        print(f"\n{HDR}{'─' * 72}{RESET}")
        print(f"{HDR} Session summary{RESET}")
        print(f"  Total events: {BOLD}{self.total}{RESET}")
        print(f"  Duration:     {BOLD}{str(_dt.timedelta(seconds=int(elapsed)))}{RESET}")
        if elapsed > 0:
            print(f"  Avg rate:     {BOLD}{self.total / elapsed * 60:.2f}/min{RESET}")

        if self.total:
            print(f"\n  {DIM}Distribution:{RESET}")
            for gesture, count in sorted(self.counters.items(), key=lambda x: (-x[1], x[0])):
                color, icon, label = _style(gesture)
                pct = count / self.total * 100
                print(f"  {color}{icon} {label:<5}{RESET} {_safe_bar(count, self.total)} {count:>3} ({pct:4.1f}%)")

            print(f"\n  {DIM}Recent history:{RESET}")
            for ts, gesture in reversed(self.history):
                color, icon, label = _style(gesture)
                print(f"  {MUTED}{ts}{RESET}  {color}{icon} {label}{RESET}")
        print(f"{HDR}{'─' * 72}{RESET}\n")

    def run(self):
        self.open_log()
        self.header()
        try:
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(256)
                except socket.timeout:
                    continue
                except OSError:
                    break

                gesture = data.decode(errors="ignore").strip().lower()
                if not gesture:
                    continue

                ts = _now()
                self.total += 1
                self.counters[gesture] += 1
                self.history.append((ts, gesture))
                self.timestamps.append(time.time())

                self.print_event(ts, gesture)
                self.write_log(ts, gesture, addr)
        except KeyboardInterrupt:
            print(f"\n{DIM}Stopping monitor...{RESET}")
        finally:
            self.summary()
            self.close_log()
            self.sock.close()


def main():
    parser = argparse.ArgumentParser(description="Monitor confirmed AntRadar gesture events over UDP.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"UDP port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--log", type=str, default=None, help="Optional event log file, e.g. gestures.log")
    parser.add_argument("--compact", action="store_true", help="Print compact one-line events without colors/panels")
    args = parser.parse_args()

    GestureEventMonitor(port=args.port, log_path=args.log, compact=args.compact).run()


if __name__ == "__main__":
    main()
