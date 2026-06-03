"""
unity_monitor.py  —  Live monitor for comenzile trimise catre Unity
====================================================================
Asculta pe UDP portul 5006 (MONITOR_PORT din serverV2.py) si afiseaza
fiecare comanda primita cu:
  - Timestamp precis
  - Iconita si culoare per gest
  - Contor cumulativ si frecventa (gesturi/minut)
  - Istoric recent in format tabel
  - Statistici totale la iesire

Rulare:
    python unity_monitor.py

    # Port diferit:
    python unity_monitor.py --port 5006

    # Salveaza log:
    python unity_monitor.py --log gestures.log
"""

import argparse
import datetime
import signal
import socket
import sys
import time
from collections import Counter, deque

# ══════════════════════════════════════════════════════════════
# CONFIGURATIE
# ══════════════════════════════════════════════════════════════

DEFAULT_PORT    = 5006      # trebuie sa corespunda cu MONITOR_PORT din serverV2.py
LISTEN_IP       = "0.0.0.0"
HISTORY_LEN     = 20        # cate gesturi recente aratam in coloana din dreapta
RATE_WINDOW_SEC = 60        # fereastra pentru calculul ratei (gesturi/minut)

# ══════════════════════════════════════════════════════════════
# ANSI COLORS & ICONS
# ══════════════════════════════════════════════════════════════

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# Per-gesture styling
GESTURE_STYLE = {
    "push":  ("\033[38;5;39m",  "👉", "PUSH  "),   # albastru deschis
    "pull":  ("\033[38;5;208m", "👈", "PULL  "),   # portocaliu
    "tap":   ("\033[38;5;226m", "👆", "TAP   "),   # galben
    "wave":  ("\033[38;5;129m", "👋", "WAVE  "),   # violet
    "hold":  ("\033[38;5;46m",  "✋", "HOLD  "),   # verde
    "none":  ("\033[90m",        "🫥", "NONE  "),   # gri
}
DEFAULT_STYLE = ("\033[97m", "🎯", "???   ")

# Header colors
HDR  = "\033[38;5;51m"       # cyan deschis
SEP  = "\033[38;5;240m"      # gri inchis
STAT = "\033[38;5;220m"      # auriu

def _style(gesture: str):
    return GESTURE_STYLE.get(gesture.lower(), DEFAULT_STYLE)

def _now_str() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

def _clear_line():
    sys.stdout.write("\r\033[2K")

def _move_up(n: int):
    sys.stdout.write(f"\033[{n}A")

# ══════════════════════════════════════════════════════════════
# MONITOR CLASS
# ══════════════════════════════════════════════════════════════

class UnityMonitor:
    def __init__(self, port: int, log_path: str | None):
        self.port     = port
        self.log_path = log_path
        self._log_file = None

        self.total_count = 0
        self.counters: Counter = Counter()
        self.history: deque = deque(maxlen=HISTORY_LEN)
        self.timestamps: deque = deque()    # for rate calculation
        self.start_time = time.time()
        self._running = True

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(0.5)   # permite iesire curata la Ctrl+C
        self.sock.bind((LISTEN_IP, self.port))

    def _open_log(self):
        if self.log_path:
            self._log_file = open(self.log_path, "a", encoding="utf-8")
            self._log_file.write(
                f"\n{'='*60}\n"
                f"  Session started: {datetime.datetime.now().isoformat()}\n"
                f"{'='*60}\n"
            )
            self._log_file.flush()

    def _write_log(self, gesture: str, ts: str):
        if self._log_file:
            self._log_file.write(f"[{ts}] {gesture.upper()}\n")
            self._log_file.flush()

    def _close_log(self):
        if self._log_file:
            self._log_file.write(
                f"\n{'='*60}\n"
                f"  Session ended:   {datetime.datetime.now().isoformat()}\n"
                f"  Total gestures:  {self.total_count}\n"
                f"{'='*60}\n"
            )
            self._log_file.close()

    # ──────────────────────────────────────────────────────────
    # Drawing helpers
    # ──────────────────────────────────────────────────────────

    def _draw_header(self):
        width = 70
        print(f"\n{HDR}{'═'*width}{RESET}")
        print(f"{HDR}{'  UNITY GESTURE MONITOR':^{width}}{RESET}")
        print(f"{HDR}  Ascultare UDP 0.0.0.0:{self.port}   "
              f"[Ctrl+C pentru iesire]{' '*(width-50)}{RESET}")
        print(f"{HDR}{'═'*width}{RESET}\n")

    def _draw_event(self, gesture: str, ts: str, total: int):
        color, icon, label = _style(gesture)
        rate = self._rate_per_minute()
        print(
            f"  {DIM}[{ts}]{RESET}  "
            f"{color}{BOLD}{icon} {label}{RESET}"
            f"{DIM}  #{total:<4}{RESET}"
            f"  {STAT}{color}(total {self.counters[gesture.lower()]:>3}x | "
            f"rata: {rate:.1f}/min){RESET}"
        )

    def _draw_history_panel(self):
        """Print the last N gestures as a mini-table."""
        width = 70
        print(f"\n{SEP}{'─'*width}{RESET}")
        print(f"{SEP}  ISTORIC ({min(len(self.history), HISTORY_LEN)} gesturi recente){RESET}")
        print(f"{SEP}{'─'*width}{RESET}")

        for i, (ts, g) in enumerate(reversed(list(self.history))):
            color, icon, label = _style(g)
            age_marker = f"{DIM}◄{RESET}" if i == 0 else " "
            print(f"  {age_marker} {DIM}{ts}{RESET}  {color}{icon} {label}{RESET}")

    def _draw_stats(self):
        """Summary panel printed on exit."""
        elapsed = time.time() - self.start_time
        width = 70
        print(f"\n\n{HDR}{'═'*width}{RESET}")
        print(f"{HDR}  STATISTICI SESIUNE{RESET}")
        print(f"{HDR}{'═'*width}{RESET}")
        print(f"  {STAT}Total gesturi detectate: {BOLD}{self.total_count}{RESET}")
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
        print(f"  {STAT}Durata sesiune:          {BOLD}{elapsed_str}{RESET}")
        if elapsed > 0:
            avg_rate = self.total_count / elapsed * 60
            print(f"  {STAT}Rata medie:              {BOLD}{avg_rate:.2f} gesturi/min{RESET}")
        print(f"\n  {DIM}Distributie pe tip:{RESET}")
        for gesture, cnt in sorted(self.counters.items(), key=lambda x: -x[1]):
            color, icon, label = _style(gesture)
            bar_len = int(cnt / max(self.total_count, 1) * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            pct = cnt / max(self.total_count, 1) * 100
            print(f"  {color}{icon} {label}{RESET}  {bar}  "
                  f"{color}{BOLD}{cnt:>3}{RESET} {DIM}({pct:.1f}%){RESET}")
        print(f"\n{HDR}{'═'*width}{RESET}\n")

    def _rate_per_minute(self) -> float:
        now = time.time()
        cutoff = now - RATE_WINDOW_SEC
        # clean old entries
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()
        n = len(self.timestamps)
        elapsed = min(now - self.start_time, RATE_WINDOW_SEC)
        if elapsed < 1:
            return 0.0
        return n / elapsed * 60

    # ──────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────

    def run(self):
        self._open_log()
        self._draw_header()
        print(f"  {DIM}Asteptam date de la serverV2.py ...{RESET}\n")

        signal.signal(signal.SIGINT, self._on_sigint)

        while self._running:
            try:
                data, addr = self.sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break

            gesture = data.decode(errors="ignore").strip()
            if not gesture:
                continue

            ts = _now_str()
            self.total_count += 1
            self.counters[gesture.lower()] += 1
            self.history.append((ts, gesture.lower()))
            self.timestamps.append(time.time())

            self._draw_event(gesture, ts, self.total_count)
            self._write_log(gesture, ts)

        self._draw_stats()
        self._close_log()
        self.sock.close()

    def _on_sigint(self, *_):
        print(f"\n\n  {DIM}Ctrl+C detectat — inchidem monitorul ...{RESET}")
        self._running = False


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Monitor comenzi UDP trimise catre Unity de la serverV2.py"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port UDP de ascultat (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--log", type=str, default=None,
        help="Fisier de log optional (ex: gestures.log)"
    )
    args = parser.parse_args()

    monitor = UnityMonitor(port=args.port, log_path=args.log)
    monitor.run()


if __name__ == "__main__":
    main()
