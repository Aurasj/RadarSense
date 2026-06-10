"""
RadarSense → Arduino Bluetooth car bridge.

Listens for confirmed gesture events from server.py over UDP and forwards
single-byte commands to an Arduino car via a Bluetooth serial connection.

Usage:
    python examples/arduino/arduino_bridge.py [--port PORT] [--com COM] [--baud BAUD]
"""

import argparse
import datetime
import socket
import sys
import threading
import time

import serial

# ── Configuration defaults ─────────────────────────────────────────────────────
# All values can be overridden via command-line arguments (see --help).

_DEFAULT_UDP_PORT = 5006   # gesture_monitor port — NOT Unity's 5005
_DEFAULT_COM_PORT = "COM6"
_DEFAULT_BAUD_RATE = 9600

# How long to wait without a UDP packet before sending "S" (stop) to the car.
# HOLD fires ~30 times/s, so this must be well above 1/30 ≈ 0.033 s.
# 0.8 s gives comfortable headroom while still feeling responsive.
_DEFAULT_WATCHDOG_TIMEOUT = 0.8

# Gesture label (lowercase) → single-byte Arduino command.
# Any gesture not listed here (including "none") maps to "S" (stop).
COMMAND_MAP = {
    "hold": "F",   # forward
    "tap":  "B",   # backward
    "push": "L",   # left turn
    "pull": "R",   # right turn
    "wave": "M",   # play music
}


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "RadarSense → Arduino Bluetooth car bridge.\n"
            "Listens for confirmed gesture events from server.py (UDP) and\n"
            "forwards single-byte commands to an Arduino over serial/Bluetooth."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--port", type=int, default=_DEFAULT_UDP_PORT,
        help=f"UDP port to listen on (default: {_DEFAULT_UDP_PORT})",
    )
    p.add_argument(
        "--com", type=str, default=_DEFAULT_COM_PORT,
        help=f"Serial/Bluetooth COM port (default: {_DEFAULT_COM_PORT})",
    )
    p.add_argument(
        "--baud", type=int, default=_DEFAULT_BAUD_RATE,
        help=f"Serial baud rate (default: {_DEFAULT_BAUD_RATE})",
    )
    p.add_argument(
        "--watchdog", type=float, default=_DEFAULT_WATCHDOG_TIMEOUT,
        help=f"Watchdog timeout in seconds (default: {_DEFAULT_WATCHDOG_TIMEOUT})",
    )
    return p.parse_args()


# ── Serial connection ──────────────────────────────────────────────────────────

def _open_serial(com_port: str, baud_rate: int) -> "serial.Serial | None":
    """Try to open the serial port. Returns the Serial object, or None on failure."""
    print(f"[Bridge] Connecting to Arduino on {com_port} at {baud_rate} baud …")
    try:
        ser = serial.Serial(com_port, baud_rate, timeout=0.1, dsrdtr=True)
        time.sleep(1.5)  # allow the Bluetooth module time to initialise
        print("[Bridge] Serial connected. Watchdog active.")
        return ser
    except Exception as exc:
        print(f"[Bridge] Serial connection failed: {exc}")
        print("[Bridge] Continuing without serial — gestures will be logged only.")
        return None


# ── UDP socket ─────────────────────────────────────────────────────────────────

def _bind_udp(udp_port: int) -> socket.socket:
    """Bind a UDP socket. Exits with a clear message if the port is already in use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 0.5 s receive timeout lets Ctrl+C be handled promptly on Windows.
    sock.settimeout(0.5)
    try:
        sock.bind(("0.0.0.0", udp_port))
    except OSError as exc:
        print(f"[Bridge] Cannot bind UDP port {udp_port}: {exc}")
        print("[Bridge] Is gesture_monitor.py already running on this port?")
        sys.exit(1)
    return sock


# ── Command sender ─────────────────────────────────────────────────────────────

_last_cmd: str = "S"


def send_cmd(cmd: str, ser: "serial.Serial | None") -> None:
    """Send a single-byte command to the Arduino, but only when the command changes."""
    global _last_cmd
    if cmd == _last_cmd:
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] Serial → [{cmd}]")
    if ser:
        try:
            ser.write(cmd.encode())
        except Exception:
            pass  # serial dropped mid-session — carry on receiving UDP
    _last_cmd = cmd


# ── Watchdog thread ────────────────────────────────────────────────────────────

_last_udp_time: float = time.time()


def _watchdog(watchdog_timeout: float, ser: "serial.Serial | None") -> None:
    """
    Safety thread: sends "S" (stop) if no UDP packet arrives within
    watchdog_timeout seconds. Protects against dropped connections or the
    user removing their hand without the radar producing a clean "none".
    """
    while True:
        if time.time() - _last_udp_time > watchdog_timeout:
            send_cmd("S", ser)
        time.sleep(0.05)


# ── Main loop ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()

    ser  = _open_serial(args.com, args.baud)
    sock = _bind_udp(args.port)

    print(f"\n[Bridge] Listening on UDP port {args.port} (gesture_monitor port).")
    print("[Bridge] Press Ctrl+C at any time to stop the car and exit.\n")

    # Start the watchdog before entering the receive loop.
    wdog = threading.Thread(
        target=_watchdog, args=(args.watchdog, ser), daemon=True,
    )
    wdog.start()

    try:
        while True:
            try:
                data, _ = sock.recvfrom(1024)
            except socket.timeout:
                continue  # loop back so Ctrl+C is checked every 0.5 s

            gesture = data.decode(errors="ignore").strip().lower()
            _last_udp_time = time.time()  # reset watchdog timer on every packet

            cmd = COMMAND_MAP.get(gesture, "S")

            if gesture == "none":
                print("[Bridge] Radar clear → stop")
            elif cmd != "S":
                print(f"[Bridge] UDP recv: {gesture.upper():<4}  → [{cmd}]")

            send_cmd(cmd, ser)

    except KeyboardInterrupt:
        print("\n[Bridge] Ctrl+C — emergency stop.")
        send_cmd("S", ser)
        send_cmd("N", ser)  # stop music if it is playing
        if ser:
            ser.close()
            print("[Bridge] Serial port closed.")
        sock.close()
        print("[Bridge] Clean exit.")
        sys.exit(0)