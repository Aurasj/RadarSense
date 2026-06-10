# Arduino Bluetooth Car — Experimental Integration

> **Status: Experimental example — not part of the core RadarSense runtime.**

This optional integration lets you drive a 4-wheel Arduino car using RadarSense
hand gestures over Bluetooth. It is self-contained and lives entirely in
`examples/arduino/`. No core RadarSense files are modified.

---

## How it works

```
Radar → app/server.py ──UDP:5006──► arduino_bridge.py ──Serial/BT──► Arduino ──► Motors
                      └──UDP:5005──► Unity (unchanged)
```

`server.py` broadcasts every confirmed gesture to **both** ports:

| Port | Normal consumer |
|------|----------------|
| 5005 | Unity (game engine) — **do not change** |
| 5006 | `gesture_monitor.py` **or** `arduino_bridge.py` |

> ⚠️ Only **one** process can bind UDP port 5006 at a time.  
> Close `gesture_monitor.py` before starting `arduino_bridge.py`, and vice versa.

---

## Hardware requirements

- Arduino Uno or Mega
- Adafruit Motor Shield v1 (AFMotor library)
- HC-05 or HC-06 Bluetooth module wired to Arduino's hardware serial
- 4 DC motors + car chassis

---

## Setup

### 1. Flash the Arduino

Open `BLUETOOTH_CAR.ino` in the Arduino IDE.  
Install the **AFMotor** library (Sketch → Include Library → Manage Libraries → search `AFMotor`).  
Select your board and port, then upload.

### 2. Pair the Bluetooth module

Pair the HC-05/HC-06 with your Windows PC (Settings → Bluetooth).  
Note the COM port assigned to the "Serial Port" service (e.g. `COM6`).

### 3. Install the Python dependency

```
pip install -r examples/arduino/requirements_arduino.txt
```

This installs `pyserial`. It is kept separate from `requirements.txt` so it does
not affect the core RadarSense install.

### 4. Edit the COM port (if needed)

The default COM port is `COM6`. Override it with `--com`:

```
python examples/arduino/arduino_bridge.py --com COM8
```

---

## Running

Start `server.py` first, then open a **separate terminal** for the bridge:

```
python examples/arduino/arduino_bridge.py
```

All options:

```
python examples/arduino/arduino_bridge.py --port 5006 --com COM6 --baud 9600 --watchdog 0.8
```

Example with a longer watchdog for a slower radar framerate:

```
python examples/arduino/arduino_bridge.py --com COM6 --watchdog 1.0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `5006`  | UDP port to listen on |
| `--com`  | `COM6`  | Bluetooth serial COM port |
| `--baud` | `9600`  | Serial baud rate |
| `--watchdog` | `0.8` | Watchdog timeout in seconds |

Stop with **Ctrl+C**. The bridge sends a stop command and closes the serial port cleanly.

---

## Gesture → car command mapping

| Gesture | Arduino command | Car action |
|---------|----------------|------------|
| `hold`  | `F`            | Forward (sustained while gesture is active) |
| `tap`   | `B`            | Backward |
| `push`  | `L`            | Left turn |
| `pull`  | `R`            | Right turn |
| `wave`  | `M`            | Play music |
| `none` / unknown | `S`  | Stop |

---

## Watchdog / auto-stop

A background thread monitors UDP traffic. If no packet arrives for **0.8 seconds**
the bridge automatically sends `S` (stop) to the car. This protects against:

- The user removing their hand without the radar detecting a clean `none`.
- The network connection dropping briefly.

The watchdog is reset on every UDP packet received, including `none`.

---

## Known limitations

- Only **one** consumer can bind UDP 5006 at a time — cannot run alongside `gesture_monitor.py`.
- If the Bluetooth serial connection drops mid-session, outgoing commands are silently discarded. The car will stop via the watchdog within 0.8 s.
- `BLUETOOTH_CAR.ino` requires the Adafruit AFMotor library; it will not compile without it.
- The bridge does not reconnect automatically if serial is lost; restart it manually.
