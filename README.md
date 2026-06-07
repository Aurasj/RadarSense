# RadarSense

Real-time radar gesture recognition using the SparkFun Acconeer A111 sensor, a TinyCNN classifier, and a Flask/Socket.IO dashboard. Confirmed gesture events are broadcast over UDP for integration with Unity or any external application.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Supported Gestures](#supported-gestures)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [Running the Gesture Monitor](#running-the-gesture-monitor)
- [Configuration](#configuration)
- [Training and Evaluation](#training-and-evaluation)
- [UDP Integration](#udp-integration)
- [External Integration](#external-integration)
- [Model Performance](#model-performance)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

RadarSense is an end-to-end pipeline that turns raw millimetre-wave radar data into discrete, stabilised gesture events in real time. It runs on a standard laptop or desktop connected over the network to a Raspberry Pi hosting the Acconeer A111 sensor. A browser-based dashboard visualises the live radar waterfall and CNN predictions; confirmed gestures are forwarded over UDP to Unity or any listener on the local network.

The project was built as a research and demonstration platform, not a commercial product. It is intended for prototyping gesture-driven interfaces and HCI experiments.

---

## System Architecture

```
SparkFun Acconeer A111 (radar sensor)
        │  RF envelope frames (35 cm window)
        ▼
Raspberry Pi  ←  Acconeer Exploration Server (TCP :6110)
        │  raw envelope frames  (TCP socket)
        ▼
RadarSense server.py  (Flask + Socket.IO, port 5002)
        │
        ├─► GestureEngine  (gesture_engine.py)
        │       Sliding window (60 frames)
        │       Noise gate → TinyCNN inference → EMA smoothing
        │       Fast-track → Debounce FSM → Cooldown
        │
        ├─► Browser dashboard  (Socket.IO — radar waterfall + live predictions)
        │
        └─► UDP broadcast  (port 5005 → Unity / port 5006 → gesture_monitor.py)
```

The radar runs at 30 fps. Each inference cycle operates on a 60-frame sliding window (≈ 2 seconds of history), resampled to a fixed 60 × 128 input tensor. Gesture stabilisation is handled entirely in software — no hardware filtering is required beyond the sensor's built-in envelope service.

---

## Features

- **Live radar streaming** — 30 fps envelope frames streamed to the browser via Socket.IO.
- **TinyCNN inference** — lightweight 3-block 2-D convolutional network (~115 KB checkpoint).
- **Gesture stabilisation pipeline** — noise gate, EMA probability smoothing, per-gesture fast-track, debounce FSM, and per-class cooldown windows.
- **Web dashboard** — real-time radar waterfall, per-class probability bars, FSM state readout, and FPS counter. Start/stop/rebaseline the radar directly from the browser.
- **UDP event bus** — confirmed gesture labels broadcast as plain UTF-8 strings over UDP to Unity (port 5005) and the local monitor (port 5006).
- **Gesture monitor** — standalone terminal tool that displays incoming UDP events with colour, session statistics, and a per-gesture distribution summary.
- **Full training stack** — dataset recorder, training script, ONNX export, confusion matrix, and classification report.
- **Environment-variable configuration** — Pi host, ports, range window, update rate, and verbosity are all overridable without touching source code.

---

## Hardware Requirements

| Component | Details |
|---|---|
| Radar sensor | SparkFun Acconeer XB112 / A111 (envelope service mode) |
| Edge device | Raspberry Pi (any model with Ethernet/Wi-Fi and Python 3.9+) |
| Host machine | Any OS running Python 3.10+, reachable from the Pi on the same LAN |
| Network | Pi and host machine must share a local network (LAN or direct Ethernet) |

The Pi runs the **Acconeer Exploration Server** (`exploration_server`) which opens a TCP socket (default port 6110). RadarSense connects to that socket; no custom firmware is required.

---

## Supported Gestures

| Gesture | Description |
|---|---|
| `none` | No hand detected in the sensing range |
| `hold` | Hand held stationary in front of the sensor |
| `push` | Hand moved toward the sensor |
| `pull` | Hand moved away from the sensor |
| `tap` | Sharp forward jab and retraction |
| `wave` | Lateral hand wave across the sensor face |

---

## Project Structure

```
RadarSense/
├── app/
│   ├── server.py              # Flask + Socket.IO server, radar thread, UDP broadcast
│   ├── gesture_engine.py      # TinyCNN model, preprocessing, FSM stabilisation
│   └── gesture_monitor.py     # UDP event monitor (standalone terminal tool)
│
├── cfg/
│   ├── gesture_cnn_boss.pt    # Trained PyTorch checkpoint (~115 KB)
│   └── gesture_cnn_meta.json  # Model metadata (labels, T, R, normalisation stats)
│
├── training/
│   ├── record_envelope_dataset.py  # Dataset recorder (requires live radar connection)
│   ├── audit_dataset.py            # Dataset integrity checker and statistics
│   ├── train_V3.py                 # Training script (TinyCNN, v3 pipeline)
│   ├── test_model.py               # Standalone model evaluation
│   ├── reports/                    # Training artifacts (curves, confusion matrix, report)
│   └── onnx/                       # ONNX export of the trained model
│
├── web/
│   ├── templates/index.html        # Dashboard HTML
│   └── static/
│       ├── style.css               # Dashboard styles
│       ├── js/                     # Dashboard JavaScript
│       └── reports/                # Report assets served to the browser
│
├── requirements.txt           # Runtime dependencies
├── requirements_train.txt     # Training/evaluation dependencies
├── LICENSE
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aurasj/RadarSense.git
cd RadarSense
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install runtime dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `acconeer-exptool==7.12.2` is pinned. This is the version verified against the SparkFun A111 Exploration Server. Do not upgrade it without re-testing the connection layer.

### 4. Set up the Raspberry Pi

Install and run the Acconeer Exploration Server on the Pi. Refer to the [Acconeer documentation](https://developer.acconeer.com/) for platform-specific instructions. The server must be reachable on TCP port 6110 from the host machine.

---

## Running the Server

```bash
cd app
python server.py
```

The server starts on `http://localhost:5002` and opens the dashboard in the default browser automatically. From the dashboard you can:

- **Start radar** — connects to the Pi and begins streaming frames.
- **Stop radar** — cleanly disconnects and waits for the Pi to release the port.
- **Rebaseline** — resets the FSM and EMA without stopping the radar (useful after repositioning the sensor).

### Environment variables

All settings can be overridden at launch:

```bash
PI_HOST=raspberrypi PI_PORT=6110 SERVER_PORT=5002 \
UNITY_IP=192.168.1.50 UNITY_PORT=5005 MONITOR_PORT=5006 \
RANGE_START=0.20 RANGE_END=0.55 UPDATE_RATE=30 \
VERBOSE_GESTURES=1 python server.py
```

See the [Configuration](#configuration) section for a full reference.

---

## Running the Gesture Monitor

`gesture_monitor.py` is a standalone terminal tool that listens for UDP events from the server and displays them with colour-coded output and session statistics. It does not require the radar to be connected — it only needs the server to be running and sending UDP packets.

```bash
cd app
python gesture_monitor.py
```

Output example:

```
RadarSense Gesture Monitor
Listening on UDP 0.0.0.0:5006
Press Ctrl+C to stop

  [14:03:22]  ✋ HOLD  #1    total=1  rate=1.2/min
  [14:03:25]  👉 PUSH  #2    total=1  rate=1.8/min
  [14:03:27]  👆 TAP   #3    total=1  rate=2.1/min
```

Press `Ctrl+C` to stop. A session summary with gesture distribution and average rate is printed on exit.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PI_HOST` | `auraspi` | Hostname or IP of the Raspberry Pi |
| `PI_PORT` | `6110` | TCP port of the Acconeer Exploration Server |
| `SERVER_PORT` | `5002` | Port the Flask dashboard listens on |
| `RANGE_START` | `0.20` | Start of sensing range in metres |
| `RANGE_END` | `0.55` | End of sensing range in metres |
| `UPDATE_RATE` | `30` | Radar update rate in fps |
| `UNITY_IP` | `127.0.0.1` | Target IP for UDP gesture events |
| `UNITY_PORT` | `5005` | UDP port for Unity |
| `MONITOR_PORT` | `5006` | UDP port for the gesture monitor |
| `VERBOSE_GESTURES` | `0` | Set to `1` to print every confirmed gesture to the terminal |
| `AUTO_STOP_ON_NO_CLIENTS` | `0` | Set to `1` to stop the radar when all browser tabs close |
| `DASHBOARD_BINS` | `128` | Bins used for the dashboard waterfall display |

---

## Training and Evaluation

Install training dependencies first:

```bash
pip install -r requirements_train.txt
```

### Record a dataset

Requires a live radar connection.

```bash
cd training
python record_envelope_dataset.py
```

Follow the on-screen prompts to record labelled clips for each gesture class.

### Audit the dataset

```bash
cd training
python audit_dataset.py
```

Prints per-class frame counts, detects missing or malformed samples, and reports the train/val split.

### Train the model

```bash
cd training
python train_V3.py
```

Trains TinyCNN on the recorded dataset. The best checkpoint is saved to `cfg/gesture_cnn_boss.pt` and training artifacts (loss curve, accuracy curve, confusion matrix, classification report) are written to `training/reports/`.

### Evaluate a saved checkpoint

```bash
cd training
python test_model.py
```

Loads `cfg/gesture_cnn_boss.pt` and runs it against the validation split. Prints the classification report and displays the confusion matrix.

---

## UDP Integration

The server sends confirmed gesture events as plain UTF-8 strings over UDP. No framing, no headers — just the gesture label followed by nothing.

**Ports:**

| Port | Consumer |
|---|---|
| `5005` | Unity (or any primary listener) |
| `5006` | `gesture_monitor.py` (or a secondary listener) |

**Payload format:**

```
hold\n   →  "hold"
push\n   →  "push"
pull\n   →  "pull"
tap\n    →  "tap"
wave\n   →  "wave"
```

`none` is never sent over UDP — only confirmed, non-null gestures are broadcast.

**Rate-limiting behaviour:**

- `tap` and `wave` are suppressed if a second event follows within 1.25 seconds of the previous one.
- `push` and `pull` are suppressed for 0.30 seconds after any primary gesture (`hold`, `tap`, `wave`) fires.
- `hold` events pass through unconditionally on every confirmation cycle — the receiving application is expected to handle hold repetition as needed.

**Minimal Python listener example:**

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))

while True:
    data, _ = sock.recvfrom(256)
    gesture = data.decode().strip()
    print(f"Gesture received: {gesture}")
```

---

## External Integration

RadarSense is designed to be integration-agnostic. Any application that can read a UDP socket receives gesture events as plain UTF-8 strings — no SDK or library required. Unity, Unreal Engine, custom tools, and scripts all integrate the same way: bind to `UNITY_IP:UNITY_PORT` (default `127.0.0.1:5005`) and read the incoming label.

The payload format is described in the [UDP Integration](#udp-integration) section above.

---

## Model Performance

**Model:** TinyCNN v3  
**Checkpoint:** `cfg/gesture_cnn_boss.pt` (~115 KB)  
**Input:** 60 frames × 128 range bins (log1p + z-score normalised)  
**Training set:** 920 samples | **Validation set:** 230 samples  
**Best epoch:** 14

| Class | Precision | Recall | F1 |
|---|---|---|---|
| none | 1.000 | 1.000 | 1.000 |
| hold | 1.000 | 0.967 | 0.983 |
| push | 1.000 | 1.000 | 1.000 |
| pull | 1.000 | 1.000 | 1.000 |
| tap | 1.000 | 1.000 | 1.000 |
| wave | 0.968 | 1.000 | 0.984 |
| **overall** | **0.996** | **0.996** | **0.996** |

**Validation accuracy: 99.57%**

> These figures reflect controlled recording conditions at 20–55 cm. Real-world performance depends on the sensor placement, the subject, ambient RF noise, and whether the recording environment matches the training environment. The model has not been evaluated across multiple subjects or in noisy RF environments.

---

## Limitations

- **Single-user, single-sensor.** The pipeline assumes one hand in one detection zone. Multiple simultaneous hands or sensors are not supported.
- **Training data is limited.** The dataset totals ~1,150 labelled clips. Performance may degrade for gestures that were underrepresented or recorded by a single subject.
- **Fixed range window.** The sensing range is hardcoded at 20–55 cm at training time. Changing `RANGE_START`/`RANGE_END` without retraining will reduce accuracy.
- **Network dependency.** The server requires a stable local network connection to the Pi. Packet loss or latency spikes can cause frame drops and degrade the gesture window.
- **No authentication.** The UDP broadcast and the Socket.IO server accept connections from any host on the network. Do not expose the server ports to an untrusted network.
- **Not production-ready.** RadarSense is a research prototype. It has not been hardened for continuous unattended operation.

---

## License

MIT License — see [LICENSE](LICENSE) for the full text.

Copyright © 2026 Aurasj
