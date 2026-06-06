# RadarSense

Real-time radar gesture recognition system built with the Acconeer A111 radar sensor and a TinyCNN-based classification pipeline.

RadarSense combines radar signal acquisition, machine learning inference, gesture stabilization, and a live web dashboard into a single end-to-end system.

---

## Features

* Real-time gesture recognition
* TinyCNN inference engine
* FSM-based gesture stabilization
* Live radar waveform visualization
* Live waterfall/heatmap visualization
* Session statistics and event logging
* UDP gesture event broadcasting
* Raspberry Pi + Acconeer A111 integration
* Model evaluation report dashboard

---

## Supported Gestures

The current model recognizes the following classes:

* none
* hold
* push
* pull
* tap
* wave

---

## System Architecture

```text
Acconeer A111 Radar
        │
        ▼
 Raspberry Pi
 (Exploration Server)
        │
        ▼
 RadarSense Server
 (Flask + Socket.IO)
        │
 ┌──────┴──────┐
 ▼             ▼
Dashboard    UDP Events
(Web UI)     (Unity / External Apps)
```

---

## Project Structure

```text
RadarSense
├── app
│   ├── gesture_engine.py
│   ├── gesture_monitor.py
│   └── server.py
├── cfg
│   ├── gesture_cnn_boss.pt
│   └── gesture_cnn_meta.json
├── docs
├── logs
├── training
│   ├── onnx
│   ├── reports
│   ├── audit_dataset.py
│   ├── record_envelope_dataset.py
│   ├── test_model.py
│   └── train_V3.py
├── web
│   ├── static
│   └── templates
├── requirements.txt
└── requirements_train.txt
```

---

## Hardware

* Acconeer A111 radar sensor
* Raspberry Pi 4
* Windows/Linux host computer

---

## Runtime Dependencies

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Training dependencies:

```bash
pip install -r requirements_train.txt
```

---

## Running the System

Start the Flask server:

```bash
python app/server.py
```

Open the dashboard:

```text
http://localhost:5002
```

---

## UDP Event Monitor

Run the terminal monitor in a separate window:

```bash
python app/gesture_monitor.py
```

---

## Training

Record dataset samples:

```bash
python training/record_envelope_dataset.py
```

Train the model:

```bash
python training/train_V3.py
```

Evaluate the model:

```bash
python training/test_model.py
```

---

## Model Performance

Validation Accuracy:

```text
99.56%
```

Classes:

```text
none
hold
push
pull
tap
wave
```

Training and evaluation artifacts are available in:

```text
training/reports/
web/static/reports/
```

---

## License

This project was developed for academic and research purposes.

```
Aurasj
```
