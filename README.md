# RadarSense

Real-time radar gesture recognition system built around the **SparkFun Acconeer A111 Pulsed Coherent Radar Sensor** and a custom **TinyCNN-based inference pipeline**.

RadarSense combines radar signal acquisition, machine learning inference, gesture stabilization, Raspberry Pi deployment, and a live web dashboard into a complete end-to-end gesture recognition system.

---

## Overview

RadarSense enables real-time hand gesture recognition using radar data instead of cameras.

The system acquires radar frames from an Acconeer A111 sensor connected to a Raspberry Pi, processes them through a custom machine learning pipeline, and streams predictions to a web dashboard with live visualizations and event monitoring.

---

## Highlights

* Real-time radar gesture recognition
* Custom TinyCNN inference engine
* FSM-based gesture stabilization
* Raspberry Pi deployment
* Live dashboard built with Flask and Socket.IO
* Radar waveform visualization
* Waterfall / heatmap visualization
* Session statistics and event logging
* UDP event broadcasting for external applications
* Training and evaluation pipeline
* Model performance reporting

---

## Supported Gestures

The current model recognizes six gesture classes:

| Class |
| ----- |
| none  |
| hold  |
| push  |
| pull  |
| tap   |
| wave  |

---

## System Architecture

```text
SparkFun Acconeer A111 Radar
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
 Dashboard     UDP Events
   (Web UI)    (Unity / External Apps)
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

* SparkFun Acconeer A111 Pulsed Coherent Radar Sensor
* Raspberry Pi 4
* Windows/Linux host computer

---

## Installation

Runtime dependencies:

```bash
pip install -r requirements.txt
```

Training dependencies:

```bash
pip install -r requirements_train.txt
```

---

## Running the System

Start the RadarSense server:

```bash
python app/server.py
```

Open the dashboard:

```text
http://localhost:5002
```

---

## UDP Event Monitor

Run the gesture event monitor in a separate terminal:

```bash
python app/gesture_monitor.py
```

---

## Dataset & Training

Record radar samples:

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

Training artifacts are stored in:

```text
training/reports/
```

Dashboard report assets are stored in:

```text
web/static/reports/
```

---

## Model Performance

Validation Accuracy:

```text
99.56%
```

Recognized Classes:

```text
none
hold
push
pull
tap
wave
```

The dashboard includes:

* Classification report
* Confusion matrix
* Accuracy curve
* Loss curve


## License

This project was developed for academic and research purposes.
