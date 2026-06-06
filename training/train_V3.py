#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for the RadarSense gesture classifier.

Loads the recorded dataset, trains the TinyCNN model and
generates the files used for evaluation and live inference.
"""

import os
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "data"
OUT_DIR = "out_model"

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

T = 60          # time frames
R = 128         # range bins after resampling
SEED = 42

CLASSES = ["none", "hold", "push", "pull", "tap", "wave"]

MODEL_PT_PATH = os.path.join(OUT_DIR, "gesture_cnn_boss.pt")
MODEL_ONNX_PATH = os.path.join(OUT_DIR, "gesture_cnn_boss.onnx")
META_PATH = os.path.join(OUT_DIR, "gesture_cnn_meta.json")
REPORT_PATH = os.path.join(OUT_DIR, "classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(OUT_DIR, "confusion_matrix.png")
HISTORY_PATH = os.path.join(OUT_DIR, "training_history.json")
SPLIT_PATH = os.path.join(OUT_DIR, "dataset_split.json")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CPU deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# PREPROCESSING
# ============================================================

def fix_T(X: np.ndarray, target_T: int) -> np.ndarray:
    """Crop or pad signal to target number of frames."""
    if X.shape[0] == target_T:
        return X

    if X.shape[0] > target_T:
        return X[:target_T]

    pad = np.repeat(X[-1:], target_T - X.shape[0], axis=0)
    return np.concatenate([X, pad], axis=0)


def resample_range(X: np.ndarray, out_bins: int) -> np.ndarray:
    """Resample range dimension to fixed number of bins."""
    T_curr, R_curr = X.shape

    x_old = np.linspace(0.0, 1.0, R_curr, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, out_bins, dtype=np.float32)

    Y = np.empty((T_curr, out_bins), dtype=np.float32)

    for t in range(T_curr):
        Y[t] = np.interp(x_new, x_old, X[t].astype(np.float32))

    return Y


def load_npz_array(path: str) -> np.ndarray:
    """
    Load radar sample from .npz.
    Supports files saved as X=... or default arr_0.
    """
    data = np.load(path, allow_pickle=True)

    if "X" in data:
        X = data["X"]
    else:
        key = list(data.keys())[0]
        X = data[key]

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"Invalid sample shape in {path}: {X.shape}")

    return X


# ============================================================
# DATASET
# ============================================================

class RadarDataset(Dataset):
    def __init__(
        self,
        file_paths,
        labels,
        mean=None,
        std=None,
        augment: bool = False,
    ):
        self.file_paths = list(file_paths)
        self.labels = list(labels)
        self.augment = augment

        self.X_raw = []

        for path in self.file_paths:
            X = load_npz_array(path)
            self.X_raw.append(X)

        if mean is None or std is None:
            processed = []

            for X in self.X_raw:
                Xp = fix_T(X, T)
                Xp = resample_range(Xp, R)
                Xp = np.log1p(np.maximum(Xp, 0.0).astype(np.float32))
                processed.append(Xp)

            all_data = np.concatenate(processed, axis=0)
            self.mean = float(np.mean(all_data))
            self.std = float(np.std(all_data) + 1e-6)
        else:
            self.mean = float(mean)
            self.std = float(std)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        X = self.X_raw[idx]
        y = self.labels[idx]

        X = fix_T(X, T)
        X = resample_range(X, R)

        if self.augment:
            # Small temporal shift
            shift = np.random.randint(-4, 5)
            X = np.roll(X, shift, axis=0)

            # Small additive noise
            noise_scale = max(float(X.std()) * 0.05, 1e-6)
            noise = np.random.normal(0.0, noise_scale, X.shape).astype(np.float32)
            X = X + noise

            # Small amplitude scaling
            gain = np.random.uniform(0.90, 1.10)
            X = X * gain

        X = np.log1p(np.maximum(X, 0.0).astype(np.float32))
        X = (X - self.mean) / self.std

        # shape: [C, T, R]
        X = np.expand_dims(X, axis=0).astype(np.float32)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# ============================================================
# MODEL
# ============================================================

class TinyCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.f = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.h = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.h(self.f(x))


# ============================================================
# DATA LOADING
# ============================================================

def collect_dataset():
    file_paths = []
    labels = []
    class_counts = {c: 0 for c in CLASSES}

    for class_idx, class_name in enumerate(CLASSES):
        folder_path = os.path.join(DATA_DIR, class_name)

        if not os.path.isdir(folder_path):
            print(f"⚠️ Missing folder: {folder_path}")
            continue

        files = sorted(glob.glob(os.path.join(folder_path, "*.npz")))

        for f in files:
            file_paths.append(f)
            labels.append(class_idx)
            class_counts[class_name] += 1

    if len(file_paths) == 0:
        raise RuntimeError(f"No .npz files found in {DATA_DIR}/")

    missing = [c for c in CLASSES if class_counts[c] == 0]
    if missing:
        raise RuntimeError(f"Classes without samples: {missing}")

    return file_paths, labels, class_counts


def compute_class_weights(class_counts):
    total_samples = sum(class_counts.values())
    weights = []

    for c in CLASSES:
        count = class_counts[c]
        weight = total_samples / (len(CLASSES) * max(count, 1))
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


# ============================================================
# TRAIN / EVAL
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        out = model(batch_x)
        loss = criterion(out, batch_y)

        loss.backward()
        optimizer.step()

        preds = out.argmax(dim=1)

        total_loss += loss.item() * batch_x.size(0)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_x.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            out = model(batch_x)
            loss = criterion(out, batch_y)

            preds = out.argmax(dim=1)

            total_loss += loss.item() * batch_x.size(0)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_x.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())

    return (
        total_loss / total_samples,
        total_correct / total_samples,
        all_preds,
        all_targets,
    )


# ============================================================
# OUTPUTS
# ============================================================

def save_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def save_training_curves(history, out_dir):
    epochs = history["epoch"]

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=300)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], label="Train accuracy")
    ax.plot(epochs, history["val_acc"], label="Validation accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_curve.png"), dpi=300)
    plt.close(fig)


def export_onnx(model, device):
    model.eval()
    dummy_input = torch.randn(1, 1, T, R, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        MODEL_ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n🚀 train_v3.py — Radar Gesture CNN Training")
    print("================================================")

    set_seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    file_paths, labels, class_counts = collect_dataset()

    print("\n📊 Dataset samples per class:")
    for c in CLASSES:
        print(f"  {c:<5}: {class_counts[c]}")

    total_samples = len(file_paths)
    print(f"\n📦 Total samples: {total_samples}")

    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        file_paths,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED,
    )

    split_info = {
        "seed": SEED,
        "train_size": len(X_train_paths),
        "val_size": len(X_val_paths),
        "train_files": X_train_paths,
        "val_files": X_val_paths,
    }

    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print(f"✂️ Split: train={len(X_train_paths)}, val={len(X_val_paths)}")

    train_dataset = RadarDataset(X_train_paths, y_train, augment=True)
    mean, std = train_dataset.mean, train_dataset.std

    val_dataset = RadarDataset(X_val_paths, y_val, mean=mean, std=std, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"📉 Normalization: mean={mean:.6f}, std={std:.6f}")

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------
    model = TinyCNN(len(CLASSES)).to(device)

    class_weights = compute_class_weights(class_counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    print(f"⚖️ Class weights: {[round(float(w), 3) for w in class_weights.cpu()]}")

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    best_acc = 0.0
    best_epoch = 0
    best_preds = None
    best_targets = None

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    print("\n🔥 Training started...\n")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, val_preds, val_targets = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_acc)

        history["epoch"].append(epoch)
        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_preds = val_preds.copy()
            best_targets = val_targets.copy()

            pack = {
                "state_dict": model.state_dict(),
                "labels": CLASSES,
                "T": T,
                "R": R,
                "mean": mean,
                "std": std,
                "model_name": "TinyCNN",
                "version": "v3",
                "best_epoch": best_epoch,
                "best_val_acc": float(best_acc),
            }

            torch.save(pack, MODEL_PT_PATH)
            #export_onnx(model, device)

    # ------------------------------------------------------------
    # Final reports
    # ------------------------------------------------------------
    print("\n✅ Training complete")
    print(f"🏆 Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")

    if best_preds is None or best_targets is None:
        raise RuntimeError("No best predictions were saved. Training failed?")

    report = classification_report(
        best_targets,
        best_preds,
        target_names=CLASSES,
        digits=4,
    )

    cm = confusion_matrix(best_targets, best_preds)
    acc = accuracy_score(best_targets, best_preds)

    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Radar Gesture Classification Report\n")
        f.write("===================================\n\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation accuracy: {best_acc:.6f}\n")
        f.write(f"Accuracy score: {acc:.6f}\n\n")
        f.write(report)

    save_confusion_matrix(cm, CLASSES, CONFUSION_MATRIX_PATH)

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    save_training_curves(history, OUT_DIR)

    meta_info = {
        "labels": CLASSES,
        "T": T,
        "R": R,
        "mean": mean,
        "std": std,
        "preprocess": "fix_T + resample_range_to_R + log1p + z_score",
        "input_shape": [1, 1, T, R],
        "model_name": "TinyCNN",
        "version": "v3",
        "best_epoch": best_epoch,
        "best_val_acc": float(best_acc),
        "class_counts": class_counts,
        "train_size": len(X_train_paths),
        "val_size": len(X_val_paths),
        "seed": SEED,
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)

    print("\n📁 Saved files:")
    print(f"  - {MODEL_PT_PATH}")
    print(f"  - {MODEL_ONNX_PATH}")
    print(f"  - {META_PATH}")
    print(f"  - {REPORT_PATH}")
    print(f"  - {CONFUSION_MATRIX_PATH}")
    print(f"  - {os.path.join(OUT_DIR, 'loss_curve.png')}")
    print(f"  - {os.path.join(OUT_DIR, 'accuracy_curve.png')}")
    print(f"  - {HISTORY_PATH}")
    print(f"  - {SPLIT_PATH}")


if __name__ == "__main__":
    main()