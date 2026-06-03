import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# --- CONFIGURATIE ---
DATA_DIR = "data"
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
T = 60      # Cadre in timp
R = 128     # Bins de distanta (range)

# Fixam ordinea claselor ca sa fim siguri ca nu se mai incurca
CLASSES = ["none", "hold", "push", "pull", "tap", "wave"]

# --- PREPROCESARE & DATASET ---
def fix_T(X, target_T):
    if X.shape[0] == target_T: return X
    if X.shape[0] > target_T: return X[:target_T]
    pad = np.repeat(X[-1:], target_T - X.shape[0], axis=0)
    return np.concatenate([X, pad], axis=0)

def resample_range(X, out_bins):
    T_curr, R_curr = X.shape
    x_old = np.linspace(0, 1, R_curr)
    x_new = np.linspace(0, 1, out_bins)
    Y = np.empty((T_curr, out_bins), dtype=np.float32)
    for t in range(T_curr):
        Y[t] = np.interp(x_new, x_old, X[t])
    return Y

class RadarDataset(Dataset):
    def __init__(self, file_paths, labels, mean=None, std=None, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        
        # Incarcam tot in RAM pentru viteza (sunt fisiere mici)
        self.X_raw = []
        for path in self.file_paths:
            # Presupunem ca datele sunt in arr_0 sau prima cheie din npz
            data = np.load(path)
            key = list(data.keys())[0] 
            self.X_raw.append(data[key].astype(np.float32))
            
        # Calculam mean/std pe intreg datasetul daca nu sunt date
        if mean is None or std is None:
            all_data = np.concatenate(self.X_raw, axis=0)
            # Aplicam log1p inainte de a calcula mean/std
            all_data_log = np.log1p(np.maximum(all_data, 0))
            self.mean = float(np.mean(all_data_log))
            self.std = float(np.std(all_data_log)) + 1e-6
        else:
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        X = self.X_raw[idx]
        y = self.labels[idx]

        # Standardizare a formei
        X = fix_T(X, T)
        X = resample_range(X, R)
        
        # Augmentare de date (doar pe antrenament)
        if self.augment:
            # 1. Shift pe timp (rulare stanga/dreapta cu 1-4 cadre)
            shift = np.random.randint(-4, 5)
            X = np.roll(X, shift, axis=0)
            
            # 2. Adaugam putin zgomot aleatoriu (ajuta mult la date putine)
            noise = np.random.normal(0, X.std() * 0.05, X.shape)
            X = X + noise

        # Normalizare (la fel ca in V1)
        X = np.log1p(np.maximum(X, 0))
        X = (X - self.mean) / self.std

        # Adaugam dimensiunea de canal (1 canal de intrare)
        X = np.expand_dims(X, axis=0).astype(np.float32)
        return torch.tensor(X), torch.tensor(y, dtype=torch.long)

# --- MODELUL CNN ROBUST ---
class TinyCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.h = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4), # Am adaugat un Dropout serios ca sa previna overfitting-ul pe cele 50 de gesturi
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.h(self.f(x))

# --- BUCLE DE EXECUTIE ---
def main():
    print("🚀 Incepem treaba, Auras! Pregatim datele...")
    
    file_paths = []
    labels = []
    class_counts = {c: 0 for c in CLASSES}
    
    # Incarcam caile catre fisiere
    for class_idx, class_name in enumerate(CLASSES):
        folder_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(folder_path):
            print(f"⚠️ Lipseste folderul: {folder_path}")
            continue
            
        files = glob.glob(os.path.join(folder_path, "*.npz"))
        for f in files:
            file_paths.append(f)
            labels.append(class_idx)
            class_counts[class_name] += 1
            
    print(f"📊 Fisiere gasite per clasa: {class_counts}")

    # Impartire train / validare (80% / 20%)
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Cream dataseturile (augmentare DOAR pe train)
    train_dataset = RadarDataset(X_train_paths, y_train, augment=True)
    mean, std = train_dataset.mean, train_dataset.std 
    val_dataset = RadarDataset(X_val_paths, y_val, mean=mean, std=std, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"📉 Normalizare stabilita - Mean: {mean:.4f}, Std: {std:.4f}")

    # --- REZOLVAM DEZECHILIBRUL ---
    # Calculam "greutatea" fiecarei clase (ex: 'none' are pondere mica, restul au pondere mare)
    total_samples = len(labels)
    class_weights = []
    for c in CLASSES:
        count = class_counts[c]
        weight = total_samples / (len(CLASSES) * (count + 1e-6))
        class_weights.append(weight)
        
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"⚖️ Greutati aplicate pt dezechilibru: {[round(w,2) for w in class_weights]}")

    # Setare model si optimizator
    model = TinyCNN(len(CLASSES))
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # --- ANTRENAMENT ---
    print("\n🔥 Incepem antrenamentul...")
    best_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        correct_train = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            preds = out.argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()
            
        # Validare
        model.eval()
        val_loss = 0
        correct_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                out = model(batch_x)
                loss = criterion(out, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = out.argmax(dim=1)
                correct_val += (preds == batch_y).sum().item()
                
        t_loss = train_loss / len(train_dataset)
        v_loss = val_loss / len(val_dataset)
        t_acc = correct_train / len(train_dataset)
        v_acc = correct_val / len(val_dataset)
        
        print(f"Epoca {epoch:02d}/{EPOCHS} | Train Loss: {t_loss:.4f} Acc: {t_acc:.2f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.2f}")
        
        if v_acc > best_acc:
            best_acc = v_acc
            # Salvam in format .pt direct dict-ul complet
            pack = {
                "state_dict": model.state_dict(),
                "labels": CLASSES,
                "T": T,
                "R": R,
                "mean": mean,
                "std": std
            }
            torch.save(pack, "gesture_cnn_boss.pt")
            
            # Exportam si ONNX in caz ca ai nevoie
            dummy_input = torch.randn(1, 1, T, R)
            torch.onnx.export(model, dummy_input, "gesture_cnn_boss.onnx",
                              input_names=["input"], output_names=["output"])

    print(f"\n✅ Antrenament complet! Cea mai buna precizie pe validare: {best_acc:.2f}")
    
    # Salvam si fisierul META pentru live script
    meta_info = {
        "labels": CLASSES,
        "T": T,
        "R": R,
        "mean": mean,
        "std": std,
        "preprocess": "fix_T + resample_range_to_R + log1p + (x-mean)/std",
        "input_shape": [1, 1, T, R]
    }
    with open("gesture_cnn_meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)
    print("📁 Fisierele gesture_cnn_boss.pt, gesture_cnn_boss.onnx si gesture_cnn_meta.json au fost salvate.")

if __name__ == "__main__":
    main()