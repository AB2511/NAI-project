# src/inference/train_cnn_classifier.py
"""
Train a 1D-CNN classifier for P300 cognitive states.
Saves:
 - models/p300_cnn_pipeline.pt (torchscript)
 - models/p300_cnn_stateful.pth (checkpoint)
 - models/p300_cnn_label_encoder.joblib
"""

import os
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -----------------------
# Config / Hyperparams
# -----------------------
DEFAULTS = {
    "window_s": 1.5,           # seconds for window
    "fs": 1000.0,              # nominal sample rate to convert ms to samples if needed (we use event-based P300)
    "batch_size": 64,
    "epochs": 60,
    "lr": 1e-3,
    "patience": 8,
    "val_split": 0.15,
    "test_split": 0.10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "export_torchscript": True,
}

# -----------------------
# Simple Dataset: sliding window over timestamps
# -----------------------
class P300WindowDataset(Dataset):
    def __init__(self, p300_df, state_df, window_s=1.5, overlap=0.0, fs=1000.0, label_encoder=None):
        """
        p300_df: DataFrame with columns ['ts', 'amplitude_uv', 'latency_ms', ...]
        state_df: DataFrame with ['ts', 'state', 'confidence', ...]
        We'll align windows around state timestamps (or use sliding windows).
        """
        self.window_s = float(window_s)
        self.half_win = self.window_s / 2.0
        self.fs = fs
        # convert to numpy arrays for speed
        self.p300 = p300_df.copy()
        self.state = state_df.copy()
        # ensure numeric ts
        self.p300['ts'] = self.p300['ts'].astype(float)
        self.state['ts'] = self.state['ts'].astype(float)

        # choose windows centered on state events (common approach)
        records = []
        for _, row in self.state.iterrows():
            ts = float(row['ts'])
            # gather p300 samples within [ts - half_win, ts + half_win]
            s = ts - self.half_win
            e = ts + self.half_win
            window = self.p300[(self.p300['ts'] >= s) & (self.p300['ts'] <= e)]
            # we use amplitude_uv and latency_ms as channels/features per sample
            # If window empty, skip
            if window.shape[0] < 3:
                continue
            # We resample by taking amplitude series and padding/truncating to fixed length L (use N = fixed samples)
            # We'll pack amplitude + latency as two channels, using simple interpolation to fixed length
            records.append({
                'center_ts': ts,
                'state': row['state'],
                'confidence': float(row.get('confidence', 1.0)),
                'p300_window': window[['amplitude_uv', 'latency_ms']].values
            })
        self.records = records
        if label_encoder is None:
            self.le = LabelEncoder()
            self.le.fit([r['state'] for r in self.records])
        else:
            self.le = label_encoder

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        arr = rec['p300_window']  # shape (n_samples, 2)
        # we will resample/interpolate to fixed length L (choose L=64)
        L = 64
        # linear interpolation along axis 0 for each channel
        n = arr.shape[0]
        xp = np.linspace(0, 1, n)
        xq = np.linspace(0, 1, L)
        arr_interp = np.stack([np.interp(xq, xp, arr[:, c]) for c in range(arr.shape[1])], axis=0)
        # arr_interp shape (channels, L)
        label = self.le.transform([rec['state']])[0]
        return torch.tensor(arr_interp, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# -----------------------
# Model: small 1D CNN
# -----------------------
class P300_CNN(nn.Module):
    def __init__(self, n_channels=2, n_classes=4, base_filters=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, base_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base_filters, base_filters*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_filters*2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters*4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_filters*4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x)

# -----------------------
# Utilities
# -----------------------
def load_csvs(p300_csv, state_csv):
    p300 = pd.read_csv(p300_csv)
    state = pd.read_csv(state_csv)
    # ensure expected columns
    # unify names: amplitude_uv, latency_ms, ts  ; state: ts, state, confidence
    # if original uses amplitude rather than amplitude_uv, adapt
    for df, expected in [(p300, ['ts','amplitude_uv','latency_ms']), (state, ['ts','state','confidence'])]:
        for col in expected:
            if col not in df.columns:
                raise ValueError(f"Missing col {col} in {('p300' if df is p300 else 'state')}")
    return p300, state

def train(args):
    p300, state = load_csvs(args.p300_csv, args.state_csv)
    # Optional confidence filter
    if args.confidence_thresh:
        state = state[state['confidence'] >= args.confidence_thresh]
    # build dataset
    ds = P300WindowDataset(p300, state, window_s=args.window_s)
    le = ds.le
    # train/test split
    idxs = np.arange(len(ds))
    train_idx, test_idx = train_test_split(idxs, test_size=args.test_split, random_state=42, stratify=[ds.records[i]['state'] for i in idxs])
    train_idx, val_idx = train_test_split(train_idx, test_size=args.val_split/(1-args.test_split), random_state=42, stratify=[ds.records[i]['state'] for i in train_idx])

    def subset(ds, idxs):
        class SubsetDS(torch.utils.data.Dataset):
            def __init__(self, ds, idxs): self.ds, self.idxs = ds, idxs
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i): return self.ds[self.idxs[i]]
        return SubsetDS(ds, idxs)

    train_ds = subset(ds, train_idx)
    val_ds   = subset(ds, val_idx)
    test_ds  = subset(ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # model
    n_classes = len(le.classes_)
    model = P300_CNN(n_channels=2, n_classes=n_classes, base_filters=args.base_filters, dropout=args.dropout).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=4)

    best_val = -1.0
    best_path = Path(args.out_dir) / "p300_cnn_stateful.pth"
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        for X, y in train_loader:
            X = X.to(args.device)
            y = y.to(args.device)
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
        train_acc = correct / total
        train_loss = running_loss/total

        # validation
        model.eval()
        vloss, vtotal, vcorrect = 0.0, 0, 0
        all_y, all_p = [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(args.device)
                yv = yv.to(args.device)
                logits = model(Xv)
                loss = criterion(logits, yv)
                vloss += loss.item() * Xv.size(0)
                preds = logits.argmax(dim=1)
                vcorrect += (preds == yv).sum().item()
                vtotal += Xv.size(0)
                all_y.append(yv.cpu().numpy())
                all_p.append(logits.softmax(dim=1).cpu().numpy())
        val_acc = vcorrect / vtotal if vtotal>0 else 0.0
        val_loss = vloss / vtotal if vtotal>0 else 0.0
        scheduler.step(val_loss)

        print(f"[{epoch}/{args.epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        # checkpoint if improved
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'le_classes': list(le.classes_)}, best_path)
            # export TorchScript for inference
            if args.export_torchscript:
                model_cpu = P300_CNN(n_channels=2, n_classes=n_classes, base_filters=args.base_filters, dropout=args.dropout)
                checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
                model_cpu.load_state_dict(checkpoint['model_state'])
                model_cpu.eval()
                example = torch.randn(1, 2, 64)
                traced = torch.jit.trace(model_cpu, example)
                traced.save(os.path.join(args.out_dir, "p300_cnn_pipeline.pt"))
            joblib.dump(le, os.path.join(args.out_dir, "p300_cnn_label_encoder.joblib"))

    # final evaluation on test
    model.load_state_dict(torch.load(best_path, weights_only=False)['model_state'])
    model.to(args.device)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(args.device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    print("Test Accuracy:", acc)
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print("Saved model to", best_path)
    return

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--p300-csv", required=True)
    p.add_argument("--state-csv", required=True)
    p.add_argument("--out-dir", default="models")
    p.add_argument("--window-s", type=float, default=DEFAULTS["window_s"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--device", default=DEFAULTS["device"])
    p.add_argument("--export-torchscript", action="store_true")
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--val-split", type=float, default=DEFAULTS["val_split"])
    p.add_argument("--test-split", type=float, default=DEFAULTS["test_split"])
    p.add_argument("--confidence-thresh", type=float, default=None)
    args = p.parse_args()
    train(args)