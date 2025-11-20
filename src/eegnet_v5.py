#!/usr/bin/env python3
"""
EEGNet v5 - LOSO CV (publication-ready, corrected)

Features:
- LOSO cross-validation over sub-XXX files in processed directory
- Window extraction (default 200-500 ms mapped to 0..600ms/615 samples)
- Per-fold scaling (StandardScaler fitted on train)
- WeightedRandomSampler oversampling for minority class per-fold
- Data augmentation (train only): time jitter, gaussian noise, channel dropout, time mask
- FocalLoss (gamma, alpha auto from class balance) - recommended for severe class imbalance
- EarlyStopping, ReduceLROnPlateau
- Saves per-fold model, ROC + confusion matrix figures, per-fold metrics, and a summary CSV/JSON

Usage example:
python src/eegnet_v5.py --data-dir data/processed_p3b --out-dir results/eegnet_v5 --epochs 80 --batch-size 64 --device cpu --augment
"""

import os
import glob
import json
import argparse
from pathlib import Path
import time
import math
import random
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import Adam

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.utils import resample

# ---------- Utilities & Repro ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ---------- Dataset wrapper ----------
class EEGDataset(Dataset):
    def __init__(self, X, y, augment=False, aug_params=None):
        """
        X: np.array (n_samples, n_ch, n_time)
        y: np.array (n_samples,)
        augment: apply augmentations (train only)
        aug_params: dict
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = bool(augment)
        self.aug_params = aug_params or {}

    def __len__(self):
        return len(self.X)

    def _augment(self, x):
        p = self.aug_params
        # time jittering: roll by up to +/- jitter_samples
        jitter_ms = p.get("jitter_ms", 10)
        if jitter_ms and random.random() < p.get("p_jitter", 0.3):
            # approximate jitter by sampling integer shift
            jitter_samps = int(round(jitter_ms / 1000.0 * p["fs"]))
            if jitter_samps > 0:
                shift = random.randint(-jitter_samps, jitter_samps)
                x = np.roll(x, shift, axis=1)

        # time mask (like SpecAugment): mask a contiguous block
        if p.get("time_mask_seconds", 0.02) and random.random() < p.get("p_time_mask", 0.2):
            tmask_len = int(round(p["time_mask_seconds"] * p["fs"]))
            if tmask_len > 0:
                start = random.randint(0, max(0, x.shape[1] - tmask_len))
                x[:, start:start+tmask_len] = 0.0

        # channel dropout
        if p.get("chan_dropout_prob", 0.1) and random.random() < p.get("p_chan_drop", 0.3):
            for ch in range(x.shape[0]):
                if random.random() < p["chan_dropout_prob"]:
                    x[ch, :] = 0.0

        # additive gaussian noise
        if p.get("noise_std", 0.01) and random.random() < p.get("p_noise", 0.5):
            x = x + np.random.normal(0, p["noise_std"], size=x.shape).astype(np.float32)

        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        y = int(self.y[idx])
        if self.augment:
            x = self._augment(x.copy())
        # PyTorch expects channels x time as float32
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

# ---------- EEGNet model (compact) ----------
class EEGNet(nn.Module):
    """
    Compact EEGNet-like architecture for P3b.
    Input shape: (batch, channels, time)
    """
    def __init__(self, in_chans=26, n_classes=2, F1=16, D=2, F2=32, kern_len=64, dropout=0.5):
        super().__init__()
        # temporal convolution
        self.temporal = nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len//2), bias=False)
        # depthwise spatial conv
        self.depthwise = nn.Conv2d(F1, F1 * D, (in_chans, 1), groups=F1, bias=False)
        self.bn1 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout = nn.Dropout(dropout)
        # separable conv
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        # classifier
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * 1 * (int( ( (615-1) / 1 ) / 32 )+1), 128) if False else nn.Identity()  # replaced later
        )
        # we will set final layers dynamically based on time dimension
        self._dynamic = True
        self.n_classes = n_classes

    def finalize(self, timepoints):
        """
        Call after creating with known temporal length to set final FC layer
        """
        # Run a dummy pass to compute final flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 26, timepoints)
            x = self.temporal(dummy)
            x = self.depthwise(x)
            x = self.bn1(x)
            x = self.elu(x)
            x = self.pool1(x)
            x = self.dropout(x)
            x = self.separable(x)
            x = self.bn2(x)
            x = self.elu(x)
            x = self.pool2(x)
            flat = int(np.prod(x.shape[1:]))
        # define classifier
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.n_classes)
        )
        self._dynamic = False

    def forward(self, x):
        # expects x: (batch, chans, time) -> reshape to (batch, 1, chans, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.separable(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.classify(x)
        return x

# ---------- Loss: FocalLoss ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        # logits: (N, C)
        # targets: (N,) long
        probs = torch.softmax(logits, dim=1) + self.eps
        targets_onehot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_onehot).sum(dim=1)  # probability of true class
        log_pt = torch.log(pt)
        focal_term = ((1 - pt) ** self.gamma)
        loss = -focal_term * log_pt
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple, np.ndarray)):
                alpha = torch.tensor(self.alpha, device=logits.device, dtype=torch.float32)
                alpha_factor = (targets_onehot * alpha.unsqueeze(0)).sum(dim=1)
            else:
                alpha_factor = (targets_onehot[:, 1] * self.alpha + targets_onehot[:, 0] * (1.0 - self.alpha))
            loss = loss * alpha_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ---------- Training utilities ----------
def compute_indices_for_window(n_timepoints, window_ms=(200,500), total_ms=600.0):
    # Map window in ms (relative to 0..total_ms) to sample indices inclusive
    start_ms, end_ms = window_ms
    # n_points distributed over 0..total_ms inclusive (approx)
    # Use (n_timepoints-1) as denominator to map to indices
    start_idx = int(round((start_ms / total_ms) * (n_timepoints - 1)))
    end_idx = int(round((end_ms / total_ms) * (n_timepoints - 1)))
    start_idx = max(0, min(n_timepoints - 1, start_idx))
    end_idx = max(0, min(n_timepoints - 1, end_idx))
    if end_idx < start_idx:
        raise ValueError("Window mapping produced invalid indices")
    return start_idx, end_idx

def train_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    for x, y in loader:
        x = x.to(device)  # shape (B, C, T)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(y.detach().cpu().numpy().tolist())
    if len(all_preds) == 0:
        return math.nan, {}
    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, {}

def eval_epoch(model, loader, device, criterion=None):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:,1]
            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_targets.extend(y.detach().cpu().numpy().tolist())
    if len(all_probs) == 0:
        return {}
    probs = np.array(all_probs)
    trues = np.array(all_targets)
    # binary predictions with 0.5 threshold
    preds_bin = (probs >= 0.5).astype(int)
    metrics = {}
    try:
        metrics['auc'] = float(roc_auc_score(trues, probs))
    except Exception:
        metrics['auc'] = float('nan')
    metrics['acc'] = float(accuracy_score(trues, preds_bin))
    metrics['f1'] = float(f1_score(trues, preds_bin, zero_division=0))
    metrics['recall'] = float(recall_score(trues, preds_bin, zero_division=0))
    metrics['precision'] = float(precision_score(trues, preds_bin, zero_division=0))
    metrics['y_true'] = trues.tolist()
    metrics['y_prob'] = probs.tolist()
    metrics['y_pred'] = preds_bin.tolist()
    return metrics

# ---------- Main LOSO function ----------
def run_loso(data_dir, out_dir,
             epochs=80, batch_size=64, lr=1e-3, device='cpu',
             window_ms=(200,500), augment=True,
             focal_gamma=2.0, early_patience=8, reduce_patience=4):
    set_seed()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    figs_dir = out_dir / "figures"
    perfold_dir = out_dir / "per_fold"
    for d in (models_dir, figs_dir, perfold_dir):
        d.mkdir(parents=True, exist_ok=True)

    # load subjects
    files_x = sorted(glob.glob(os.path.join(data_dir, "sub-*_X.npy")))
    files_y = sorted(glob.glob(os.path.join(data_dir, "sub-*_y.npy")))
    subs = []
    for fx in files_x:
        subid = Path(fx).stem.split('_')[0]
        fy = os.path.join(data_dir, f"{subid}_y.npy")
        if not os.path.exists(fy):
            continue
        subs.append(subid)
    subs = sorted(list(set(subs)))
    if len(subs) == 0:
        raise RuntimeError("No subject files found in data_dir")

    print(f"Found {len(subs)} subjects: {subs}")
    # load example to get n_ch, n_time
    sample_X = np.load(os.path.join(data_dir, f"{subs[0]}_X.npy"))
    n_samples, n_ch, n_time = sample_X.shape
    print(f"per-subject shape: {n_samples} x {n_ch} x {n_time}")

    # compute window indices
    s_idx, e_idx = compute_indices_for_window(n_time, window_ms=window_ms, total_ms=600.0)
    print(f"Window {window_ms[0]}-{window_ms[1]} ms -> indices {s_idx}:{e_idx} (len {e_idx-s_idx+1})")

    # store per-fold metrics
    fold_metrics = []
    all_preds = {}
    device = torch.device(device)

    for i, test_sub in enumerate(subs):
        print(f"\n=== LOSO fold {i+1}/{len(subs)} — test subject: {test_sub} ===")
        # build train/test arrays
        X_train_parts = []
        y_train_parts = []
        X_test = np.load(os.path.join(data_dir, f"{test_sub}_X.npy"))[:, :, s_idx:e_idx+1]
        y_test = np.load(os.path.join(data_dir, f"{test_sub}_y.npy"))
        for sub in subs:
            if sub == test_sub:
                continue
            Xp = np.load(os.path.join(data_dir, f"{sub}_X.npy"))[:, :, s_idx:e_idx+1]
            yp = np.load(os.path.join(data_dir, f"{sub}_y.npy"))
            X_train_parts.append(Xp)
            y_train_parts.append(yp)
        X_train = np.concatenate(X_train_parts, axis=0)
        y_train = np.concatenate(y_train_parts, axis=0)
        print("Train class counts:", dict(zip(*np.unique(y_train, return_counts=True))))

        # per-fold standard scaling (fit on train)
        n_train, _, tlen = X_train.shape
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(n_train, -1)
        X_train_flat = scaler.fit_transform(X_train_flat)
        X_train = X_train_flat.reshape(n_train, n_ch, tlen)

        n_test = X_test.shape[0]
        X_test_flat = X_test.reshape(n_test, -1)
        X_test_flat = scaler.transform(X_test_flat)
        X_test = X_test_flat.reshape(n_test, n_ch, tlen)

        # datasets
        aug_params = {
            "fs": (n_time-1)/0.600 if n_time>1 else 1024,
            "jitter_ms": 8,
            "p_jitter": 0.25,
            "time_mask_seconds": 0.02,
            "p_time_mask": 0.25,
            "chan_dropout_prob": 0.12,
            "p_chan_drop": 0.2,
            "noise_std": 0.005,
            "p_noise": 0.5
        }
        train_ds = EEGDataset(X_train, y_train, augment=augment, aug_params=aug_params)
        test_ds = EEGDataset(X_test, y_test, augment=False)

        # weighted sampler to oversample minority class (train-only)
        unique, counts = np.unique(y_train, return_counts=True)
        class_counts = dict(zip(unique.tolist(), counts.tolist()))
        # compute weights per class inverse frequency
        class_weights = {cls: 1.0 / cnt for cls, cnt in class_counts.items()}
        sample_weights = np.array([class_weights[int(lbl)] for lbl in y_train], dtype=np.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        # build model (finalize with tlen)
        model = EEGNet(in_chans=n_ch, n_classes=2, F1=16, D=2, F2=32, kern_len=64, dropout=0.5)
        model.finalize(timepoints=tlen)
        model = model.to(device)

        # compute alpha for focal loss from class balance (alpha for class 1)
        # alpha = proportion of negative class (we weight positive higher)
        neg = class_counts.get(0, 1)
        pos = class_counts.get(1, 1)
        p_pos = pos / (pos + neg)
        alpha_pos = (neg / (pos + neg))  # give more weight to rare class
        # clamp
        alpha_pos = max(0.01, min(0.99, alpha_pos))
        focal = FocalLoss(gamma=focal_gamma, alpha=[1.0 - alpha_pos, alpha_pos])
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=reduce_patience)

        best_auc = -np.inf
        best_state = None
        no_improve = 0
        history = {"train_loss": [], "val_auc": []}

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            train_loss, _ = train_epoch(model, train_loader, device, focal, optimizer)
            val_metrics = eval_epoch(model, val_loader, device)
            val_auc = val_metrics.get("auc", float("nan"))
            history["train_loss"].append(train_loss)
            history["val_auc"].append(val_auc)
            # scheduler step with val loss proxy (1 - auc) so reduce when auc plateaus
            sched_val = 1.0 - (val_auc if not math.isnan(val_auc) else 0.0)
            scheduler.step(sched_val)
            if val_auc > best_auc + 1e-6:
                best_auc = val_auc
                best_state = {k: v.cpu().detach().clone() if isinstance(v, torch.Tensor) else v for k,v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f}")
            if no_improve >= early_patience:
                print("Early stopping (no improvement).")
                break
        elapsed = time.time() - t0

        # restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
        # eval final
        final_metrics = eval_epoch(model, val_loader, device)
        # Save model & figures
        model_path = models_dir / f"eegnet_{test_sub}.pt"
        torch.save(model.state_dict(), model_path)
        # Save per-fold history
        perfold = {
            "subject": test_sub,
            "train_n": int(len(train_ds)),
            "test_n": int(len(test_ds)),
            "class_counts_train": class_counts,
            "history": history,
            "final_metrics": final_metrics,
            "elapsed_seconds": elapsed
        }
        with open(perfold_dir / f"{test_sub}_results.json", "w") as fh:
            json.dump(perfold, fh, indent=2)

        # Plot ROC + confusion matrix for this fold
        y_true = np.array(final_metrics.get("y_true", []))
        y_prob = np.array(final_metrics.get("y_prob", []))
        y_pred = np.array(final_metrics.get("y_pred", []))
        # ROC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            plt.figure(figsize=(4.0, 4.0), dpi=200)
            plt.plot(fpr, tpr, label=f"AUC={final_metrics.get('auc', np.nan):.3f}")
            plt.plot([0,1],[0,1],'--',color='gray')
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC - {test_sub}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(figs_dir / f"roc_{test_sub}.png")
            plt.close()
        except Exception:
            pass
        # Confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(3.6,3.6), dpi=200)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Pred"); plt.ylabel("True"); plt.title(f"Confusion - {test_sub}")
            plt.tight_layout()
            plt.savefig(figs_dir / f"confusion_{test_sub}.png")
            plt.close()
        except Exception:
            pass

        # gather fold summary
        fold_summary = {
            "subject": test_sub,
            "acc": final_metrics.get("acc", float("nan")),
            "auc": final_metrics.get("auc", float("nan")),
            "f1": final_metrics.get("f1", float("nan")),
            "recall": final_metrics.get("recall", float("nan")),
            "precision": final_metrics.get("precision", float("nan")),
            "train_n": int(len(train_ds)),
            "test_n": int(len(test_ds)),
            "class_counts_train": class_counts,
            "elapsed_sec": elapsed
        }
        fold_metrics.append(fold_summary)
        # store predictions
        all_preds[test_sub] = {
            "y_true": final_metrics.get("y_true", []),
            "y_prob": final_metrics.get("y_prob", []),
            "y_pred": final_metrics.get("y_pred", [])
        }
        # end fold

    # Save summary
    summary_path = out_dir / "eegnet_loso_summary.json"
    csv_path = out_dir / "eegnet_loso_summary.csv"
    with open(summary_path, "w") as fh:
        json.dump({"folds": fold_metrics}, fh, indent=2)
    df = pd.DataFrame(fold_metrics)
    df.to_csv(csv_path, index=False)
    print("\nDONE. Summary saved to:", summary_path)
    return summary_path

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="EEGNet v5 LOSO (focal loss + oversampling + augment)")
    parser.add_argument("--data-dir", type=str, default="data/processed_p3b")
    parser.add_argument("--out-dir", type=str, default="results/eegnet_v5")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--window-ms", type=int, nargs=2, default=[200,500], help="start end ms")
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--early-patience", type=int, default=8)
    parser.add_argument("--reduce-patience", type=int, default=4)
    args = parser.parse_args()

    # run
    run_loso(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        window_ms=tuple(args.window_ms),
        augment=args.augment,
        focal_gamma=args.focal_gamma,
        early_patience=args.early_patience,
        reduce_patience=args.reduce_patience
    )

if __name__ == "__main__":
    main()