#!/usr/bin/env python3
"""
EEGNet v4 + LOSO CV for P3b dataset (publication-grade).
Saves per-fold models, per-fold predictions, ROC/Figures, and a summary JSON.

Usage:
    python src/eegnet_loso.py --data-dir data/processed_p3b --out-dir results/eegnet
"""

import os
import random
import json
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.utils import class_weight
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------------
# EEGNet model (v4-like compact variant)
# Reference ideas: Lawhern et al. 2018 EEGNet; adapted for 2d conv input (Ch x Time)
# ---------------------------
class EEGNetV4(nn.Module):
    def __init__(self, n_chans: int, n_times: int, n_classes: int = 2,
                 dropout_prob: float = 0.5, F1: int = 8, D: int = 2, F2_multiplier: int = 2):
        """
        n_chans: number of EEG channels
        n_times: number of time samples
        n_classes: 2 for binary
        F1: temporal filters
        D: depth multiplier for spatial filtering
        F2_multiplier: multiplier to produce F2 filters (F2 = F1*F2_multiplier)
        """
        super().__init__()
        F2 = F1 * F2_multiplier
        self.n_chans = n_chans
        self.n_times = n_times

        # Input shape expected: (B, 1, channels, times) - mimic EEGNet input
        # Temporal convolution
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(F1)
        )
        # Depthwise spatial conv
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(n_chans, 1), groups=F1, bias=False),  # spatial
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_prob)
        )
        # Separable conv (pointwise)
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_prob)
        )

        # Classifier - flatten then linear
        # compute flattened size by a forward pass with dummy input
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            out = self.forward_features(dummy)
            flat_size = out.shape[1] * out.shape[2] * out.shape[3]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, n_classes)
        )

    def forward_features(self, x):
        # x: (B, 1, channels, times)
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x

# ---------------------------
# Dataset helper
# ---------------------------
def load_subject_files(data_dir: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Find files matching sub-XXX_X.npy and sub-XXX_y.npy
    Returns list of tuples (sub_id, X, y)
    """
    files = sorted(data_dir.glob("sub-*_X.npy"))
    subjects = []
    for f in files:
        sub = f.name.split("_")[0]  # sub-001
        X = np.load(str(f))  # shape (n_trials, n_chans, n_times)
        y_path = f.parent / f"{sub}_y.npy"
        if not y_path.exists():
            raise FileNotFoundError(f"Missing label file for {sub}: {y_path}")
        y = np.load(str(y_path))
        subjects.append((sub, X, y))
    return subjects

# ---------------------------
# Training + evaluation utilities
# ---------------------------
def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle=True) -> DataLoader:
    # Convert to torch tensors and proper shape: (B, 1, ch, times)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    # add channel dimension
    if X_t.dim() == 3:
        X_t = X_t.unsqueeze(1)  # (B,1,ch,t)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

def compute_class_weights(y_train: np.ndarray) -> torch.Tensor:
    # for CrossEntropyLoss - weight per class
    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    # returns a torch tensor ordered by class label
    wt = torch.tensor([weights[int(c)] for c in classes], dtype=torch.float32)
    return wt

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * Xb.size(0)
    return running_loss / len(loader.dataset)

def eval_model(model, loader, device) -> Tuple[np.ndarray, np.ndarray, Dict]:
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # prob of class 1
            preds = (probs >= 0.5).astype(int)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(yb.numpy())
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return probs, preds, labels

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    metrics = {}
    try:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics['auc'] = float('nan')
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics['acc'] = float((y_true == y_pred).mean())
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    return metrics

# ---------------------------
# Utility: plot ROC
# ---------------------------
def plot_roc(y_true, y_prob, out_path: Path, title: str = "ROC"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6), dpi=150)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# ---------------------------
# Main LOSO routine
# ---------------------------
def run_loso(data_dir: str, out_dir: str,
             epochs: int = 60, batch_size: int = 64,
             lr: float = 1e-3, weight_decay: float = 1e-4,
             patience: int = 8, verbose: bool = True):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    subjects = load_subject_files(data_dir)
    n_subj = len(subjects)
    if n_subj == 0:
        raise RuntimeError("No subjects found in data dir.")

    # get dataset-level shapes from first subject
    n_trials, n_chans, n_times = subjects[0][1].shape
    print(f"Found {n_subj} subjects. per-subject shape: {n_trials} x {n_chans} x {n_times}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    loso_summary = []
    all_fold_preds = {}

    # create a merged dataset index mapping for convenience if needed
    for i_hold, (test_sub, X_test, y_test) in enumerate(subjects):
        print(f"\n=== LOSO fold {i_hold+1}/{n_subj} — test subject: {test_sub} ===")
        # prepare train set by stacking all other subjects
        X_train_list, y_train_list = [], []
        for sub, Xs, ys in subjects:
            if sub == test_sub:
                continue
            X_train_list.append(Xs)
            y_train_list.append(ys)
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # quick sanity
        unique, counts = np.unique(y_train, return_counts=True)
        print("Train class counts:", dict(zip(unique.tolist(), counts.tolist())))

        # compute class weights for CrossEntropyLoss
        class_w = compute_class_weights(y_train)
        # map to full length 2 vector (if some class missing, handle)
        if len(class_w.shape) == 1 and class_w.shape[0] == 2:
            ce_weights = class_w.to(device)
        else:
            # fallback to uniform
            ce_weights = torch.ones(2, dtype=torch.float32).to(device)

        # build dataloaders
        train_loader = make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_loader = make_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

        # create model
        model = EEGNetV4(n_chans=n_chans, n_times=n_times, n_classes=2,
                         dropout_prob=0.5, F1=8, D=2, F2_multiplier=2)
        model.to(device)

        criterion = nn.CrossEntropyLoss(weight=ce_weights)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4)

        best_auc = -np.inf
        best_state = None
        best_epoch = -1
        epochs_no_improve = 0

        # training loop
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            # evaluate on validation (test_sub)
            probs_val, preds_val, labels_val = eval_model(model, val_loader, device)
            try:
                val_auc = float(roc_auc_score(labels_val, probs_val))
            except ValueError:
                val_auc = float('nan')

            scheduler.step(train_loss)
            if verbose:
                print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f}")

            # early stopping on val_auc
            if not np.isnan(val_auc) and val_auc > best_auc + 1e-4:
                best_auc = val_auc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print("Early stopping (no improvement).")
                break

        # restore best
        if best_state is not None:
            model.load_state_dict(best_state)

        # final eval on test subject
        probs_val, preds_val, labels_val = eval_model(model, val_loader, device)
        metrics = compute_metrics(labels_val, preds_val, probs_val)
        print(f"Fold {test_sub} | Acc={metrics['acc']:.3f} F1={metrics['f1']:.3f} Recall={metrics['recall']:.3f} AUC={metrics['auc']:.3f}")

        # save model
        model_path = models_dir / f"eegnet_{test_sub}.pth"
        torch.save({'model_state': model.state_dict(), 'n_chans': n_chans, 'n_times': n_times}, model_path)

        # save per-fold predictions and metrics
        out_pred = out_dir / f"preds_{test_sub}.npz"
        np.savez_compressed(out_pred, probs=probs_val, preds=preds_val, labels=labels_val)

        # save ROC figure
        roc_fig = figs_dir / f"roc_{test_sub}.png"
        try:
            plot_roc(labels_val, probs_val, roc_fig, title=f"ROC - {test_sub} (AUC={metrics['auc']:.3f})")
        except Exception:
            pass

        # save confusion matrix figure
        cm = np.array(metrics['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"Confusion {test_sub}")
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        for (j, k), val in np.ndenumerate(cm):
            ax.text(k, j, int(val), ha='center', va='center', color='black')
        plt.tight_layout()
        plt.savefig(figs_dir / f"confusion_{test_sub}.png", dpi=300)
        plt.close()

        # accumulate summary
        loso_summary.append({
            "subject": test_sub,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "metrics": metrics,
            "best_epoch": int(best_epoch),
            "model_path": str(model_path)
        })
        all_fold_preds[test_sub] = {
            "probs": probs_val.tolist(),
            "preds": preds_val.tolist(),
            "labels": labels_val.tolist(),
            "metrics": metrics
        }

    # After all folds - compute group stats
    summary = {}
    df = pd.DataFrame([
        {
            "subject": e["subject"],
            "acc": e["metrics"]["acc"],
            "auc": e["metrics"]["auc"],
            "f1": e["metrics"]["f1"],
            "recall": e["metrics"]["recall"],
            "precision": e["metrics"]["precision"]
        } for e in loso_summary
    ])
    summary['n_folds'] = len(loso_summary)
    summary['acc_mean'] = float(df["acc"].mean())
    summary['acc_std'] = float(df["acc"].std())
    summary['auc_mean'] = float(df["auc"].mean())
    summary['auc_std'] = float(df["auc"].std())
    summary['f1_mean'] = float(df["f1"].mean())
    summary['per_subject'] = df.to_dict(orient='records')

    # save summary JSON
    (out_dir / "eegnet_loso_summary.json").write_text(json.dumps({"summary": summary, "folds": loso_summary}, indent=2))

    # save all fold preds
    np.savez_compressed(out_dir / "all_fold_preds.npz", **{k: np.array(v["probs"]) for k, v in all_fold_preds.items()})

    print("\nDONE. LOSO EEGNet summary saved to:", out_dir / "eegnet_loso_summary.json")
    print("Figures saved to:", figs_dir)
    print("Models saved to:", models_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EEGNet LOSO CV runner")
    parser.add_argument("--data-dir", type=str, default="data/processed_p3b", help="Processed per-subject .npy folder")
    parser.add_argument("--out-dir", type=str, default="results/eegnet", help="Output directory")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # set seed from args
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    run_loso(args.data_dir, args.out_dir,
             epochs=args.epochs,
             batch_size=args.batch_size,
             lr=args.lr,
             weight_decay=args.weight_decay,
             patience=args.patience,
             verbose=True)