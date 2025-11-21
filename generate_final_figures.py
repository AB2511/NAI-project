# ============================================================
# FINAL NAI PROJECT FIGURE + METRICS GENERATOR (REAL DATA ONLY)
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

DATA_DIR = "data/processed_p3b"
LOSO_DIR = "results/eegnet_v5"
FIG_OUT = "figures/final"

os.makedirs(FIG_OUT, exist_ok=True)

print("====== LOADING REAL LOSO RESULTS ======")

# ---- 1. Load per-fold predictions ----
pred_file = os.path.join(LOSO_DIR, "all_fold_preds.npz")
pred_npz = np.load(pred_file, allow_pickle=True)

fold_preds = pred_npz["y_pred"]
fold_labels = pred_npz["y_true"]
fold_probs = pred_npz["y_score"]

print("Loaded LOSO predictions:", len(fold_preds), "folds")

# ======================================
# FIGURE 1 — ROC CURVE (REAL)
# ======================================
fpr, tpr, _ = roc_curve(fold_labels, fold_probs)
auc_val = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_val:.3f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — EEGNet LOSO (REAL DATA)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(FIG_OUT, "roc_curve.png"))
plt.close()

print("Saved: roc_curve.png")

# ======================================
# FIGURE 2 — CONFUSION MATRIX (REAL)
# ======================================
cm = confusion_matrix(fold_labels, fold_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — EEGNet LOSO (REAL DATA)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_OUT, "confusion_matrix.png"))
plt.close()

print("Saved: confusion_matrix.png")

# ======================================
# FIGURE 3 — Grand average ERP (Pz)
# ======================================
# Load time vector and metadata
t = np.load(os.path.join(DATA_DIR, "time_vector.npy"))
with open(os.path.join(DATA_DIR, "metadata.json"), "r") as f:
    meta = json.load(f)

# Exact 26 channel names after preprocessing (dropped FP1, FP2, F7, F8, EOG channels)
ch_names = ['F3', 'FC3', 'C3', 'C5', 'P3', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Oz', 'Pz', 'CPz', 
           'Fz', 'F4', 'FC4', 'FCz', 'Cz', 'C4', 'C6', 'P4', 'P8', 'P10', 'PO8', 'PO4', 'O2']
pz_idx = ch_names.index("Pz")  # Should be index 11

all_target = []
all_standard = []

for i in range(1,21):
    X = np.load(os.path.join(DATA_DIR, f"sub-{i:03d}_X.npy"))
    y = np.load(os.path.join(DATA_DIR, f"sub-{i:03d}_y.npy"))
    
    all_target.append(X[y==1][:, pz_idx, :])
    all_standard.append(X[y==0][:, pz_idx, :])

ERP_target = np.mean(np.vstack(all_target), axis=0)
ERP_standard = np.mean(np.vstack(all_standard), axis=0)

plt.figure(figsize=(7,4))
plt.plot(t, ERP_target, label="Target", c="red")
plt.plot(t, ERP_standard, label="Standard", c="blue")
plt.axvline(0, c="black", lw=1)
plt.title("Grand-Average ERP at Pz")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(FIG_OUT, "grand_average_pz.png"))
plt.close()

print("Saved: grand_average_pz.png")

# ======================================
# FIGURE 4 — Group Topomap
# ======================================
from mne.viz import plot_topomap
import mne

# Build info using exact 26 channels that remain after preprocessing
chan_file = "data/raw_p3b/sub-001/eeg/sub-001_task-P3_channels.tsv"
elect = "data/raw_p3b/sub-001/eeg/sub-001_task-P3_electrodes.tsv"

import pandas as pd
ch_df = pd.read_csv(chan_file, sep="\t")
el_df = pd.read_csv(elect, sep="\t")

# Filter to only the 26 channels that remain in processed data
remaining_ch = ch_df[ch_df.name.isin(ch_names)]
remaining_el = el_df[el_df.name.isin(ch_names)]

# Ensure order matches our ch_names list
remaining_el = remaining_el.set_index('name').reindex(ch_names).reset_index()

coords = np.vstack([remaining_el.x, remaining_el.y, remaining_el.z]).T

info = mne.create_info(ch_names, sfreq=256, ch_types="eeg")
info.set_montage(mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, coords))))

# average topography at ~400 ms
t_idx = np.argmin(np.abs(t - 0.400))

all_vals = []
for i in range(1,21):
    X = np.load(os.path.join(DATA_DIR, f"sub-{i:03d}_X.npy"))
    y = np.load(os.path.join(DATA_DIR, f"sub-{i:03d}_y.npy"))
    avg = X[y==1].mean(axis=0)
    all_vals.append(avg[:, t_idx])

topo_val = np.mean(np.vstack(all_vals), axis=0)

plt.figure(figsize=(5,5))
plot_topomap(topo_val, info, cmap="RdBu_r")
plt.title("Group-Level Topomap (400 ms)")
plt.savefig(os.path.join(FIG_OUT, "group_topomap_400ms.png"))
plt.close()

print("Saved: group_topomap_400ms.png")

# ======================================
# FIGURE 5 — LOSO AUC vs Subject
# ======================================
summary_path = os.path.join(LOSO_DIR, "eegnet_loso_summary.json")
with open(summary_path, "r") as f:
    summ = json.load(f)

subs = [int(fold["subject"].replace("sub-", "")) for fold in summ["folds"]]
vals = [fold["auc"] for fold in summ["folds"]]

plt.figure(figsize=(7,4))
plt.bar(subs, vals)
plt.xlabel("Subject")
plt.ylabel("AUC")
plt.title("LOSO AUC per Subject — REAL EEGNet")
plt.tight_layout()
plt.savefig(os.path.join(FIG_OUT, "loso_auc_per_subject.png"))
plt.close()

print("Saved: loso_auc_per_subject.png")

# ======================================
# FIGURE 6 — P3 Amplitude vs AUC
# ======================================
p3_amp = []
for i in range(1,21):
    X = np.load(os.path.join(DATA_DIR, f"sub-{i:03d}_X.npy"))
    y = np.load(os.path.join(DATA_DIR, f"sub-{i:03d}_y.npy"))
    
    erp_t = X[y==1][:, pz_idx, :].mean(axis=0)
    p3_amp.append(erp_t[np.argmax(erp_t)])

plt.figure(figsize=(6,5))
plt.scatter(p3_amp, vals)
plt.xlabel("P3 Amplitude (µV)")
plt.ylabel("LOSO AUC")
plt.title("Brain–Behavior Correlation")
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(FIG_OUT, "p3_amplitude_vs_auc.png"))
plt.close()

print("Saved: p3_amplitude_vs_auc.png")

print("\n===== ALL 6 PUBLICATION FIGURES GENERATED SUCCESSFULLY =====")