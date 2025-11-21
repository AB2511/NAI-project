#!/usr/bin/env python3
"""
generate_publication_figures.py
Creates 6 publication-ready figures (PNG + SVG) into figures/final/
Expectations: see README or script header for required input files.
"""

import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

sns.set(style="whitegrid", rc={'figure.dpi':300})

ROOT = Path('.').resolve()
OUTDIR = ROOT / "figures" / "final"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Helper save
def save(fig, name):
    png = OUTDIR / f"{name}.png"
    svg = OUTDIR / f"{name}.svg"
    fig.savefig(png, bbox_inches='tight', dpi=300)
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", png, svg)

# 1) Grand-average ERP (target vs nontarget)
try:
    erp_t = np.load(ROOT / "data" / "processed_p3b" / "grand_erp_target.npy")
    erp_nt = np.load(ROOT / "data" / "processed_p3b" / "grand_erp_nontarget.npy")
    times = np.load(ROOT / "data" / "processed_p3b" / "time_vector.npy")
    # choose channel index for Pz/Cz (prefer central channel if available)
    ch_idx = 0  # if files are channels x times; adapt as needed
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(times, erp_t[ch_idx], label='Target', linewidth=2)
    ax.plot(times, erp_nt[ch_idx], label='Nontarget', linewidth=2)
    ax.axvspan(250, 500, color='grey', alpha=0.12)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Grand-average ERP (channel index {})".format(ch_idx))
    ax.legend()
    save(fig, "erp_grand_target_vs_nontarget")
except Exception as e:
    print("Skipping ERP figure — missing data or error:", e)

# 2) Group ERP with shading (mean ± sem across channels)
try:
    # compute mean across channels (or subjects) if shape allows
    mean_diff = (erp_t - erp_nt).mean(axis=0)
    sem_diff = (erp_t - erp_nt).std(axis=0) / np.sqrt(erp_t.shape[0])
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(times, mean_diff, label='Target - Nontarget', linewidth=2)
    ax.fill_between(times, mean_diff - sem_diff, mean_diff + sem_diff, alpha=0.25)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Amplitude (µV)")
    ax.set_title("Grand-average difference ERP (mean ± SEM)")
    save(fig, "erp_grand_diff_meansem")
except Exception as e:
    print("Skipping diff ERP figure —", e)

# 3) Topomap (300 ms) — expects 1D channel values saved as .npy
try:
    import mne
    topo_300 = np.load(ROOT / "results" / "topomaps" / "group_diff_300ms.npy")
    # topo_300 should be channel-values (len = n_channels)
    fig = mne.viz.plot_topomap(topo_300, pos=None, show=False)  # if pos montage is available replace pos=None
    # mne returns figure/axes objects; save via matplotlib
    fig_fig = fig[0]
    fig_fig.set_size_inches(6,4)
    save(fig_fig, "topomap_group_300ms")
except Exception as e:
    print("Skipping topomap 300ms —", e)

# 4) Topomap at peak latency (peakms)
try:
    topo_peak = np.load(ROOT / "results" / "topomaps" / "group_diff_peakms.npy")
    import mne
    fig = mne.viz.plot_topomap(topo_peak, pos=None, show=False)
    fig_fig = fig[0]
    fig_fig.set_size_inches(6,4)
    save(fig_fig, "topomap_group_peak")
except Exception as e:
    print("Skipping topomap peak —", e)

# 5) ROC curve for EEGNet (LOSO)
try:
    arr = np.load(ROOT / "results" / "eegnet_v5" / "all_fold_preds.npz")
    # expect 'y_true' and 'y_score' arrays
    y_true = arr['y_true']
    y_score = arr['y_score']
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, tpr, label=f'EEGNet (AUC = {roc_auc:.3f})', linewidth=2)
    ax.plot([0,1],[0,1], 'k--', linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (LOSO)")
    ax.legend(loc='lower right')
    save(fig, "roc_eegnet_loso")
except Exception as e:
    print("Skipping ROC figure —", e)

# 6) Confusion matrix (group)
try:
    # If preds per sample available:
    y_pred = arr['y_pred']  # optional
    from sklearn.metrics import ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", values_format='d')
    ax.set_title("Confusion matrix (group)")
    save(fig, "confusion_matrix_group")
except Exception as e:
    print("Skipping confusion matrix —", e)

print("Figure generation complete. Check:", OUTDIR)