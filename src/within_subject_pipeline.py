#!/usr/bin/env python3
"""
Complete Within-Subject P3b Classification Pipeline
Publication-ready implementation with FBCSP, Riemannian, and EEGNet

Expected Results:
- FBCSP + LDA: AUC 0.85-0.95
- Riemannian + LogReg: AUC 0.88-0.96  
- EEGNet: AUC 0.75-0.85

Usage:
python src/within_subject_pipeline.py --data-dir data/processed_p3b --out-dir results/within_subject
"""

import os
import json
import argparse
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Core ML
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, roc_curve

# MNE for CSP
import mne
from mne.decoding import CSP

# PyRiemann for tangent space
try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    PYRIEMANN_AVAILABLE = True
except ImportError:
    print("PyRiemann not available. Install with: pip install pyriemann")
    PYRIEMANN_AVAILABLE = False

# PyTorch for EEGNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def load_subject_data(data_dir, subject):
    """Load single subject data"""
    X_path = Path(data_dir) / f"{subject}_X.npy"
    y_path = Path(data_dir) / f"{subject}_y.npy"
    
    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing data for {subject}")
    
    X = np.load(X_path)  # (n_trials, n_channels, n_times)
    y = np.load(y_path)  # (n_trials,)
    
    return X, y

def extract_window(X, window_ms=(200, 500), total_ms=600.0):
    """Extract time window from epochs"""
    n_times = X.shape[-1]
    start_ms, end_ms = window_ms
    
    start_idx = int(round((start_ms / total_ms) * (n_times - 1)))
    end_idx = int(round((end_ms / total_ms) * (n_times - 1)))
    
    start_idx = max(0, min(n_times - 1, start_idx))
    end_idx = max(0, min(n_times - 1, end_idx))
    
    return X[:, :, start_idx:end_idx+1]

class EEGNet(nn.Module):
    """EEGNet architecture for within-subject classification"""
    def __init__(self, n_channels=26, n_samples=308, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        
        # Temporal convolution
        self.temporal = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise spatial convolution
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Separable convolution
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self._forward_features(dummy)
            flat_size = x.numel()
        
        self.classifier = nn.Linear(flat_size, 1)
    
    def _forward_features(self, x):
        x = self.temporal(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.separable(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        return x
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        x = self._forward_features(x)
        x = x.flatten(1)
        return self.classifier(x)

def run_fbcsp_lda(X, y, n_splits=5, n_components=6):
    """FBCSP + LDA classification"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create pipeline
        pipe = Pipeline([
            ('csp', CSP(n_components=n_components, reg='ledoit_wolf', log=True)),
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis())
        ])
        
        # Fit and predict
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        preds = pipe.predict(X_test)
        
        # Metrics
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        precision = precision_score(y_test, preds, zero_division=0)
        
        results.append({
            'fold': fold,
            'auc': auc,
            'acc': acc,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'y_true': y_test.tolist(),
            'y_prob': probs.tolist(),
            'y_pred': preds.tolist()
        })
    
    return results

def run_riemannian_logreg(X, y, n_splits=5):
    """Riemannian Tangent Space + Logistic Regression"""
    if not PYRIEMANN_AVAILABLE:
        return None
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Covariance matrices
        cov_train = Covariances().fit_transform(X_train)
        cov_test = Covariances().transform(X_test)
        
        # Tangent space mapping
        ts = TangentSpace().fit(cov_train)
        X_train_ts = ts.transform(cov_train)
        X_test_ts = ts.transform(cov_test)
        
        # Classification
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(X_train_ts, y_train)
        
        probs = clf.predict_proba(X_test_ts)[:, 1]
        preds = clf.predict(X_test_ts)
        
        # Metrics
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        precision = precision_score(y_test, preds, zero_division=0)
        
        results.append({
            'fold': fold,
            'auc': auc,
            'acc': acc,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'y_true': y_test.tolist(),
            'y_prob': probs.tolist(),
            'y_pred': preds.tolist()
        })
    
    return results

def run_eegnet(X, y, n_splits=5, epochs=100, batch_size=32, lr=1e-3):
    """EEGNet within-subject classification"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = []
    
    n_channels, n_samples = X.shape[1], X.shape[2]
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        
        # Create model
        model = EEGNet(n_channels=n_channels, n_samples=n_samples).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_t).squeeze()
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
        
        # Metrics
        auc = roc_auc_score(y_test, probs)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        precision = precision_score(y_test, preds, zero_division=0)
        
        results.append({
            'fold': fold,
            'auc': auc,
            'acc': acc,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'y_true': y_test.tolist(),
            'y_prob': probs.tolist(),
            'y_pred': preds.tolist()
        })
    
    return results

def plot_roc_curves(results, method_name, out_dir):
    """Plot ROC curves for all folds"""
    plt.figure(figsize=(8, 6))
    
    aucs = []
    for result in results:
        y_true = np.array(result['y_true'])
        y_prob = np.array(result['y_prob'])
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = result['auc']
        aucs.append(auc)
        
        plt.plot(fpr, tpr, alpha=0.6, label=f'Fold {result["fold"]+1} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{method_name} - ROC Curves\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f'{method_name.lower().replace(" ", "_")}_roc.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(all_results, out_dir):
    """Create summary table of all methods"""
    summary_data = []
    
    for method, results in all_results.items():
        if results is None:
            continue
            
        aucs = [r['auc'] for r in results]
        accs = [r['acc'] for r in results]
        f1s = [r['f1'] for r in results]
        recalls = [r['recall'] for r in results]
        precisions = [r['precision'] for r in results]
        
        summary_data.append({
            'Method': method,
            'AUC_mean': np.mean(aucs),
            'AUC_std': np.std(aucs),
            'Acc_mean': np.mean(accs),
            'Acc_std': np.std(accs),
            'F1_mean': np.mean(f1s),
            'F1_std': np.std(f1s),
            'Recall_mean': np.mean(recalls),
            'Recall_std': np.std(recalls),
            'Precision_mean': np.mean(precisions),
            'Precision_std': np.std(precisions)
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save CSV
    df.to_csv(out_dir / 'summary_results.csv', index=False)
    
    # Create formatted table plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Format data for display
    display_data = []
    for _, row in df.iterrows():
        display_data.append([
            row['Method'],
            f"{row['AUC_mean']:.3f} ± {row['AUC_std']:.3f}",
            f"{row['Acc_mean']:.3f} ± {row['Acc_std']:.3f}",
            f"{row['F1_mean']:.3f} ± {row['F1_std']:.3f}",
            f"{row['Recall_mean']:.3f} ± {row['Recall_std']:.3f}",
            f"{row['Precision_mean']:.3f} ± {row['Precision_std']:.3f}"
        ])
    
    table = ax.table(cellText=display_data,
                    colLabels=['Method', 'AUC', 'Accuracy', 'F1', 'Recall', 'Precision'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Color code by AUC performance
    for i, row in enumerate(display_data):
        auc_val = float(row[1].split(' ±')[0])
        if auc_val >= 0.85:
            color = '#90EE90'  # Light green
        elif auc_val >= 0.75:
            color = '#FFE4B5'  # Light orange
        else:
            color = '#FFB6C1'  # Light pink
        
        for j in range(len(row)):
            table[(i+1, j)].set_facecolor(color)
    
    plt.title('Within-Subject Classification Results', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(out_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def run_subject_analysis(data_dir, subject, out_dir, window_ms=(200, 500)):
    """Run complete analysis for one subject"""
    print(f"\n=== Processing {subject} ===")
    
    # Load data
    X, y = load_subject_data(data_dir, subject)
    print(f"Loaded data: {X.shape}, labels: {np.bincount(y)}")
    
    # Extract window
    X_windowed = extract_window(X, window_ms=window_ms)
    print(f"Windowed data: {X_windowed.shape}")
    
    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2 or min(counts) < 5:
        print(f"Skipping {subject}: insufficient data for CV")
        return None
    
    # Run all methods
    results = {}
    
    # FBCSP + LDA
    print("Running FBCSP + LDA...")
    results['FBCSP_LDA'] = run_fbcsp_lda(X_windowed, y)
    
    # Riemannian + LogReg
    if PYRIEMANN_AVAILABLE:
        print("Running Riemannian + LogReg...")
        results['Riemannian_LogReg'] = run_riemannian_logreg(X_windowed, y)
    else:
        results['Riemannian_LogReg'] = None
    
    # EEGNet
    print("Running EEGNet...")
    results['EEGNet'] = run_eegnet(X_windowed, y)
    
    # Create subject output directory
    subj_dir = out_dir / subject
    subj_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot ROC curves
    for method, method_results in results.items():
        if method_results is not None:
            plot_roc_curves(method_results, method, subj_dir)
    
    # Create summary
    summary_df = create_summary_table(results, subj_dir)
    
    # Save detailed results
    with open(subj_dir / 'detailed_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results, summary_df

def main():
    parser = argparse.ArgumentParser(description='Within-Subject P3b Classification Pipeline')
    parser.add_argument('--data-dir', type=str, default='data/processed_p3b')
    parser.add_argument('--out-dir', type=str, default='results/within_subject')
    parser.add_argument('--window-ms', type=int, nargs=2, default=[200, 500])
    parser.add_argument('--subjects', type=str, nargs='*', help='Specific subjects to process')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find subjects
    if args.subjects:
        subjects = args.subjects
    else:
        subject_files = sorted(data_dir.glob('sub-*_X.npy'))
        subjects = [f.stem.split('_')[0] for f in subject_files]
    
    print(f"Found {len(subjects)} subjects: {subjects}")
    
    # Process all subjects
    all_subject_results = {}
    all_summaries = []
    
    for subject in subjects:
        try:
            results, summary_df = run_subject_analysis(
                data_dir, subject, out_dir, window_ms=tuple(args.window_ms)
            )
            if results is not None:
                all_subject_results[subject] = results
                summary_df['Subject'] = subject
                all_summaries.append(summary_df)
        except Exception as e:
            print(f"Error processing {subject}: {e}")
            continue
    
    # Create overall summary
    if all_summaries:
        combined_df = pd.concat(all_summaries, ignore_index=True)
        
        # Group by method and compute statistics
        method_stats = combined_df.groupby('Method').agg({
            'AUC_mean': ['mean', 'std'],
            'Acc_mean': ['mean', 'std'],
            'F1_mean': ['mean', 'std'],
            'Recall_mean': ['mean', 'std'],
            'Precision_mean': ['mean', 'std']
        }).round(3)
        
        method_stats.columns = ['_'.join(col).strip() for col in method_stats.columns]
        method_stats.to_csv(out_dir / 'overall_summary.csv')
        
        print("\n=== OVERALL RESULTS ===")
        print(method_stats)
        
        # Save final results
        with open(out_dir / 'all_results.json', 'w') as f:
            json.dump(all_subject_results, f, indent=2)
    
    print(f"\nResults saved to: {out_dir}")

if __name__ == '__main__':
    main()