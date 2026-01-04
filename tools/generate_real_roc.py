"""
Generate ROC curve from ACTUAL per-fold predictions.
This creates a proper mean ROC curve across all LOSO folds.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import os

OUT_DIR = 'figures/final'
os.makedirs(OUT_DIR, exist_ok=True)

# Load all per-fold results
fold_dir = Path('results/eegnet_v5/per_fold')
all_fpr = []
all_tpr = []
all_auc = []

# Common FPR points for interpolation
mean_fpr = np.linspace(0, 1, 100)

print("Loading per-fold ROC data...")
for fold_file in sorted(fold_dir.glob('*.json')):
    with open(fold_file) as f:
        data = json.load(f)
    
    y_true = np.array(data['final_metrics']['y_true'])
    y_prob = np.array(data['final_metrics']['y_prob'])
    
    # Compute ROC curve for this fold
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fold_auc = auc(fpr, tpr)
    
    # Interpolate to common FPR points
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0  # Ensure starts at 0
    
    all_tpr.append(interp_tpr)
    all_auc.append(fold_auc)
    
    print(f"  {data['subject']}: AUC = {fold_auc:.3f}")

# Compute mean and std
mean_tpr = np.mean(all_tpr, axis=0)
std_tpr = np.std(all_tpr, axis=0)
mean_tpr[-1] = 1.0  # Ensure ends at 1

mean_auc = np.mean(all_auc)
std_auc = np.std(all_auc, ddof=1)

print(f"\nMean AUC: {mean_auc:.2f} ± {std_auc:.2f}")

# Plot
fig, ax = plt.subplots(figsize=(6, 6))

# Chance line
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance (AUC = 0.50)')

# Mean ROC with confidence band
ax.plot(mean_fpr, mean_tpr, 'b-', lw=2.5, 
        label=f'EEGNet (AUC = {mean_auc:.2f} ± {std_auc:.2f})')

tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.2, color='blue',
                label='± 1 std. dev.')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve (LOSO Cross-Validation)', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/roc_eegnet_loso.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{OUT_DIR}/roc_eegnet_loso.svg', bbox_inches='tight')
print(f"\nSaved: {OUT_DIR}/roc_eegnet_loso.png")
plt.close()

print("\n✅ ROC figure generated from REAL per-fold predictions!")
