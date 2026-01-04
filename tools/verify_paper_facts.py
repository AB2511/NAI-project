"""Verify all factual claims in the paper against actual results."""
import json
import numpy as np

# Load EEGNet LOSO results
with open('results/eegnet_v5/eegnet_loso_summary.json') as f:
    eegnet = json.load(f)

aucs = [f['auc'] for f in eegnet['folds']]
accs = [f['acc'] for f in eegnet['folds']]

print('=== EEGNet LOSO Results ===')
print(f'AUC: {np.mean(aucs):.2f} +/- {np.std(aucs, ddof=1):.2f}')
print(f'Accuracy: {np.mean(accs):.2f}')
print(f'Min AUC: {min(aucs):.2f} (subject: {eegnet["folds"][np.argmin(aucs)]["subject"]})')
print(f'Max AUC: {max(aucs):.2f} (subject: {eegnet["folds"][np.argmax(aucs)]["subject"]})')

# Check subjects > 0.67
above_67 = [(f['subject'], f['auc']) for f in eegnet['folds'] if f['auc'] > 0.67]
print(f'Subjects > 0.67: {[s[0] for s in above_67]}')

# Check subjects < 0.50
below_50 = [(f['subject'], f['auc']) for f in eegnet['folds'] if f['auc'] < 0.50]
print(f'Subjects < 0.50: {[s[0] for s in below_50]}')

# Load ML baselines
with open('results/ml_baselines/loso_baselines_summary.json') as f:
    ml = json.load(f)

print()
print('=== ML Baselines ===')
for m in ml:
    print(f"{m['model']}: AUC={m['auc_mean']:.2f} +/- {m['auc_std']:.2f}, Acc={m['acc_mean']:.2f}")

# Per-subject AUC table verification
print()
print('=== Per-Subject AUC (for Appendix B) ===')
for f in eegnet['folds']:
    print(f"{f['subject']}: {f['auc']:.2f}")

# Training data stats
print()
print('=== Training Stats ===')
train_ns = [f['train_n'] for f in eegnet['folds']]
test_ns = [f['test_n'] for f in eegnet['folds']]
print(f'Train epochs per fold: ~{int(np.mean(train_ns))}')
print(f'Test epochs per fold: ~{int(np.mean(test_ns))}')
