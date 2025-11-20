import json
import numpy as np

# Load LOSO results
with open("loso_results.json", "r") as f:
    results = json.load(f)

# Calculate summary statistics
metrics = ["accuracy", "f1", "roc_auc", "precision", "recall"]

print("=== LOSO Cross-Validation Results Summary ===")
print(f"Number of subjects: {len(results['accuracy'])}")
print()

for metric in metrics:
    values = np.array(results[metric])
    mean_val = np.mean(values)
    std_val = np.std(values)
    ci_95 = 1.96 * std_val / np.sqrt(len(values))
    
    print(f"{metric.upper()}:")
    print(f"  Mean: {mean_val:.3f} ± {std_val:.3f}")
    print(f"  95% CI: [{mean_val-ci_95:.3f}, {mean_val+ci_95:.3f}]")
    print(f"  Range: [{np.min(values):.3f}, {np.max(values):.3f}]")
    print()

# Subject-wise performance
print("=== Per-Subject Performance ===")
subjects = list(results["confusion_matrices"].keys())
for i, sub in enumerate(subjects):
    acc = results["accuracy"][i]
    f1 = results["f1"][i]
    auc = results["roc_auc"][i]
    print(f"{sub}: Acc={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")

print("\n=== Research Paper Summary ===")
print(f"LOSO Accuracy: {np.mean(results['accuracy']):.1f}% ± {np.std(results['accuracy']):.1f}%")
print(f"LOSO F1-Score: {np.mean(results['f1']):.3f} ± {np.std(results['f1']):.3f}")
print(f"LOSO ROC-AUC: {np.mean(results['roc_auc']):.3f} ± {np.std(results['roc_auc']):.3f}")