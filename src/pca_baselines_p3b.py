#!/usr/bin/env python3
"""
Publication-grade ML baselines for P3b (LOSO + PCA + LDA/SVM/Logistic/XGBoost)

Usage:
    python pca_baselines_p3b.py --data-dir data/processed_p3b --out-dir results/ml_baselines
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import time
import warnings
from collections import defaultdict

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.pipeline import make_pipeline

# xgboost may be optional
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# fix randomness
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

warnings.filterwarnings("ignore", category=UserWarning)

def load_data(data_dir):
    data_dir = Path(data_dir)
    # load global arrays
    X_all = np.load(data_dir / "all_X.npy")  # (N, C, T)
    y_all = np.load(data_dir / "all_y.npy")  # (N,)

    # reconstruct subject ids from per-subject files
    subj_files = sorted([p.name for p in data_dir.iterdir() if p.name.startswith("sub-") and p.name.endswith("_X.npy")])
    # Expect sub-001_X.npy ... sub-020_X.npy
    subjects = []
    samples_accum = 0
    sub_id_list = []
    for fname in subj_files:
        sub_prefix = fname.replace("_X.npy", "")
        # read y to get how many samples for that subject
        y_sub = np.load(data_dir / f"{sub_prefix}_y.npy")
        n = len(y_sub)
        sub_id_list.extend([sub_prefix.replace("sub-", "")] * n)
        samples_accum += n
    if samples_accum != X_all.shape[0]:
        print("Warning: Total samples across subject files do not match all_X.npy. Falling back to heuristic grouping.")
        # fallback: try to group by file order and equal splits
        # But ideally this shouldn't happen.
    groups = np.array(sub_id_list)
    if groups.shape[0] != X_all.shape[0]:
        # try to infer subject indices from file order by concatenating sub_X files
        print("Reconstructing groups by concatenating individual subject files (safe fallback).")
        groups = []
        new_X_list = []
        new_y_list = []
        subj_files_sorted = sorted([p for p in data_dir.iterdir() if p.name.startswith("sub-") and p.name.endswith("_X.npy")])
        for px in subj_files_sorted:
            sub_prefix = px.name.replace("_X.npy", "")
            x_sub = np.load(data_dir / f"{sub_prefix}_X.npy")
            y_sub = np.load(data_dir / f"{sub_prefix}_y.npy")
            new_X_list.append(x_sub)
            new_y_list.append(y_sub)
            groups.extend([sub_prefix.replace("sub-", "")] * len(y_sub))
        X_all = np.concatenate(new_X_list, axis=0)
        y_all = np.concatenate(new_y_list, axis=0)
        groups = np.array(groups)

    return X_all, y_all, groups

def flatten(X):
    # X: (N, C, T) -> (N, C*T)
    N, C, T = X.shape
    return X.reshape(N, C * T)

def per_fold_class_weights(y_train):
    # return dict for sklearn (class_weight) and scale_pos_weight for xgboost
    unique, counts = np.unique(y_train, return_counts=True)
    d = dict(zip(unique, counts))
    n_neg = d.get(0, 0)
    n_pos = d.get(1, 0)
    if n_pos == 0:
        cw = {0: 1.0, 1: 1.0}
        scale_pos_weight = 1.0
    else:
        # inverse frequency weighting
        w0 = (n_pos + n_neg) / (2.0 * n_neg) if n_neg > 0 else 1.0
        w1 = (n_pos + n_neg) / (2.0 * n_pos)
        cw = {0: w0, 1: w1}
        scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    return cw, scale_pos_weight

def safe_auc(y_true, y_proba):
    try:
        return roc_auc_score(y_true, y_proba)
    except Exception:
        return float("nan")

def evaluate_models(X, y, groups, out_dir, n_components=0.95, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logo = LeaveOneGroupOut()
    fold_results = []
    models_trained = defaultdict(list)

    model_names = ["LDA", "Logistic", "SVM", "XGBoost"] if _HAS_XGB else ["LDA", "Logistic", "SVM"]
    fold_idx = 0
    t0 = time.time()
    for train_idx, test_idx in logo.split(X, y, groups):
        fold_idx += 1
        test_sub = groups[test_idx[0]]
        if verbose:
            print(f"\n=== LOSO fold {fold_idx} — test subject: {test_sub} ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Flatten and scale within-fold
        X_train_flat = flatten(X_train)
        X_test_flat = flatten(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)

        # PCA fit on train only
        pca = PCA(n_components=n_components, svd_solver='full', random_state=RANDOM_SEED)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        # class balancing weights for this fold
        cw, scale_pos_weight = per_fold_class_weights(y_train)
        if verbose:
            print("Train class counts:", np.unique(y_train, return_counts=True), " | class_weights:", cw)

        # define models (note: set random_state where applicable)
        # LDA
        lda = LinearDiscriminantAnalysis()
        # Logistic (use saga for multicore; set class_weight)
        log = LogisticRegression(solver='saga', class_weight=cw, max_iter=5000, random_state=RANDOM_SEED)
        # SVM (use probability=True for ROC; class_weight balanced)
        svm = SVC(kernel='rbf', probability=True, class_weight=cw, random_state=RANDOM_SEED)
        # XGBoost
        if _HAS_XGB:
            xg = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=RANDOM_SEED,
                scale_pos_weight=scale_pos_weight
            )

        # Train & evaluate each
        models = {"LDA": lda, "Logistic": log, "SVM": svm}
        if _HAS_XGB:
            models["XGBoost"] = xg

        for name, clf in models.items():
            try:
                clf.fit(X_train_pca, y_train)
                y_pred = clf.predict(X_test_pca)
                # try predict_proba for AUC; fallback to decision_function
                y_proba = None
                if hasattr(clf, "predict_proba"):
                    try:
                        y_proba = clf.predict_proba(X_test_pca)[:, 1]
                    except Exception:
                        y_proba = None
                if y_proba is None and hasattr(clf, "decision_function"):
                    try:
                        df = clf.decision_function(X_test_pca)
                        # if shape (n_samples,) -> use directly
                        if df.ndim == 1:
                            y_proba = df
                        else:
                            y_proba = df[:, 1]
                    except Exception:
                        y_proba = None

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = safe_auc(y_test, y_proba) if y_proba is not None else float("nan")
                cm = confusion_matrix(y_test, y_pred).tolist()

                res = {
                    "fold": fold_idx,
                    "test_subject": test_sub,
                    "model": name,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "acc": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "auc": float(auc) if not np.isnan(auc) else None,
                    "confusion_matrix": cm
                }
                fold_results.append(res)
                models_trained[name].append({
                    "fold": fold_idx,
                    "test_subject": test_sub,
                    "model_obj": clf,
                    "scaler": scaler,
                    "pca": pca
                })
                if verbose:
                    print(f"{name} | Acc={acc:.3f} F1={f1:.3f} Recall={rec:.3f} AUC={auc if not np.isnan(auc) else 'nan'}")
            except Exception as e:
                print(f"  ERROR training {name} on fold {fold_idx}: {e}")

    # results -> dataframe
    df = pd.DataFrame(fold_results)
    df.to_csv(out_dir / "loso_baselines_per_fold.csv", index=False)
    # Save trained metadata (not whole model objects) - we save one PCA+scaler per fold for reference
    joblib.dump(models_trained, out_dir / "models_trained_foldwise.joblib", compress=3)

    # compute aggregated stats per model
    summary = []
    for name in model_names:
        df_m = df[df["model"] == name]
        if df_m.shape[0] == 0:
            continue
        acc_mean = df_m["acc"].mean()
        acc_std = df_m["acc"].std(ddof=1)
        n = df_m.shape[0]
        acc_ci_low = acc_mean - 1.96 * acc_std / np.sqrt(max(1, n))
        acc_ci_high = acc_mean + 1.96 * acc_std / np.sqrt(max(1, n))

        auc_mean = df_m["auc"].dropna().mean() if "auc" in df_m else float("nan")
        auc_std = df_m["auc"].dropna().std(ddof=1) if "auc" in df_m else float("nan")

        summary.append({
            "model": name,
            "n_folds": int(df_m.shape[0]),
            "acc_mean": float(acc_mean),
            "acc_std": float(acc_std) if not np.isnan(acc_std) else None,
            "acc_95ci": [float(acc_ci_low), float(acc_ci_high)],
            "auc_mean": float(auc_mean) if not np.isnan(auc_mean) else None,
            "auc_std": float(auc_std) if not np.isnan(auc_std) else None
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "loso_baselines_summary.csv", index=False)
    with open(out_dir / "loso_baselines_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    t1 = time.time()
    print(f"\nDone. Time elapsed: {(t1-t0)/60:.2f} minutes")
    print(f"Per-fold CSV saved to: {out_dir / 'loso_baselines_per_fold.csv'}")
    print(f"Summary saved to: {out_dir / 'loso_baselines_summary.csv'}")
    return df, summary_df

def main():
    parser = argparse.ArgumentParser(description="PCA + Classical ML baselines (LOSO) for P3b")
    parser.add_argument("--data-dir", type=str, default="data/processed_p3b", help="Processed data dir containing all_X.npy, all_y.npy, sub-###_X.npy files.")
    parser.add_argument("--out-dir", type=str, default="results/ml_baselines", help="Directory to store results.")
    parser.add_argument("--pca-variance", type=float, default=0.95, help="PCA variance to keep (0-1) or integer components")
    parser.add_argument("--no-xgb", action="store_true", help="Disable XGBoost even if available")
    args = parser.parse_args()

    if args.no_xgb:
        global _HAS_XGB
        _HAS_XGB = False

    print("Loading data from:", args.data_dir)
    X_all, y_all, groups = load_data(args.data_dir)
    print("X shape:", X_all.shape, "y shape:", y_all.shape, "n_groups:", len(np.unique(groups)))
    # run evaluation
    df, summary_df = evaluate_models(X_all, y_all, groups, args.out_dir, n_components=args.pca_variance, verbose=True)
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()