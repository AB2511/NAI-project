# src/inference/train_hybrid_model.py
"""
Train hybrid pipeline: CNN encoder (use saved torch checkpoint) to extract embeddings,
then train an XGBoost classifier on embeddings.
Saves models/hybrid_xgb.joblib + models/encoder.pt (torchscript)
"""

import argparse, os, joblib, numpy as np, pandas as pd
import torch, torch.nn as nn
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Reuse the P300WindowDataset from train_cnn_classifier: copy minimal loader or import
from train_cnn_classifier import P300WindowDataset, P300_CNN, load_csvs  # adjust path if necessary

def extract_embeddings(model, ds, device="cpu", batch_size=64):
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            # we want intermediate features before final fc: reuse model.net up to Flatten
            # easiest: create a small wrapper
            feat = model.net[:-3](X)  # adapt index if structure differs
            feat = feat.view(feat.size(0), -1).cpu().numpy()
            embeddings.append(feat)
            labels.append(y.numpy())
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)
    return embeddings, labels

def main(args):
    p300, state = load_csvs(args.p300_csv, args.state_csv)
    ds = P300WindowDataset(p300, state, window_s=args.window_s)
    le = ds.le
    # create dataset indices
    idxs = np.arange(len(ds))
    tr, te = train_test_split(idxs, test_size=0.2, random_state=42, stratify=[ds.records[i]['state'] for i in idxs])
    tr_ds = torch.utils.data.Subset(ds, tr)
    te_ds = torch.utils.data.Subset(ds, te)

    # load encoder
    model = P300_CNN(n_channels=2, n_classes=len(le.classes_), base_filters=args.base_filters, dropout=args.dropout)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    X_train, y_train = extract_embeddings(model, tr_ds, device=args.device, batch_size=args.batch_size)
    X_test, y_test   = extract_embeddings(model, te_ds, device=args.device, batch_size=args.batch_size)

    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric="mlogloss")
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)
    print("Hybrid Test acc:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=le.classes_))
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(xgb, os.path.join(args.out_dir, "hybrid_xgb.joblib"))
    joblib.dump(le, os.path.join(args.out_dir, "p300_cnn_label_encoder.joblib"))
    print("Saved hybrid model to", args.out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--p300-csv", required=True)
    p.add_argument("--state-csv", required=True)
    p.add_argument("--checkpoint", required=True, help="checkpoint (p300_cnn_stateful.pth)")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--window-s", type=float, default=1.5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--base-filters", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    args = p.parse_args()
    main(args)