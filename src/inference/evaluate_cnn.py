# src/inference/evaluate_cnn.py
import argparse, joblib, os, numpy as np, pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from train_cnn_classifier import P300WindowDataset

def load_and_predict(script_model_path, p300_csv, state_csv):
    model = torch.jit.load(script_model_path, map_location="cpu")
    p300, state = pd.read_csv(p300_csv), pd.read_csv(state_csv)
    ds = P300WindowDataset(p300, state)
    le = ds.le
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            preds = logits.argmax(dim=1).numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())
    print("Accuracy:", accuracy_score(y_true,y_pred))
    print(classification_report(y_true,y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--script-model", required=True)
    p.add_argument("--p300-csv", required=True)
    p.add_argument("--state-csv", required=True)
    args = p.parse_args()
    load_and_predict(args.script_model, args.p300_csv, args.state_csv)