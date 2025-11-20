import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
from sklearn.utils import shuffle
from glob import glob
from tqdm import tqdm
import json

# -------------------------
# 1. EEGNet architecture
# -------------------------
class EEGNet(nn.Module):
    def __init__(self, chans=64, samples=1024, num_classes=2):
        super().__init__()
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), padding="same", bias=False),
            nn.BatchNorm2d(8)
        )
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(8, 16, (chans, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        
        self.separable = nn.Sequential(
            nn.Conv2d(16, 16, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )
        
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * (samples // 32), num_classes)
        )
    
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.classify(x)
        return x

# -------------------------
# 2. Dataset wrapper
# -------------------------
class P3BDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# 3. Train function with class balancing
# -------------------------
def train_model(train_loader, test_loader, chans, samples, class_weights):
    model = EEGNet(chans=chans, samples=samples)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(10):  # More epochs for better learning
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
    
    # Eval
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            out = model(X_batch)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())
            all_labels.extend(y_batch.numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# -------------------------
# 4. Load all subjects
# -------------------------
def load_all_subjects(base_path="data/processed_p3b"):
    subjects = {}
    for npy_file in sorted(glob(os.path.join(base_path, "*_X.npy"))):
        sub = os.path.basename(npy_file).split("_")[0]
        X = np.load(npy_file)
        y = np.load(os.path.join(base_path, f"{sub}_y.npy"))
        subjects[sub] = (X, y)
    return subjects

# -------------------------
# 5. LOSO cross-validation
# -------------------------
def run_loso():
    subjects = load_all_subjects()
    subs = sorted(subjects.keys())
    
    metrics = {
        "accuracy": [],
        "f1": [],
        "roc_auc": [],
        "precision": [],
        "recall": [],
        "confusion_matrices": {}
    }
    
    print("Running LOSO on", len(subs), "subjects...")
    
    for test_sub in subs:
        print(f"\n=== Test Subject: {test_sub} ===")
        
        X_test, y_test = subjects[test_sub]
        
        # Train on all except this one
        X_train_list, y_train_list = [], []
        for s in subs:
            if s != test_sub:
                X_tr, y_tr = subjects[s]
                X_train_list.append(X_tr)
                y_train_list.append(y_tr)
        
        X_train = np.concatenate(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # Shuffle training data
        X_train, y_train = shuffle(X_train, y_train)
        
        # Build datasets
        chans = X_train.shape[1]
        samples = X_train.shape[2]
        
        train_loader = DataLoader(
            P3BDataset(X_train, y_train),
            batch_size=64,
            shuffle=True
        )
        
        test_loader = DataLoader(
            P3BDataset(X_test, y_test),
            batch_size=64,
            shuffle=False
        )
        
        # Calculate class weights for balancing
        unique, counts = np.unique(y_train, return_counts=True)
        class_weights = len(y_train) / (len(unique) * counts)
        print(f"Class weights: {class_weights}")
        
        # Train + evaluate
        labels, preds, probs = train_model(train_loader, test_loader, chans, samples, class_weights)
        
        # Metrics
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        roc_auc = roc_auc_score(labels, probs)
        
        metrics["accuracy"].append(acc)
        metrics["f1"].append(f1)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["roc_auc"].append(roc_auc)
        metrics["confusion_matrices"][test_sub] = confusion_matrix(labels, preds).tolist()
        
        print(f"Acc={acc:.3f}, F1={f1:.3f}, Recall={recall:.3f}, AUC={roc_auc:.3f}")
    
    # Save metrics
    with open("loso_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("\nDONE. Saved LOSO metrics → loso_results.json")
    return metrics

if __name__ == "__main__":
    run_loso()