import os
import json
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ---------------------------------------------------------------------
#                           EEGNet v4 Architecture
# ---------------------------------------------------------------------

class EEGNet(nn.Module):
    def __init__(self, Chans=26, Samples=308, dropoutRate=0.25, kernelLength=64,
                 F1=8, D=2, F2=16, norm_rate=0.25):
        super().__init__()
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernelLength), padding='same', bias=False),
            nn.BatchNorm2d(F1)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2 * (Samples // 32), 1)
        )
    
    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return self.classify(x)

# ---------------------------------------------------------------------
#                   Dataset class for LOSO EEG training
# ---------------------------------------------------------------------

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------------------------------------------------
#                           Focal Loss
# ---------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.unsqueeze(1))
        probas = torch.sigmoid(logits)
        p_t = targets * probas + (1 - targets) * (1 - probas)
        focal_term = (1 - p_t) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * focal_term * bce_loss).mean()

# ---------------------------------------------------------------------
#                       Utility Functions
# ---------------------------------------------------------------------

def load_subject_data(data_dir, subject):
    path = os.path.join(data_dir, subject + ".npz")
    obj = np.load(path)
    return obj["X"], obj["y"]

def window_slice(X, start_ms=200, end_ms=500, sfreq=128):
    start = int(start_ms / 1000 * sfreq)
    end = int(end_ms / 1000 * sfreq)
    return X[:, :, start:end]

def create_sampler(y):
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(y),
                                         y=y)
    weights = np.array([class_weights[int(label)] for label in y])
    return WeightedRandomSampler(weights, len(weights), replacement=True)

# ---------------------------------------------------------------------
#                     Train 1 LOSO Fold
# ---------------------------------------------------------------------

def train_one_fold(model, train_loader, val_loader, device, epochs=60, lr=1e-3):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(gamma=2.0)
    
    best_auc = 0
    patience = 8
    bad_epochs = 0
    
    for epoch in range(1, epochs + 1):
        
        model.train()
        train_losses = []
        
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                logits = model(Xb)
                all_logits.append(logits.cpu().numpy())
                all_targets.append(yb.numpy())
        
        logits = np.vstack(all_logits).ravel()
        targets = np.hstack(all_targets)
        probas = 1 / (1 + np.exp(-logits))
        
        auc = roc_auc_score(targets, probas)
        
        print(f"Epoch {epoch:02d} | train_loss={np.mean(train_losses):.4f} | val_auc={auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            bad_epochs = 0
            torch.save(model.state_dict(), "temp_best_model.pth")
        else:
            bad_epochs += 1
        
        if bad_epochs >= patience:
            print("Early stopping")
            break
    
    model.load_state_dict(torch.load("temp_best_model.pth"))
    return model, best_auc

# ---------------------------------------------------------------------
#                   LOSO Cross-Subject Evaluation
# ---------------------------------------------------------------------

def run_loso(data_dir, out_dir, device, epochs, batch_size):
    
    os.makedirs(out_dir, exist_ok=True)
    
    subjects = sorted([f.replace(".npz", "") for f in os.listdir(data_dir) if f.endswith(".npz")])
    results = {"folds": []}
    
    for test_sub in subjects:
        print(f"\n=== LOSO: Testing on {test_sub} ===")
        
        # Load test
        X_test, y_test = load_subject_data(data_dir, test_sub)
        X_test = window_slice(X_test)
        
        # Load train
        X_train_list, y_train_list = [], []
        for sub in subjects:
            if sub == test_sub:
                continue
            Xt, yt = load_subject_data(data_dir, sub)
            Xt = window_slice(Xt)
            X_train_list.append(Xt)
            y_train_list.append(yt)
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        # Standardization
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        scaler.fit(X_train_reshaped)
        X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_test = scaler.transform(X_test_reshaped).reshape(X_test.shape)
        
        # Format
        X_train = X_train[:, None, :, :]
        X_test = X_test[:, None, :, :]
        
        # Sampler
        sampler = create_sampler(y_train)
        
        train_loader = DataLoader(EEGDataset(X_train, y_train),
                                  batch_size=batch_size,
                                  sampler=sampler)
        val_loader = DataLoader(EEGDataset(X_test, y_test),
                                batch_size=batch_size,
                                shuffle=False)
        
        # Model
        model = EEGNet().to(device)
        model, best_auc = train_one_fold(
            model, train_loader, val_loader, device, epochs=epochs
        )
        
        # Final evaluation
        model.eval()
        all_logits = []
        with torch.no_grad():
            for Xb, _ in val_loader:
                Xb = Xb.to(device)
                logits = model(Xb)
                all_logits.append(logits.cpu().numpy())
        
        logits = np.vstack(all_logits).ravel()
        probas = 1 / (1 + np.exp(-logits))
        
        acc = accuracy_score(y_test, probas > 0.5)
        f1 = f1_score(y_test, probas > 0.5)
        rec = recall_score(y_test, probas > 0.5)
        
        results["folds"].append({
            "subject": test_sub,
            "auc": float(best_auc),
            "acc": float(acc),
            "f1": float(f1),
            "recall": float(rec)
        })
        
        with open(os.path.join(out_dir, f"{test_sub}_metrics.json"), "w") as f:
            json.dump(results["folds"][-1], f, indent=4)
    
    # Save summary
    with open(os.path.join(out_dir, "eegnet_loso_summary.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nDONE. Summary saved.")

# ---------------------------------------------------------------------
#                           Main CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    run_loso(args.data_dir, args.out_dir, device, args.epochs, args.batch_size)