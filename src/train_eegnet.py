# src/train_eegnet.py
import os
import argparse
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import joblib
import json

# --------- Focal Loss for Class Imbalance ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
    
    def forward(self, logits, targets):
        ce_loss = cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets].to(logits.device)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()

# --------- Enhanced EEGNet implementation ----------
class EEGNet(nn.Module):
    def __init__(self, nchans, nsamples, n_classes, dropout=0.25):
        super().__init__()
        # Block 1: Temporal convolution
        self.firstconv = nn.Sequential(
            nn.Conv1d(nchans, 16, kernel_size=64, padding=32, bias=False),
            nn.BatchNorm1d(16)
        )
        
        # Block 2: Depthwise convolution (spatial filtering)
        self.depthwise = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, groups=16, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Dropout(dropout)
        )
        
        # Block 3: Separable convolution
        self.separable = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=16, padding=8, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(8),
            nn.Dropout(dropout)
        )
        
        # Calculate output size after convolutions
        out_len = nsamples // (4 * 8)  # Two pooling layers
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * out_len, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    
    def forward(self, x):
        # x shape: (B, C, T)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.classifier(x)
        return x

# --------- Dataset ----------
class EEGDataset(Dataset):
    def __init__(self, X, y, transform=None):
        # X expected shape: (N, C, T)
        self.X = X.astype(np.float32)
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        x = self.X[i]
        if self.transform:
            x = self.transform(x)
        return x, self.y[i]

# --------- helpers ----------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def collate_batch(batch):
    xs = [torch.from_numpy(item[0]) for item in batch]
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long)
    xs = torch.stack(xs)  # (B, C, T)
    return xs, ys

# --------- train loop ----------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        # ensure (B, C, T)
        if X.ndim == 3:
            pass
        out = model(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += X.size(0)
    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            out = model(X)
            loss = criterion(out, y)
            running_loss += loss.item() * X.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += X.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    if total == 0:
        return None, None, None
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return running_loss / total, correct / total, (all_preds, all_labels)

# --------- main ----------
def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    # prefer combined all_X.npy if exists
    data_dir = Path(args.data_dir)
    allX_path = data_dir / 'all_X.npy'
    ally_path = data_dir / 'all_y.npy'
    if allX_path.exists() and ally_path.exists():
        X = np.load(allX_path)    # (N, C, T)
        y = np.load(ally_path)
    else:
        # fall back to combining subject files
        X_list, y_list = [], []
        for p in sorted(data_dir.glob('sub-*_X.npy')):
            X_list.append(np.load(p))
            y_list.append(np.load(str(p).replace('_X.npy', '_y.npy')))
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
    
    # sanity shape: ensure channels are first (N, C, T)
    if X.ndim == 4 and X.shape[1] != X.shape[2]:
        # sometimes shape is (N, T, C)
        X = np.transpose(X, (0, 2, 1))
    
    N, C, T = X.shape
    print(f'Loaded data: N={N}, C={C}, T={T}, classes={np.unique(y)}')
    
    # encode labels to ints
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, out_dir / 'label_encoder.joblib')
    
    # train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_enc, test_size=args.test_size+args.val_size, stratify=y_enc, random_state=args.seed)
    val_prop = args.val_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_prop, stratify=y_temp, random_state=args.seed)
    
    print(f'Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}')
    
    # dataset & dataloaders
    train_ds = EEGDataset(X_train, y_train)
    val_ds = EEGDataset(X_val, y_val)
    test_ds = EEGDataset(X_test, y_test)
    
    # compute class weights for weighted sampler
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)
    
    # model
    model = EEGNet(nchans=C, nsamples=T, n_classes=len(le.classes_), dropout=args.dropout).to(device)
    
    # Use Focal Loss to handle class imbalance
    if args.use_focal:
        criterion = FocalLoss(gamma=2, alpha=[1.0, 3.0])  # Boost minority class
        print("Using Focal Loss with alpha=[1.0, 3.0]")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
        print("Using CrossEntropyLoss with label smoothing")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
    
    best_val = 0.0
    patience = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _ = eval_epoch(model, val_loader, criterion, device)
        print(f"[{epoch}/{args.epochs}] Train loss={train_loss:.4f} acc={train_acc:.3f} | Val loss={val_loss:.4f} acc={val_acc:.3f}")
        scheduler.step(val_acc if val_acc is not None else 0.0)
        
        # checkpoint
        if val_acc is not None and val_acc > best_val:
            best_val = val_acc
            torch.save({'model_state': model.state_dict(), 'le_classes': le.classes_, 'C':C, 'T':T}, out_dir / 'eegnet_best.pth')
            print("Saved best model.")
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop:
                print("Early stopping.")
                break
    
    # final evaluate best model on test
    ckpt = torch.load(out_dir / 'eegnet_best.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    _, test_acc, (preds, labels) = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # save final model and TorchScript for fast inference
    torch.save(model.state_dict(), out_dir / 'eegnet_state_dict.pth')
    model.eval()
    example = torch.randn(1, C, T).to(device)
    try:
        traced = torch.jit.trace(model.to('cpu'), example.cpu())
        traced.save(out_dir / 'eegnet_traced.pt')
        print("Exported TorchScript model.")
    except Exception as e:
        print("TorchScript export failed:", e)
    
    meta = {
        'test_acc': float(test_acc) if test_acc is not None else 0.0,
        'best_val_acc': float(best_val),
        'classes': [str(c) for c in le.classes_],
        'N': int(N), 'C': int(C), 'T': int(T)
    }
    with open(out_dir / 'train_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print("Training finished. Metadata saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/processed_p3b', help='where preprocessed .npy files live')
    parser.add_argument('--out-dir', type=str, default='models', help='where to save models')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-size', type=float, default=0.1)
    parser.add_argument('--test-size', type=float, default=0.1)
    parser.add_argument('--early-stop', type=int, default=15)
    parser.add_argument('--use-focal', action='store_true', help='use focal loss instead of CE')
    parser.add_argument('--cpu', action='store_true', help='force cpu')
    args = parser.parse_args()
    main(args)