# src/evaluate_eegnet.py
import numpy as np
import joblib
import json
import torch
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(data_dir='data/processed_p3b'):
    """Load preprocessed data"""
    data_path = Path(data_dir)
    
    # Try to load combined files first
    all_x_path = data_path / 'all_X.npy'
    all_y_path = data_path / 'all_y.npy'
    
    if all_x_path.exists() and all_y_path.exists():
        X = np.load(all_x_path)
        y = np.load(all_y_path)
    else:
        # Fall back to individual subject files
        X_list, y_list = [], []
        for p in sorted(data_path.glob('sub-*_X.npy')):
            X_list.append(np.load(p))
            y_list.append(np.load(str(p).replace('_X.npy', '_y.npy')))
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
    
    return X, y

def evaluate(model_dir='models', data_dir='data/processed_p3b', use_traced=True):
    """Evaluate trained EEGNet model"""
    model_path = Path(model_dir)
    
    # Load data
    X, y = load_data(data_dir)
    X = X.astype('float32')
    
    # Load label encoder
    le = joblib.load(model_path / 'label_encoder.joblib')
    y_encoded = le.transform(y)
    
    # Create string class names for display
    class_names = [f"Class_{int(c)}" for c in le.classes_]
    
    # Load model
    device = torch.device('cpu')  # Use CPU for evaluation
    
    if use_traced and (model_path / 'eegnet_traced.pt').exists():
        # Use TorchScript traced model (faster)
        model = torch.jit.load(model_path / 'eegnet_traced.pt', map_location=device)
        print("Using TorchScript traced model")
    else:
        # Load regular PyTorch model
        from train_eegnet import EEGNet
        
        # Load metadata to get model architecture
        with open(model_path / 'train_meta.json', 'r') as f:
            meta = json.load(f)
        
        model = EEGNet(nchans=meta['C'], nsamples=meta['T'], n_classes=len(meta['classes']))
        model.load_state_dict(torch.load(model_path / 'eegnet_state_dict.pth', map_location=device, weights_only=True))
        print("Using regular PyTorch model")
    
    model.eval()
    
    # Make predictions in batches
    batch_size = 128
    all_preds = []
    
    print(f"Evaluating on {len(X)} samples...")
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_end = min(i + batch_size, len(X))
            x_batch = torch.from_numpy(X[i:batch_end])
            
            # Get predictions
            outputs = model(x_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    
    all_preds = np.concatenate(all_preds)
    
    # Print classification report
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_encoded, all_preds, target_names=class_names))
    
    # Create and save confusion matrix
    cm = confusion_matrix(y_encoded, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names, 
                yticklabels=class_names, 
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('EEGNet P3b Classification - Confusion Matrix')
    plt.tight_layout()
    
    # Save confusion matrix
    cm_path = model_path / 'confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    # Calculate and display accuracy
    accuracy = (all_preds == y_encoded).mean()
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Show class-wise accuracy
    print("\nClass-wise Accuracy:")
    for i, class_name in enumerate(class_names):
        class_mask = (y_encoded == i)
        if class_mask.sum() > 0:
            class_acc = (all_preds[class_mask] == y_encoded[class_mask]).mean()
            print(f"  {class_name}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    plt.show()
    
    return accuracy, all_preds, y_encoded

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained EEGNet model')
    parser.add_argument('--model-dir', type=str, default='models', 
                       help='Directory containing trained model')
    parser.add_argument('--data-dir', type=str, default='data/processed_p3b',
                       help='Directory containing preprocessed data')
    parser.add_argument('--no-traced', action='store_true',
                       help='Use regular PyTorch model instead of TorchScript')
    
    args = parser.parse_args()
    
    evaluate(model_dir=args.model_dir, 
            data_dir=args.data_dir, 
            use_traced=not args.no_traced)