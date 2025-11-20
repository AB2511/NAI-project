# evaluate_model.py
"""
Comprehensive model evaluation and analysis for P300 cognitive state classifier.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, learning_curve
import warnings
warnings.filterwarnings('ignore')

def load_model(model_path):
    """Load trained model pipeline."""
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_and_prepare_data(p300_csv, state_csv, pipeline):
    """Load and prepare data using same preprocessing as training."""
    from train_classifier import build_dataset
    
    df = build_dataset(p300_csv, state_csv, 
                      window_s=pipeline.get('window_s', 2.0),
                      state_confidence_thresh=pipeline.get('confidence_thresh', 0.5))
    
    if df is None:
        return None, None, None
    
    df = df.dropna()
    
    # Encode labels
    le = pipeline['label_encoder']
    y = le.transform(df['state'])
    
    # Extract features
    feature_cols = pipeline.get('feature_columns', [])
    if feature_cols:
        X = df[feature_cols].values
    else:
        # Fallback
        feature_cols = [col for col in df.columns if col not in ['ts', 'state', 'confidence']]
        X = df[feature_cols].values
    
    return X, y, df

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()  # Close instead of show for headless mode

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance."""
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature importances")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to: {save_path}")
    
    plt.close()  # Close instead of show for headless mode

def plot_learning_curve(model, X, y, save_path=None):
    """Plot learning curve to diagnose overfitting."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training accuracy')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation accuracy')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to: {save_path}")
    
    plt.close()  # Close instead of show for headless mode

def analyze_predictions(y_true, y_pred, y_pred_proba, classes):
    """Analyze prediction patterns."""
    print("\n=== Prediction Analysis ===")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, class_name in enumerate(classes):
        mask = y_true == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            print(f"{class_name}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    # Confidence analysis
    print(f"\nPrediction confidence statistics:")
    max_proba = np.max(y_pred_proba, axis=1)
    print(f"Mean confidence: {np.mean(max_proba):.4f}")
    print(f"Std confidence: {np.std(max_proba):.4f}")
    print(f"Min confidence: {np.min(max_proba):.4f}")
    print(f"Max confidence: {np.max(max_proba):.4f}")
    
    # Confidence vs accuracy
    correct = (y_true == y_pred)
    print(f"\nConfidence for correct predictions: {np.mean(max_proba[correct]):.4f}")
    print(f"Confidence for incorrect predictions: {np.mean(max_proba[~correct]):.4f}")

def main(args):
    print("=== P300 Cognitive State Classifier Evaluation ===")
    
    # Load model
    pipeline = load_model(args.model_path)
    if pipeline is None:
        return
    
    model = pipeline['model']
    scaler = pipeline['scaler']
    le = pipeline['label_encoder']
    
    print(f"Model classes: {list(le.classes_)}")
    print(f"Window size: {pipeline.get('window_s', 'unknown')}s")
    
    # Load data
    X, y, df = load_and_prepare_data(args.p300_csv, args.state_csv, pipeline)
    if X is None:
        print("Failed to load data")
        return
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)
    
    # Basic metrics
    accuracy = accuracy_score(y, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=le.classes_))
    
    # Detailed analysis
    analyze_predictions(y, y_pred, y_pred_proba, le.classes_)
    
    # Cross-validation
    print(f"\nCross-validation (5-fold):")
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Visualizations
    if args.plot:
        print("\nGenerating plots...")
        
        # Confusion matrix
        plot_confusion_matrix(y, y_pred, le.classes_, 
                            save_path=os.path.join(args.out_dir, 'confusion_matrix.png'))
        
        # Feature importance
        feature_cols = pipeline.get('feature_columns', [f'feature_{i}' for i in range(X.shape[1])])
        plot_feature_importance(model, feature_cols, top_n=20,
                              save_path=os.path.join(args.out_dir, 'feature_importance.png'))
        
        # Learning curve
        plot_learning_curve(model, X_scaled, y,
                           save_path=os.path.join(args.out_dir, 'learning_curve.png'))
    
    # Save detailed results
    results = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_samples': len(X),
        'n_features': X.shape[1],
        'classes': list(le.classes_),
        'class_distribution': {le.classes_[i]: int(np.sum(y == i)) for i in range(len(le.classes_))}
    }
    
    results_path = os.path.join(args.out_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("=== P300 Classifier Evaluation Results ===\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(y, y_pred, target_names=le.classes_))
    
    print(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate P300 cognitive state classifier")
    parser.add_argument('--model-path', type=str, default='models/p300_xgb_pipeline.joblib',
                       help='Path to trained model')
    parser.add_argument('--p300-csv', type=str, default='src/logs/p300_stream.csv',
                       help='Path to P300 stream CSV file')
    parser.add_argument('--state-csv', type=str, default='src/logs/state_stream.csv',
                       help='Path to state stream CSV file')
    parser.add_argument('--out-dir', type=str, default='models',
                       help='Output directory for evaluation results')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    main(args)