# train_classifier.py
"""
ML classifier training for P300 cognitive state recognition.

Loads P300 and state CSV logs, builds windowed features, trains XGBoost classifier.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def aggregate_window_features(p300_df, window_start, window_end):
    """Extract statistical features from P300 samples in time window."""
    w = p300_df[(p300_df['ts'] >= window_start) & (p300_df['ts'] <= window_end)]
    if w.empty or len(w) < 2:
        return None
    
    vals = {}
    feature_cols = ['amplitude_uv', 'latency_ms', 'smoothed_amp_uv', 'fatigue_index']
    
    for col in feature_cols:
        if col not in w.columns:
            continue
            
        arr = w[col].values.astype(float)
        if len(arr) == 0:
            continue
            
        vals[f'{col}_mean'] = arr.mean()
        vals[f'{col}_std'] = arr.std(ddof=0) if len(arr) > 1 else 0.0
        vals[f'{col}_min'] = arr.min()
        vals[f'{col}_max'] = arr.max()
        vals[f'{col}_median'] = np.median(arr)
        vals[f'{col}_skew'] = skew(arr) if len(arr) > 2 else 0.0
        vals[f'{col}_kurtosis'] = kurtosis(arr) if len(arr) > 3 else 0.0
        vals[f'{col}_last'] = arr[-1]
        vals[f'{col}_count'] = len(arr)
        
        # slope (linear trend)
        if len(arr) > 1:
            vals[f'{col}_slope'] = (arr[-1] - arr[0]) / (len(arr) - 1)
        else:
            vals[f'{col}_slope'] = 0.0
            
        # range
        vals[f'{col}_range'] = arr.max() - arr.min()
        
        # percentiles
        vals[f'{col}_p25'] = np.percentile(arr, 25)
        vals[f'{col}_p75'] = np.percentile(arr, 75)
    
    return vals

def build_dataset(p300_csv, state_csv, window_s=2.0, state_confidence_thresh=0.5):
    """Build supervised dataset from P300 and state streams."""
    print(f"Loading data from {p300_csv} and {state_csv}")
    
    try:
        p300 = pd.read_csv(p300_csv)
        states = pd.read_csv(state_csv)
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return None
    
    print(f"Loaded {len(p300)} P300 samples and {len(states)} state samples")
    
    # Filter by confidence threshold
    states = states[states['confidence'] >= state_confidence_thresh]
    print(f"After confidence filtering (>={state_confidence_thresh}): {len(states)} state samples")
    
    if len(states) == 0:
        print("No state samples after confidence filtering!")
        return None
    
    rows = []
    for i, s in states.iterrows():
        t_label = float(s['ts'])
        win_start = t_label - window_s
        win_end = t_label
        
        feats = aggregate_window_features(p300, win_start, win_end)
        if feats is None:
            continue
            
        feats['ts'] = t_label
        feats['state'] = s['state']
        feats['confidence'] = float(s['confidence'])
        rows.append(feats)
    
    if not rows:
        print("No valid feature windows created!")
        return None
        
    df = pd.DataFrame(rows)
    print(f"Created {len(df)} feature samples")
    print(f"State distribution: {df['state'].value_counts().to_dict()}")
    
    return df

def evaluate_model(model, X_test, y_test, label_encoder):
    """Comprehensive model evaluation."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print("\nTop 10 Feature Importances:")
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        importances = list(zip(feature_names, model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        for name, imp in importances[:10]:
            print(f"{name}: {imp:.4f}")
    
    return accuracy, y_pred, y_pred_proba

def main(args):
    print("=== P300 Cognitive State Classifier Training ===")
    
    # Build dataset
    df = build_dataset(args.p300_csv, args.state_csv, 
                      window_s=args.window, 
                      state_confidence_thresh=args.confidence_thresh)
    
    if df is None or len(df) < 10:
        print("Insufficient data for training!")
        return
    
    # Prepare features and labels
    df = df.dropna()
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['state'])
    
    # Drop non-feature columns
    feature_cols = [col for col in df.columns if col not in ['ts', 'state', 'confidence']]
    X = df[feature_cols].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Classes: {le.classes_}")
    
    # Check for class imbalance
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(le.inverse_transform(unique), counts))
    print(f"Class distribution: {class_dist}")
    
    # Split data
    test_size = min(0.3, max(0.1, 50 / len(df)))  # Adaptive test size
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining XGBoost classifier...")
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.array([class_weights[i] for i in y_train])
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    # Train model
    model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    
    # Evaluate
    accuracy, y_pred, y_pred_proba = evaluate_model(model, X_test_scaled, y_test, le)
    
    # Cross-validation if we have enough data
    if len(df) > 30:
        print("\nCross-validation scores:")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=min(5, len(df)//10), scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save model
    os.makedirs(args.out_dir, exist_ok=True)
    pipeline = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_columns': feature_cols,
        'window_s': args.window,
        'confidence_thresh': args.confidence_thresh,
        'accuracy': accuracy
    }
    
    model_path = os.path.join(args.out_dir, 'p300_xgb_pipeline.joblib')
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Save feature names for debugging
    feature_info_path = os.path.join(args.out_dir, 'feature_info.txt')
    with open(feature_info_path, 'w') as f:
        f.write("Feature columns:\n")
        for i, col in enumerate(feature_cols):
            f.write(f"{i}: {col}\n")
        f.write(f"\nClasses: {list(le.classes_)}\n")
        f.write(f"Window size: {args.window}s\n")
        f.write(f"Confidence threshold: {args.confidence_thresh}\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n")
    
    print(f"Feature info saved to: {feature_info_path}")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train P300 cognitive state classifier")
    parser.add_argument('--p300-csv', type=str, default='src/logs/p300_stream.csv',
                       help='Path to P300 stream CSV file')
    parser.add_argument('--state-csv', type=str, default='src/logs/state_stream.csv',
                       help='Path to state stream CSV file')
    parser.add_argument('--window', type=float, default=2.0,
                       help='Time window size in seconds for feature extraction')
    parser.add_argument('--confidence-thresh', type=float, default=0.5,
                       help='Minimum confidence threshold for state labels')
    parser.add_argument('--out-dir', type=str, default='models',
                       help='Output directory for trained model')
    
    args = parser.parse_args()
    main(args)