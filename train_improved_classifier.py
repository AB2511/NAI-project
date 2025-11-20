# train_improved_classifier.py
"""
Improved ML classifier training with hyperparameter tuning and better validation.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def aggregate_window_features(p300_df, window_start, window_end):
    """Extract comprehensive statistical features from P300 samples in time window."""
    w = p300_df[(p300_df['ts'] >= window_start) & (p300_df['ts'] <= window_end)]
    if w.empty or len(w) < 3:  # Need at least 3 samples for meaningful stats
        return None
    
    vals = {}
    feature_cols = ['amplitude_uv', 'latency_ms', 'smoothed_amp_uv', 'fatigue_index']
    
    for col in feature_cols:
        if col not in w.columns:
            continue
            
        arr = w[col].values.astype(float)
        if len(arr) == 0:
            continue
        
        # Basic statistics
        vals[f'{col}_mean'] = arr.mean()
        vals[f'{col}_std'] = arr.std(ddof=0) if len(arr) > 1 else 0.0
        vals[f'{col}_min'] = arr.min()
        vals[f'{col}_max'] = arr.max()
        vals[f'{col}_median'] = np.median(arr)
        vals[f'{col}_skew'] = skew(arr) if len(arr) > 2 else 0.0
        vals[f'{col}_kurtosis'] = kurtosis(arr) if len(arr) > 3 else 0.0
        vals[f'{col}_last'] = arr[-1]
        vals[f'{col}_first'] = arr[0]
        vals[f'{col}_count'] = len(arr)
        
        # Trend and dynamics
        if len(arr) > 1:
            vals[f'{col}_slope'] = (arr[-1] - arr[0]) / (len(arr) - 1)
            vals[f'{col}_diff_mean'] = np.mean(np.diff(arr))
            vals[f'{col}_diff_std'] = np.std(np.diff(arr))
        else:
            vals[f'{col}_slope'] = 0.0
            vals[f'{col}_diff_mean'] = 0.0
            vals[f'{col}_diff_std'] = 0.0
            
        # Range and percentiles
        vals[f'{col}_range'] = arr.max() - arr.min()
        vals[f'{col}_iqr'] = np.percentile(arr, 75) - np.percentile(arr, 25)
        vals[f'{col}_p10'] = np.percentile(arr, 10)
        vals[f'{col}_p25'] = np.percentile(arr, 25)
        vals[f'{col}_p75'] = np.percentile(arr, 75)
        vals[f'{col}_p90'] = np.percentile(arr, 90)
        
        # Energy and power features
        vals[f'{col}_energy'] = np.sum(arr**2)
        vals[f'{col}_rms'] = np.sqrt(np.mean(arr**2))
        
        # Zero crossings and peaks
        if len(arr) > 2:
            zero_crossings = np.sum(np.diff(np.sign(arr - arr.mean())) != 0)
            vals[f'{col}_zero_crossings'] = zero_crossings
        else:
            vals[f'{col}_zero_crossings'] = 0
    
    # Cross-feature relationships
    if 'amplitude_uv' in w.columns and 'latency_ms' in w.columns:
        amp_arr = w['amplitude_uv'].values.astype(float)
        lat_arr = w['latency_ms'].values.astype(float)
        if len(amp_arr) > 1 and len(lat_arr) > 1:
            vals['amp_lat_corr'] = np.corrcoef(amp_arr, lat_arr)[0, 1] if not np.isnan(np.corrcoef(amp_arr, lat_arr)[0, 1]) else 0.0
            vals['amp_lat_ratio'] = np.mean(amp_arr) / (np.mean(lat_arr) + 1e-6)
    
    return vals

def build_dataset(p300_csv, state_csv, window_s=2.0, state_confidence_thresh=0.6, overlap=0.5):
    """Build supervised dataset with overlapping windows for more samples."""
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
    step_size = window_s * (1 - overlap)  # Overlap windows
    
    for i, s in states.iterrows():
        t_label = float(s['ts'])
        
        # Create multiple overlapping windows ending at this label
        for offset in np.arange(0, window_s, step_size):
            win_end = t_label - offset
            win_start = win_end - window_s
            
            feats = aggregate_window_features(p300, win_start, win_end)
            if feats is None:
                continue
                
            feats['ts'] = win_end
            feats['state'] = s['state']
            feats['confidence'] = float(s['confidence'])
            feats['window_offset'] = offset
            rows.append(feats)
    
    if not rows:
        print("No valid feature windows created!")
        return None
        
    df = pd.DataFrame(rows)
    print(f"Created {len(df)} feature samples (with overlapping windows)")
    print(f"State distribution: {df['state'].value_counts().to_dict()}")
    
    return df

def train_model_with_tuning(X_train, y_train, X_val, y_val, model_type='xgboost'):
    """Train model with hyperparameter tuning."""
    print(f"\nTraining {model_type} with hyperparameter tuning...")
    
    if model_type == 'xgboost':
        base_model = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
    elif model_type == 'random_forest':
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Use smaller parameter grid for faster tuning if dataset is large
    if len(X_train) > 10000:
        if model_type == 'xgboost':
            param_grid = {
                'n_estimators': [200],
                'max_depth': [4, 5],
                'learning_rate': [0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.8]
            }
        elif model_type == 'random_forest':
            param_grid = {
                'n_estimators': [200],
                'max_depth': [10, 15],
                'min_samples_split': [5],
                'min_samples_leaf': [2],
                'max_features': ['sqrt']
            }
    
    # Grid search with cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring='accuracy',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate on validation set
    best_model = grid_search.best_estimator_
    val_score = best_model.score(X_val, y_val)
    print(f"Validation score: {val_score:.4f}")
    
    return best_model, grid_search.best_params_

def main(args):
    print("=== Improved P300 Cognitive State Classifier Training ===")
    
    # Build dataset with overlapping windows
    df = build_dataset(args.p300_csv, args.state_csv, 
                      window_s=args.window, 
                      state_confidence_thresh=args.confidence_thresh,
                      overlap=args.overlap)
    
    if df is None or len(df) < 50:
        print("Insufficient data for training!")
        return
    
    # Prepare features and labels
    df = df.dropna()
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['state'])
    
    # Drop non-feature columns
    feature_cols = [col for col in df.columns if col not in ['ts', 'state', 'confidence', 'window_offset']]
    X = df[feature_cols].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Classes: {le.classes_}")
    
    # Check for class imbalance
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(le.inverse_transform(unique), counts))
    print(f"Class distribution: {class_dist}")
    
    # Time-based split to avoid data leakage
    df_sorted = df.sort_values('ts')
    split_idx = int(len(df_sorted) * 0.8)
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    X_train = train_df[feature_cols].values
    y_train = le.transform(train_df['state'])
    X_test = test_df[feature_cols].values
    y_test = le.transform(test_df['state'])
    
    # Further split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {}
    
    if args.model_type in ['xgboost', 'both']:
        xgb_model, xgb_params = train_model_with_tuning(
            X_train_scaled, y_train, X_val_scaled, y_val, 'xgboost'
        )
        models['xgboost'] = (xgb_model, xgb_params)
    
    if args.model_type in ['random_forest', 'both']:
        rf_model, rf_params = train_model_with_tuning(
            X_train_scaled, y_train, X_val_scaled, y_val, 'random_forest'
        )
        models['random_forest'] = (rf_model, rf_params)
    
    # Evaluate models and select best
    best_model = None
    best_score = 0
    best_name = None
    
    print("\n=== Model Comparison ===")
    for name, (model, params) in models.items():
        test_score = model.score(X_test_scaled, y_test)
        y_pred = model.predict(X_test_scaled)
        
        print(f"\n{name.upper()} Results:")
        print(f"Test Accuracy: {test_score:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} (Accuracy: {best_score:.4f})")
    
    # Save best model
    os.makedirs(args.out_dir, exist_ok=True)
    pipeline = {
        'model': best_model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_columns': feature_cols,
        'window_s': args.window,
        'confidence_thresh': args.confidence_thresh,
        'overlap': args.overlap,
        'accuracy': best_score,
        'model_type': best_name,
        'best_params': models[best_name][1]
    }
    
    model_path = os.path.join(args.out_dir, 'p300_improved_pipeline.joblib')
    joblib.dump(pipeline, model_path)
    print(f"\nBest model saved to: {model_path}")
    
    # Save detailed info
    info_path = os.path.join(args.out_dir, 'improved_model_info.txt')
    with open(info_path, 'w') as f:
        f.write("=== Improved P300 Classifier Info ===\n\n")
        f.write(f"Best model: {best_name}\n")
        f.write(f"Test accuracy: {best_score:.4f}\n")
        f.write(f"Feature count: {len(feature_cols)}\n")
        f.write(f"Classes: {list(le.classes_)}\n")
        f.write(f"Window size: {args.window}s\n")
        f.write(f"Overlap: {args.overlap}\n")
        f.write(f"Confidence threshold: {args.confidence_thresh}\n")
        f.write(f"Best parameters: {models[best_name][1]}\n\n")
        f.write("Feature columns:\n")
        for i, col in enumerate(feature_cols):
            f.write(f"{i}: {col}\n")
    
    print(f"Model info saved to: {info_path}")
    print("\nImproved training completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train improved P300 cognitive state classifier")
    parser.add_argument('--p300-csv', type=str, default='src/logs/p300_stream.csv',
                       help='Path to P300 stream CSV file')
    parser.add_argument('--state-csv', type=str, default='src/logs/state_stream.csv',
                       help='Path to state stream CSV file')
    parser.add_argument('--window', type=float, default=2.0,
                       help='Time window size in seconds for feature extraction')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap ratio (0.0 to 0.9)')
    parser.add_argument('--confidence-thresh', type=float, default=0.7,
                       help='Minimum confidence threshold for state labels')
    parser.add_argument('--model-type', type=str, default='both', 
                       choices=['xgboost', 'random_forest', 'both'],
                       help='Model type to train')
    parser.add_argument('--out-dir', type=str, default='models',
                       help='Output directory for trained model')
    
    args = parser.parse_args()
    main(args)