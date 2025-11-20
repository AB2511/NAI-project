# src/inference/ml_inference.py
"""
ML inference module for real-time P300 cognitive state prediction.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import skew, kurtosis
import logging

log = logging.getLogger("ml_inference")

class P300MLPredictor:
    """Real-time ML predictor for P300 cognitive states."""
    
    def __init__(self, model_path=None, buffer_size=512):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        self.window_s = 2.0
        self.confidence_thresh = 0.5
        
        # Rolling buffer for P300 samples
        self.p300_buffer = deque(maxlen=buffer_size)
        
        # Prediction state
        self.last_prediction = None
        self.last_prediction_time = None
        self.prediction_confidence = 0.0
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained model pipeline."""
        try:
            pipeline = joblib.load(model_path)
            self.model = pipeline['model']
            self.scaler = pipeline['scaler']
            self.label_encoder = pipeline['label_encoder']
            self.feature_columns = pipeline.get('feature_columns', [])
            self.window_s = pipeline.get('window_s', 2.0)
            self.confidence_thresh = pipeline.get('confidence_thresh', 0.5)
            
            log.info(f"Loaded ML model from {model_path}")
            log.info(f"Classes: {list(self.label_encoder.classes_)}")
            log.info(f"Window size: {self.window_s}s")
            
            return True
        except Exception as e:
            log.error(f"Failed to load model from {model_path}: {e}")
            return False
    
    def is_ready(self):
        """Check if predictor is ready for inference."""
        return (self.model is not None and 
                self.scaler is not None and 
                self.label_encoder is not None)
    
    def add_p300_sample(self, sample):
        """Add P300 sample to rolling buffer."""
        if not isinstance(sample, dict):
            return
        
        # Ensure required fields
        required_fields = ['ts', 'amplitude_uv', 'latency_ms', 'smoothed_amp_uv', 'fatigue_index']
        if not all(field in sample for field in required_fields):
            return
        
        self.p300_buffer.append(sample)
    
    def extract_window_features(self, window_end_time=None):
        """Extract features from current buffer window."""
        if not self.p300_buffer:
            return None
        
        if window_end_time is None:
            window_end_time = self.p300_buffer[-1]['ts']
        
        window_start_time = window_end_time - self.window_s
        
        # Filter samples in window
        window_samples = [
            s for s in self.p300_buffer 
            if window_start_time <= s['ts'] <= window_end_time
        ]
        
        if len(window_samples) < 2:
            return None
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(window_samples)
        
        features = {}
        feature_cols = ['amplitude_uv', 'latency_ms', 'smoothed_amp_uv', 'fatigue_index']
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            arr = df[col].values.astype(float)
            if len(arr) == 0:
                continue
            
            # Basic statistics
            features[f'{col}_mean'] = arr.mean()
            features[f'{col}_std'] = arr.std(ddof=0) if len(arr) > 1 else 0.0
            features[f'{col}_min'] = arr.min()
            features[f'{col}_max'] = arr.max()
            features[f'{col}_median'] = np.median(arr)
            features[f'{col}_skew'] = skew(arr) if len(arr) > 2 else 0.0
            features[f'{col}_kurtosis'] = kurtosis(arr) if len(arr) > 3 else 0.0
            features[f'{col}_last'] = arr[-1]
            features[f'{col}_count'] = len(arr)
            
            # Trend
            if len(arr) > 1:
                features[f'{col}_slope'] = (arr[-1] - arr[0]) / (len(arr) - 1)
            else:
                features[f'{col}_slope'] = 0.0
            
            # Range and percentiles
            features[f'{col}_range'] = arr.max() - arr.min()
            features[f'{col}_p25'] = np.percentile(arr, 25)
            features[f'{col}_p75'] = np.percentile(arr, 75)
        
        return features
    
    def predict(self, window_end_time=None):
        """Make prediction from current buffer state."""
        if not self.is_ready():
            return None
        
        # Extract features
        features = self.extract_window_features(window_end_time)
        if features is None:
            return None
        
        try:
            # Convert to DataFrame with correct column order
            if self.feature_columns:
                # Use saved feature column order
                feature_vector = []
                for col in self.feature_columns:
                    feature_vector.append(features.get(col, 0.0))
                X = np.array(feature_vector).reshape(1, -1)
            else:
                # Fallback: use all features
                X = pd.DataFrame([features]).values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict
            pred_class = self.model.predict(X_scaled)[0]
            pred_proba = self.model.predict_proba(X_scaled)[0]
            
            # Get class name and confidence
            pred_label = self.label_encoder.inverse_transform([pred_class])[0]
            confidence = float(pred_proba.max())
            
            # Update state
            self.last_prediction = pred_label
            self.last_prediction_time = time.time()
            self.prediction_confidence = confidence
            
            result = {
                'state': pred_label,
                'confidence': confidence,
                'timestamp': self.last_prediction_time,
                'class_probabilities': {
                    self.label_encoder.classes_[i]: float(prob) 
                    for i, prob in enumerate(pred_proba)
                }
            }
            
            return result
            
        except Exception as e:
            log.error(f"Prediction error: {e}")
            return None
    
    def get_status(self):
        """Get current predictor status."""
        return {
            'model_loaded': self.is_ready(),
            'buffer_size': len(self.p300_buffer),
            'window_size_s': self.window_s,
            'last_prediction': self.last_prediction,
            'last_prediction_time': self.last_prediction_time,
            'prediction_confidence': self.prediction_confidence,
            'classes': list(self.label_encoder.classes_) if self.label_encoder else []
        }