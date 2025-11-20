# run_ml_inference.py
"""
Real-time ML inference demo using the persistent server and trained model.
"""

import os
import time
import requests
import joblib
import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import skew, kurtosis

class RealTimeP300Classifier:
    """Real-time P300 cognitive state classifier."""
    
    def __init__(self, model_path, server_url="http://127.0.0.1:8765", window_s=2.0):
        self.server_url = server_url
        self.window_s = window_s
        self.p300_buffer = deque(maxlen=1000)
        
        # Load model
        self.pipeline = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        
        if os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Model not found: {model_path}")
    
    def load_model(self, model_path):
        """Load trained model pipeline."""
        try:
            self.pipeline = joblib.load(model_path)
            self.model = self.pipeline['model']
            self.scaler = self.pipeline['scaler']
            self.label_encoder = self.pipeline['label_encoder']
            self.feature_columns = self.pipeline.get('feature_columns', [])
            print(f"Loaded model: {model_path}")
            print(f"Classes: {list(self.label_encoder.classes_)}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def is_ready(self):
        """Check if classifier is ready."""
        return self.model is not None
    
    def get_server_data(self):
        """Get latest data from persistent server."""
        try:
            response = requests.get(f"{self.server_url}/status", timeout=2)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Server request failed: {e}")
        return None
    
    def extract_features(self, p300_samples):
        """Extract features from P300 samples."""
        if len(p300_samples) < 3:
            return None
        
        df = pd.DataFrame(p300_samples)
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
    
    def predict(self, features):
        """Make prediction from features."""
        if not self.is_ready() or features is None:
            return None
        
        try:
            # Convert to feature vector
            if self.feature_columns:
                # Use saved feature column order
                feature_vector = []
                for col in self.feature_columns:
                    feature_vector.append(features.get(col, 0.0))
                X = np.array(feature_vector).reshape(1, -1)
            else:
                # Fallback: use all features
                X = pd.DataFrame([features]).values
            
            # Scale and predict
            X_scaled = self.scaler.transform(X)
            pred_class = self.model.predict(X_scaled)[0]
            pred_proba = self.model.predict_proba(X_scaled)[0]
            
            # Get class name and confidence
            pred_label = self.label_encoder.inverse_transform([pred_class])[0]
            confidence = float(pred_proba.max())
            
            return {
                'state': pred_label,
                'confidence': confidence,
                'timestamp': time.time(),
                'class_probabilities': {
                    self.label_encoder.classes_[i]: float(prob) 
                    for i, prob in enumerate(pred_proba)
                }
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def run_inference_loop(self, interval=1.0):
        """Run continuous inference loop."""
        print("Starting real-time inference loop...")
        print(f"Server: {self.server_url}")
        print(f"Window size: {self.window_s}s")
        print(f"Update interval: {interval}s")
        print("-" * 50)
        
        last_prediction_time = 0
        
        while True:
            try:
                # Get server data
                server_data = self.get_server_data()
                if server_data is None:
                    print("No server data available")
                    time.sleep(interval)
                    continue
                
                # Check if server is running
                if server_data.get('status') != 'connected':
                    print(f"Server status: {server_data.get('status', 'unknown')}")
                    time.sleep(interval)
                    continue
                
                # Get P300 samples
                p300_samples = server_data.get('raw_p300_last', [])
                if not p300_samples:
                    print("No P300 samples available")
                    time.sleep(interval)
                    continue
                
                # Use the most recent samples (LSL timestamps are different from system time)
                recent_samples = p300_samples[-20:]  # Use last 20 samples for window
                
                if len(recent_samples) < 3:
                    print(f"Insufficient samples in window: {len(recent_samples)}")
                    time.sleep(interval)
                    continue
                
                # Extract features and predict
                features = self.extract_features(recent_samples)
                prediction = self.predict(features)
                
                if prediction:
                    # Get rule-based state for comparison
                    rule_state = server_data.get('latest_state', {})
                    
                    print(f"Time: {time.strftime('%H:%M:%S')}")
                    print(f"ML Prediction: {prediction['state']} (conf: {prediction['confidence']:.3f})")
                    if rule_state:
                        print(f"Rule-based:    {rule_state.get('state', 'N/A')} (conf: {rule_state.get('confidence', 0):.3f})")
                    print(f"Samples used:  {len(recent_samples)}")
                    
                    # Show class probabilities
                    probs = prediction['class_probabilities']
                    prob_str = " | ".join([f"{k}: {v:.3f}" for k, v in probs.items()])
                    print(f"Probabilities: {prob_str}")
                    print("-" * 50)
                    
                    last_prediction_time = time.time()
                else:
                    print("Prediction failed")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\nStopping inference loop...")
                break
            except Exception as e:
                print(f"Error in inference loop: {e}")
                time.sleep(interval)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time P300 ML inference")
    parser.add_argument('--model-path', type=str, default='models/p300_xgb_pipeline.joblib',
                       help='Path to trained model')
    parser.add_argument('--server-url', type=str, default='http://127.0.0.1:8765',
                       help='Persistent server URL')
    parser.add_argument('--window', type=float, default=2.0,
                       help='Time window for feature extraction (seconds)')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Prediction interval (seconds)')
    
    args = parser.parse_args()
    
    # Create classifier
    classifier = RealTimeP300Classifier(
        model_path=args.model_path,
        server_url=args.server_url,
        window_s=args.window
    )
    
    if not classifier.is_ready():
        print("Classifier not ready. Please train a model first.")
        return
    
    # Run inference loop
    classifier.run_inference_loop(interval=args.interval)

if __name__ == "__main__":
    main()