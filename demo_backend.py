#!/usr/bin/env python3
"""
Demo Backend for NAI Dashboard
Generates synthetic EEG data and P300 responses for demonstration
"""

import time
import json
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class DemoDataGenerator:
    def __init__(self):
        self.running = False
        self.start_time = time.time()
        self.sample_count = 0
        self.p300_count = 0
        self.state_count = 0
        
        # Synthetic data parameters
        self.fs = 256  # Sampling rate
        self.n_channels = 26
        self.current_state = "focused"
        self.p300_amplitude = 0.0
        
    def generate_eeg_sample(self):
        """Generate synthetic EEG sample"""
        # Base EEG noise (alpha, beta, theta components)
        t = self.sample_count / self.fs
        
        # Alpha rhythm (8-12 Hz)
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)
        
        # Beta activity (13-30 Hz) 
        beta = 0.3 * np.sin(2 * np.pi * 20 * t)
        
        # Theta (4-8 Hz)
        theta = 0.2 * np.sin(2 * np.pi * 6 * t)
        
        # Random noise
        noise = np.random.normal(0, 0.1, self.n_channels)
        
        # Combine components
        base_signal = alpha + beta + theta
        eeg_sample = base_signal + noise
        
        # Add P300 component if active
        if self.p300_amplitude > 0:
            # P300 is strongest at Pz (channel ~20)
            eeg_sample[20] += self.p300_amplitude
            self.p300_amplitude *= 0.95  # Decay
            
        self.sample_count += 1
        return eeg_sample.tolist()
    
    def trigger_p300(self):
        """Trigger a synthetic P300 response"""
        self.p300_amplitude = 2.0  # Strong positive deflection
        self.p300_count += 1
        
    def update_state(self):
        """Update cognitive state randomly"""
        states = ["focused", "distracted", "overloaded", "relaxed"]
        self.current_state = np.random.choice(states)
        self.state_count += 1

# Global demo generator
demo_gen = DemoDataGenerator()

@app.route('/status')
def status():
    """Backend status endpoint"""
    uptime = time.time() - demo_gen.start_time
    return jsonify({
        "status": "running",
        "uptime": uptime,
        "samples_processed": demo_gen.sample_count,
        "p300_samples": demo_gen.p300_count,
        "state_samples": demo_gen.state_count,
        "ml_model_loaded": True,
        "current_state": demo_gen.current_state
    })

@app.route('/eeg/latest')
def latest_eeg():
    """Latest EEG sample"""
    sample = demo_gen.generate_eeg_sample()
    return jsonify({
        "timestamp": time.time(),
        "channels": sample,
        "fs": demo_gen.fs
    })

@app.route('/p300/latest')
def latest_p300():
    """Latest P300 data"""
    # Randomly trigger P300 events
    if np.random.random() < 0.1:  # 10% chance
        demo_gen.trigger_p300()
        
    return jsonify({
        "timestamp": time.time(),
        "amplitude": demo_gen.p300_amplitude,
        "latency": 300 + np.random.normal(0, 50),  # ~300ms ± 50ms
        "detected": demo_gen.p300_amplitude > 0.5
    })

@app.route('/state/latest')
def latest_state():
    """Latest cognitive state"""
    # Randomly update state
    if np.random.random() < 0.05:  # 5% chance
        demo_gen.update_state()
        
    return jsonify({
        "timestamp": time.time(),
        "state": demo_gen.current_state,
        "confidence": np.random.uniform(0.7, 0.95),
        "features": {
            "alpha_power": np.random.uniform(0.3, 0.8),
            "beta_power": np.random.uniform(0.2, 0.6),
            "theta_power": np.random.uniform(0.1, 0.4)
        }
    })

@app.route('/ml/predict')
def ml_predict():
    """ML model prediction"""
    states = ["focused", "distracted", "overloaded", "relaxed"]
    probabilities = np.random.dirichlet([2, 1, 1, 1])  # Bias toward focused
    
    prediction = {
        "timestamp": time.time(),
        "predicted_state": states[np.argmax(probabilities)],
        "probabilities": {state: float(prob) for state, prob in zip(states, probabilities)},
        "model_confidence": float(np.max(probabilities))
    }
    
    return jsonify(prediction)

def run_demo_backend():
    """Run the demo backend server"""
    logger.info("🚀 Starting NAI Demo Backend")
    logger.info("📊 Generating synthetic EEG data...")
    logger.info("🧠 P300 detection active")
    logger.info("🎯 Cognitive state classification running")
    logger.info("🌐 Backend available at: http://127.0.0.1:8765")
    
    demo_gen.running = True
    app.run(host='127.0.0.1', port=8765, debug=False)

if __name__ == "__main__":
    run_demo_backend()