#!/usr/bin/env python3
"""
Simple HTTP Backend for NAI Dashboard
Provides synthetic data without external dependencies
"""

import json
import time
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading

class DemoDataGenerator:
    def __init__(self):
        self.start_time = time.time()
        self.sample_count = 0
        self.p300_count = 0
        self.state_count = 0
        self.current_state = "focused"
        self.p300_amplitude = 0.0
        
    def get_status(self):
        uptime = time.time() - self.start_time
        
        # Generate P300 data
        p300_data = self.get_p300_data()
        
        # Generate state data  
        state_data = self.get_state_data()
        
        # Generate ML prediction
        ml_pred = self.get_ml_prediction()
        
        # Update counters
        self.sample_count += 1
        
        return {
            "status": "running",
            "last_update": time.time(),
            "uptime_s": uptime,
            "samples_total": self.sample_count,
            "p300_count": self.p300_count,
            "state_count": self.state_count,
            
            # P300 data
            "latest_p300": {
                "amplitude_uv": p300_data["amplitude"],
                "latency_ms": p300_data["latency"],
                "smoothed_amp_uv": p300_data["amplitude"] * 0.8,  # Smoothed version
                "fatigue_index": min(0.9, uptime / 300.0)  # Increases over time
            },
            
            # State data
            "latest_state": {
                "state": state_data["state"],
                "confidence": state_data["confidence"],
                "processing_latency_ms": np.random.uniform(15, 25)
            },
            
            # ML prediction
            "ml_model_loaded": True,
            "ml_prediction": {
                "state": ml_pred["predicted_state"],
                "confidence": ml_pred["model_confidence"],
                "class_probabilities": ml_pred["probabilities"]
            },
            
            # ML status
            "ml_status": {
                "model_loaded": True,
                "buffer_size": 256,
                "window_size_s": 1.0,
                "classes": ["focused", "distracted", "overloaded", "relaxed"]
            },
            
            # Raw timeseries data (last 10 samples)
            "raw_p300_last": [
                {
                    "ts": time.time() - i,
                    "amplitude_uv": float(np.random.uniform(0.5, 3.0)),
                    "smoothed_amp_uv": float(np.random.uniform(0.4, 2.5))
                } for i in range(10, 0, -1)
            ],
            
            "raw_state_last": [
                {
                    "ts": time.time() - i * 2,
                    "state": np.random.choice(["focused", "distracted", "overloaded", "relaxed"]),
                    "confidence": float(np.random.uniform(0.7, 0.95))
                } for i in range(12, 0, -1)
            ]
        }
    
    def get_p300_data(self):
        # Randomly trigger P300 events
        if np.random.random() < 0.1:
            self.p300_amplitude = 2.0
            self.p300_count += 1
        else:
            self.p300_amplitude *= 0.9  # Decay
            
        return {
            "timestamp": time.time(),
            "amplitude": float(self.p300_amplitude),
            "latency": float(300 + np.random.normal(0, 50)),
            "detected": self.p300_amplitude > 0.5
        }
    
    def get_state_data(self):
        # Randomly update state
        if np.random.random() < 0.05:
            states = ["focused", "distracted", "overloaded", "relaxed"]
            self.current_state = np.random.choice(states)
            self.state_count += 1
            
        return {
            "timestamp": time.time(),
            "state": self.current_state,
            "confidence": float(np.random.uniform(0.7, 0.95)),
            "features": {
                "alpha_power": float(np.random.uniform(0.3, 0.8)),
                "beta_power": float(np.random.uniform(0.2, 0.6)),
                "theta_power": float(np.random.uniform(0.1, 0.4))
            }
        }
    
    def get_ml_prediction(self):
        states = ["focused", "distracted", "overloaded", "relaxed"]
        probabilities = np.random.dirichlet([2, 1, 1, 1])
        
        return {
            "timestamp": time.time(),
            "predicted_state": states[np.argmax(probabilities)],
            "probabilities": {state: float(prob) for state, prob in zip(states, probabilities)},
            "model_confidence": float(np.max(probabilities))
        }

# Global data generator
demo_gen = DemoDataGenerator()

class NAIRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        # Handle CORS
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        
        # Route requests
        if path == '/status':
            response = demo_gen.get_status()
        elif path == '/p300/latest':
            response = demo_gen.get_p300_data()
        elif path == '/state/latest':
            response = demo_gen.get_state_data()
        elif path == '/ml/predict':
            response = demo_gen.get_ml_prediction()
        else:
            response = {"error": "Not found", "path": path}
        
        # Send JSON response
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass

def run_server():
    server_address = ('127.0.0.1', 8765)
    httpd = HTTPServer(server_address, NAIRequestHandler)
    print(f"🚀 NAI Backend Server running at http://127.0.0.1:8765")
    print("📊 Providing synthetic EEG data for dashboard")
    print("🧠 P300 detection and state classification active")
    print("Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        httpd.server_close()

if __name__ == "__main__":
    run_server()