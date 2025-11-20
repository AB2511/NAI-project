#!/usr/bin/env python3
"""
Demo script to train CNN classifier on P300 data.
Run this after collecting some data with demo_ml_streams.py
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Check if data files exist
    p300_csv = "src/logs/p300_stream.csv"
    state_csv = "src/logs/state_stream.csv"
    
    if not os.path.exists(p300_csv):
        print(f"❌ P300 data file not found: {p300_csv}")
        print("Run demo_ml_streams.py first to collect data")
        return 1
    
    if not os.path.exists(state_csv):
        print(f"❌ State data file not found: {state_csv}")
        print("Run demo_ml_streams.py first to collect data")
        return 1
    
    print("🔍 Found data files:")
    print(f"  P300: {p300_csv}")
    print(f"  State: {state_csv}")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train CNN classifier
    print("\n🚀 Training CNN classifier...")
    cmd = [
        sys.executable, "src/inference/train_cnn_classifier.py",
        "--p300-csv", p300_csv,
        "--state-csv", state_csv,
        "--out-dir", "models",
        "--export-torchscript",
        "--epochs", "30",  # Reduced for demo
        "--batch-size", "32"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ CNN training completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ CNN training failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return 1
    
    # Check if models were created
    expected_files = [
        "models/p300_cnn_pipeline.pt",
        "models/p300_cnn_stateful.pth", 
        "models/p300_cnn_label_encoder.joblib"
    ]
    
    print("\n📁 Checking generated files:")
    for file_path in expected_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ {file_path} (missing)")
    
    # Evaluate the model
    print("\n📊 Evaluating CNN model...")
    eval_cmd = [
        sys.executable, "src/inference/evaluate_cnn.py",
        "--script-model", "models/p300_cnn_pipeline.pt",
        "--p300-csv", p300_csv,
        "--state-csv", state_csv
    ]
    
    try:
        result = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
        print("✅ CNN evaluation completed!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ CNN evaluation failed: {e}")
        print("STDERR:", e.stderr)
    
    print("\n🎉 CNN training demo completed!")
    print("You can now restart persistent_server.py to use the new CNN model.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())