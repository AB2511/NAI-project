#!/usr/bin/env python3
"""
Run ML training pipeline for P300 cognitive state classification.
"""

import os
import sys
import subprocess
import argparse

def check_data_files(p300_csv, state_csv):
    """Check if data files exist and have content."""
    if not os.path.exists(p300_csv):
        print(f"Error: P300 CSV file not found: {p300_csv}")
        return False
    
    if not os.path.exists(state_csv):
        print(f"Error: State CSV file not found: {state_csv}")
        return False
    
    # Check file sizes
    p300_size = os.path.getsize(p300_csv)
    state_size = os.path.getsize(state_csv)
    
    if p300_size < 1000:  # Less than 1KB
        print(f"Warning: P300 CSV file is very small ({p300_size} bytes)")
    
    if state_size < 500:  # Less than 0.5KB
        print(f"Warning: State CSV file is very small ({state_size} bytes)")
    
    print(f"Data files found:")
    print(f"  P300: {p300_csv} ({p300_size} bytes)")
    print(f"  State: {state_csv} ({state_size} bytes)")
    
    return True

def install_requirements():
    """Install required packages for ML training."""
    required_packages = [
        'scikit-learn',
        'xgboost',
        'pandas',
        'numpy',
        'scipy',
        'joblib'
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            return False
    
    print("All packages installed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run P300 ML training pipeline")
    parser.add_argument('--p300-csv', type=str, default='src/logs/p300_stream.csv',
                       help='Path to P300 stream CSV file')
    parser.add_argument('--state-csv', type=str, default='src/logs/state_stream.csv',
                       help='Path to state stream CSV file')
    parser.add_argument('--window', type=float, default=2.0,
                       help='Time window size in seconds')
    parser.add_argument('--confidence-thresh', type=float, default=0.5,
                       help='Minimum confidence threshold for state labels')
    parser.add_argument('--out-dir', type=str, default='models',
                       help='Output directory for trained model')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install required dependencies')
    
    args = parser.parse_args()
    
    # Change to NAI-project directory if not already there
    if not os.path.exists('train_classifier.py'):
        if os.path.exists('NAI-project/train_classifier.py'):
            os.chdir('NAI-project')
        else:
            print("Error: Could not find train_classifier.py")
            return 1
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_requirements():
            return 1
    
    # Check data files
    if not check_data_files(args.p300_csv, args.state_csv):
        print("\nTo collect training data, run the system first:")
        print("  python run_nai_system.py")
        print("  # Let it run for a few minutes to collect data")
        print("  # Then run this training script again")
        return 1
    
    # Run training
    print(f"\nStarting ML training...")
    print(f"Window size: {args.window}s")
    print(f"Confidence threshold: {args.confidence_thresh}")
    print(f"Output directory: {args.out_dir}")
    
    cmd = [
        sys.executable, 'train_classifier.py',
        '--p300-csv', args.p300_csv,
        '--state-csv', args.state_csv,
        '--window', str(args.window),
        '--confidence-thresh', str(args.confidence_thresh),
        '--out-dir', args.out_dir
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Training completed successfully!")
        
        model_path = os.path.join(args.out_dir, 'p300_xgb_pipeline.joblib')
        if os.path.exists(model_path):
            print(f"✅ Model saved to: {model_path}")
            print("\nTo use the trained model:")
            print(f"  export NAI_ML_MODEL_PATH={model_path}")
            print("  python src/inference/persistent_server.py")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return 1

if __name__ == "__main__":
    sys.exit(main())