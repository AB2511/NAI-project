#!/usr/bin/env python3
"""
Complete LOSO Training Pipeline
Reproduces all results from the NAI project in a single script.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {description} completed")
        if result.stdout:
            print(f"Output: {result.stdout}")

def main():
    parser = argparse.ArgumentParser(description="Run complete LOSO training pipeline")
    parser.add_argument("--data-dir", default="data/processed_p3b", help="Processed data directory")
    parser.add_argument("--results-dir", default="results", help="Results output directory")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing step")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("figures/final", exist_ok=True)
    
    print("🧠 NAI Project - Complete LOSO Training Pipeline")
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Device: {args.device}")
    
    # Step 1: Preprocess data (if needed)
    if not args.skip_preprocess:
        if not os.path.exists(args.data_dir):
            run_command(
                "python src/preprocess_p3b.py",
                "Data preprocessing"
            )
    
    # Step 2: EEGNet LOSO training
    run_command(
        f"python src/eegnet_v5_loso.py --data-dir {args.data_dir} "
        f"--out-dir {args.results_dir}/eegnet_loso --device {args.device} --epochs {args.epochs}",
        "EEGNet LOSO cross-validation"
    )
    
    # Step 3: Traditional ML baselines
    run_command(
        f"python src/pca_baselines_p3b.py --data-dir {args.data_dir} "
        f"--out-dir {args.results_dir}/ml_baselines",
        "Traditional ML baselines"
    )
    
    # Step 4: Generate summary statistics
    run_command(
        f"python calculate_loso_summary.py --results-dir {args.results_dir}",
        "Summary statistics calculation"
    )
    
    # Step 5: Generate final figures
    run_command(
        f"python src/paper_analysis_p3b.py --data-dir {args.data_dir} "
        f"--out-dir figures/final",
        "Publication figures generation"
    )
    
    print("\n🎉 PIPELINE COMPLETE!")
    print(f"Results saved to: {args.results_dir}")
    print(f"Figures saved to: figures/final")
    print("\nKey outputs:")
    print(f"- LOSO summary: {args.results_dir}/eegnet_loso/eegnet_loso_summary.json")
    print(f"- ML comparison: {args.results_dir}/ml_baselines/loso_baselines_summary.json")
    print(f"- Final figures: figures/final/")

if __name__ == "__main__":
    main()