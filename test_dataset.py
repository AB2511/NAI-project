#!/usr/bin/env python3
"""
Test P300 Dataset Loading
"""

import sys
from pathlib import Path

try:
    import mne
    print("✅ MNE imported successfully")
except ImportError as e:
    print("❌ MNE import failed:", e)
    print("Install with: pip install mne")
    sys.exit(1)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths
data_path = Path("NAI-project/ds003061/sub-001/eeg")

runs = [
    data_path / "sub-001_task-P300_run-1_eeg.set",
    data_path / "sub-001_task-P300_run-2_eeg.set",
    data_path / "sub-001_task-P300_run-3_eeg.set",
]

print("Checking for dataset files...")
for run_file in runs:
    if run_file.exists():
        print(f"✅ Found: {run_file}")
    else:
        print(f"❌ Missing: {run_file}")

# Try to load if files exist
if all(run_file.exists() for run_file in runs):
    print("\nLoading dataset...")
    raw_list = []
    for run_file in runs:
        print("Loading:", run_file)
        raw = mne.io.read_raw_eeglab(run_file, preload=True)
        raw_list.append(raw)
    
    # Concatenate all runs
    raw = mne.concatenate_raws(raw_list)
    print("\n✅ Dataset loaded successfully!")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    print(f"Channel names: {raw.ch_names[:10]}")
    
    # Check for events
    try:
        events = mne.find_events(raw)
        print(f"Events found: {len(events)}")
    except ValueError:
        # Try annotations instead
        events, event_id = mne.events_from_annotations(raw)
        print(f"Events from annotations: {len(events)}")
        print(f"Event types: {event_id}")
    
else:
    print("\n❌ Dataset files not found.")
    print("Please download the OpenNeuro P300 dataset (ds003061).")
    print("Expected location: NAI-project/ds003061/sub-001/eeg/")
    print("\nTo download:")
    print("1. Go to: https://openneuro.org/datasets/ds003061")
    print("2. Download the dataset")
    print("3. Extract to NAI-project/ds003061/")