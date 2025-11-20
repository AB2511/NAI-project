#!/usr/bin/env python3
"""
Create Synthetic P300 Dataset for Testing
"""

import numpy as np
import mne
from pathlib import Path
import os

def create_synthetic_p300_data():
    """Create synthetic P300 EEG data"""
    
    # Parameters
    sfreq = 256  # Sampling frequency
    n_channels = 32
    duration = 600  # 10 minutes per run
    n_runs = 3
    
    # Channel names (standard 10-20 system)
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3',
        'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF8'
    ]
    
    # Create directory
    data_dir = Path("NAI-project/ds003061/sub-001/eeg")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for run_idx in range(1, n_runs + 1):
        print(f"Creating synthetic run {run_idx}...")
        
        # Generate baseline EEG
        n_samples = int(duration * sfreq)
        data = np.random.randn(n_channels, n_samples) * 10  # 10 µV noise
        
        # Add realistic frequency components
        t = np.linspace(0, duration, n_samples)
        
        # Alpha rhythm (8-12 Hz) - stronger in posterior
        alpha_freq = 10 + np.random.randn() * 1
        alpha = 15 * np.sin(2 * np.pi * alpha_freq * t)
        data[22:25, :] += alpha  # Pz, P4, P8
        
        # Beta activity (15-25 Hz) - central
        beta_freq = 20 + np.random.randn() * 3
        beta = 8 * np.sin(2 * np.pi * beta_freq * t)
        data[12:16, :] += beta  # C3, Cz, C4, T8
        
        # Create info structure
        info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        
        # Create raw object
        raw = mne.io.RawArray(data, info)
        
        # Add standard montage
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        # Generate P300 events
        events = []
        event_times = np.arange(2, duration-2, 1.5)  # Every 1.5 seconds
        
        for i, event_time in enumerate(event_times):
            event_sample = int(event_time * sfreq)
            
            # 20% targets, 80% non-targets
            if np.random.random() < 0.2:
                event_id = 2  # Target
                # Add P300 response
                p300_latency = 0.35  # 350ms
                p300_amplitude = 8 + np.random.randn() * 2  # 8±2 µV
                p300_width = 0.1  # 100ms width
                
                # P300 time course (Gaussian)
                p300_start = int((event_time + p300_latency - p300_width) * sfreq)
                p300_end = int((event_time + p300_latency + p300_width) * sfreq)
                p300_samples = np.arange(p300_start, p300_end)
                
                if p300_end < n_samples:
                    p300_times = p300_samples / sfreq - (event_time + p300_latency)
                    p300_response = p300_amplitude * np.exp(-0.5 * (p300_times / (p300_width/3))**2)
                    
                    # Add to central/parietal channels
                    data[13, p300_samples] += p300_response  # Cz
                    data[22, p300_samples] += p300_response * 0.8  # Pz
                    data[12, p300_samples] += p300_response * 0.6  # C3
                    data[14, p300_samples] += p300_response * 0.6  # C4
                    
            else:
                event_id = 1  # Non-target
                
            events.append([event_sample, 0, event_id])
            
        events = np.array(events)
        
        # Add events to raw
        annotations = mne.annotations_from_events(events, sfreq, 
                                                 event_desc={1: 'nontarget', 2: 'target'})
        raw.set_annotations(annotations)
        
        # Save as EEGLAB format
        filename = data_dir / f"sub-001_task-P300_run-{run_idx}_eeg.set"
        raw.export(filename, fmt='eeglab', overwrite=True)
        
        print(f"✅ Created: {filename}")
        print(f"   Events: {len(events)} ({np.sum(events[:, 2] == 2)} targets)")
        
    print(f"\n✅ Synthetic P300 dataset created in {data_dir}")
    return True

if __name__ == "__main__":
    create_synthetic_p300_data()