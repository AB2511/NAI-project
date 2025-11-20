"""
Simple, robust topomap pipeline for ERP CORE P3b
Handles electrode position issues gracefully
"""

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_subject_data(sub_dir):
    """Load and preprocess subject data"""
    eeg_dir = sub_dir / 'eeg'
    set_files = list(eeg_dir.glob('*.set'))
    if not set_files:
        raise FileNotFoundError(f'No .set file found in {eeg_dir}')
    
    # Load raw data
    raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)
    
    # Remove problematic channels
    bad_channels = ['HEOG_left', 'HEOG_right', 'VEOG_lower', 'VEOG_upper', 'FP1', 'FP2']
    channels_to_drop = [ch for ch in bad_channels if ch in raw.ch_names]
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)
    
    # Load events
    events_tsv = list(eeg_dir.glob('*events.tsv'))[0]
    df = pd.read_csv(events_tsv, sep='\t')
    stim_events = df[df["trial_type"] == "stimulus"].copy()
    
    # Create events array
    events = np.zeros((len(stim_events), 3), dtype=int)
    events[:, 0] = (stim_events["onset"].values * raw.info["sfreq"]).astype(int)
    event_codes = stim_events["value"].values.astype(int)
    events[:, 2] = np.where(event_codes == 12, 1, 2)  # 1=target, 2=standard
    
    return raw, events

def compute_evokeds(raw, events):
    """Compute evoked responses"""
    # Create epochs
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    epochs = mne.Epochs(
        raw, events, 
        event_id={"target": 1, "standard": 2}, 
        tmin=-0.2, tmax=0.8,
        baseline=(-0.2, 0.0), 
        preload=True, 
        picks=picks, 
        verbose=False
    )
    
    # Re-reference
    epochs.set_eeg_reference('average', verbose=False)
    
    # Compute evokeds
    ev_target = epochs["target"].average()
    ev_standard = epochs["standard"].average()
    ev_diff = mne.combine_evoked([ev_target, ev_standard], weights=[1, -1])
    
    return ev_target, ev_standard, ev_diff

def find_p3b_peak(evoked, tmin=0.3, tmax=0.5):
    """Find P3b peak time in parietal channels"""
    # Find parietal channels
    parietal_chs = ['Pz', 'CPz', 'Cz', 'P3', 'P4', 'CP1', 'CP2']
    available_chs = [ch for ch in parietal_chs if ch in evoked.ch_names]
    
    if not available_chs:
        # Fallback to all channels
        available_chs = evoked.ch_names
    
    # Get indices
    ch_indices = [evoked.ch_names.index(ch) for ch in available_chs]
    
    # Find peak in time window
    times = evoked.times
    start_idx, end_idx = times.searchsorted([tmin, tmax])
    
    # Average across channels and find peak
    data_window = evoked.data[ch_indices, start_idx:end_idx]
    mean_data = np.abs(data_window).mean(axis=0)
    peak_idx = mean_data.argmax()
    
    return times[start_idx + peak_idx]

def create_simple_topomap(evoked, time_point, title, save_path):
    """Create a simple topomap without montage issues"""
    try:
        # Try with standard montage first
        montage = mne.channels.make_standard_montage('biosemi64')
        evoked_copy = evoked.copy()
        evoked_copy.set_montage(montage, on_missing='ignore')
        
        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
        
        # Get data at time point
        time_idx = evoked_copy.time_as_index(time_point)
        data_at_time = evoked_copy.data[:, time_idx]
        
        # Ensure data is 1D
        if data_at_time.ndim > 1:
            data_at_time = data_at_time.flatten()
        
        # Plot topomap
        mne.viz.plot_topomap(
            data_at_time, 
            evoked_copy.info,
            axes=ax,
            show=False
        )
        
        ax.set_title(title, fontsize=12)
        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not create topomap for {title}: {e}")
        return False

def main():
    data_root = Path("data/erp_core_p3b")
    out_dir = Path("results/simple_topomaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    subjects = sorted([d for d in data_root.iterdir() if d.name.startswith('sub-')])
    
    all_targets = []
    all_standards = []
    all_diffs = []
    
    print(f"Processing {len(subjects)} subjects...")
    
    for sub_dir in subjects:
        print(f"Processing {sub_dir.name}...")
        
        try:
            # Load data
            raw, events = load_subject_data(sub_dir)
            ev_target, ev_standard, ev_diff = compute_evokeds(raw, events)
            
            # Find P3b peak
            peak_time = find_p3b_peak(ev_diff)
            
            # Create subject output directory
            sub_out = out_dir / sub_dir.name
            sub_out.mkdir(exist_ok=True)
            
            # Create topomaps
            create_simple_topomap(
                ev_diff, peak_time,
                f"{sub_dir.name} Difference @ {int(peak_time*1000)}ms",
                sub_out / f"{sub_dir.name}_diff_peak.png"
            )
            
            create_simple_topomap(
                ev_target, peak_time,
                f"{sub_dir.name} Target @ {int(peak_time*1000)}ms",
                sub_out / f"{sub_dir.name}_target_peak.png"
            )
            
            create_simple_topomap(
                ev_standard, peak_time,
                f"{sub_dir.name} Standard @ {int(peak_time*1000)}ms",
                sub_out / f"{sub_dir.name}_standard_peak.png"
            )
            
            # Store for group average
            all_targets.append(ev_target)
            all_standards.append(ev_standard)
            all_diffs.append(ev_diff)
            
        except Exception as e:
            print(f"Error processing {sub_dir.name}: {e}")
    
    # Create group averages
    if all_diffs:
        print("Creating group averages...")
        
        ga_target = mne.grand_average(all_targets)
        ga_standard = mne.grand_average(all_standards)
        ga_diff = mne.grand_average(all_diffs)
        
        # Find group peak
        group_peak = find_p3b_peak(ga_diff)
        print(f"Group P3b peak at {group_peak:.3f}s ({int(group_peak*1000)}ms)")
        
        # Create group output directory
        group_out = out_dir / "group"
        group_out.mkdir(exist_ok=True)
        
        # Create group topomaps
        create_simple_topomap(
            ga_diff, group_peak,
            f"Group Difference @ {int(group_peak*1000)}ms",
            group_out / f"group_diff_{int(group_peak*1000)}ms.png"
        )
        
        create_simple_topomap(
            ga_target, group_peak,
            f"Group Target @ {int(group_peak*1000)}ms",
            group_out / f"group_target_{int(group_peak*1000)}ms.png"
        )
        
        create_simple_topomap(
            ga_standard, group_peak,
            f"Group Standard @ {int(group_peak*1000)}ms",
            group_out / f"group_standard_{int(group_peak*1000)}ms.png"
        )
        
        # Also create fixed time windows
        for t_ms in [300, 400]:
            t_s = t_ms / 1000.0
            create_simple_topomap(
                ga_diff, t_s,
                f"Group Difference @ {t_ms}ms",
                group_out / f"group_diff_{t_ms}ms.png"
            )
    
    print(f"Done! Topomaps saved to {out_dir}")

if __name__ == "__main__":
    main()