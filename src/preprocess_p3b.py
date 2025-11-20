import os
import mne
import numpy as np
import pandas as pd

# Path to your dataset
RAW_DIR = "data/erp_core_p3b/"
OUT_DIR = "data/processed_p3b/"
os.makedirs(OUT_DIR, exist_ok=True)

# Subjects to preprocess (001-020 for larger dataset)
subjects = [f"sub-{i:03d}" for i in range(1, 21)]  # 001–020

# Channels to drop (P3b does not need these noisy channels)
DROP_CHANNELS = [
    "FP1", "FP2", "F7", "F8",
    "VEOG_lower", "HEOG_right", "HEOG_left"
]

# Valid events for P3 (ERP CORE standard mapping)
EVENT_MAP = {
    "13": 0,   # standard
    "23": 1    # target
}

def preprocess_subject(sub):
    print(f"\n=== Processing {sub} ===")
    
    set_file = os.path.join(RAW_DIR, sub, "eeg", f"{sub}_task-P3_eeg.set")
    if not os.path.exists(set_file):
        print("Missing:", set_file)
        return None
    
    # Load EEG
    raw = mne.io.read_raw_eeglab(set_file, preload=True)
    
    # Drop irrelevant channels (frontal/EOG noise)
    to_drop = [ch for ch in DROP_CHANNELS if ch in raw.ch_names]
    if to_drop:
        raw.drop_channels(to_drop)
        print("Dropped channels:", to_drop)
    
    # Filter
    raw.filter(0.1, 30.0, fir_design="firwin")
    
    # Average reference
    raw.set_eeg_reference("average")
    
    # Load events from TSV file (BIDS format)
    events_tsv = os.path.join(RAW_DIR, sub, "eeg", f"{sub}_task-P3_events.tsv")
    events_df = pd.read_csv(events_tsv, sep='\t')
    
    # Filter for stimulus events only (not responses)
    stim_events = events_df[events_df['trial_type'] == 'stimulus'].copy()
    
    # ERP CORE P3b: value 13 = target, values 11,12,14,15 = standard
    stim_events['event_id'] = stim_events['value'].apply(
        lambda x: 1 if x == 13 else 0  # 1=target, 0=standard
    )
    
    # Convert to MNE events format: [sample, 0, event_id]
    events = np.column_stack([
        stim_events['sample'].values.astype(int),
        np.zeros(len(stim_events), dtype=int),
        stim_events['event_id'].values.astype(int)
    ])
    
    # Create event mapping for MNE
    event_map = {"standard": 0, "target": 1}
    
    print("Events loaded:", len(events))
    print("Target events:", np.sum(events[:, 2] == 1))
    print("Standard events:", np.sum(events[:, 2] == 0))
    
    # Epoching — crop 0 to 600ms (P3b window)
    epochs = mne.Epochs(
        raw, events, event_map,
        tmin=0.0, tmax=0.6,  # Focus on P3b window
        baseline=(0.0, 0.1),  # baseline = first 100ms
        reject=dict(eeg=250e-6),  # 250 µV threshold (more lenient)
        preload=True
    )
    
    if len(epochs) == 0:
        print("All epochs rejected.")
        return None
    
    print("Remaining epochs:", len(epochs))
    
    X = epochs.get_data()  # shape: (n_epochs, channels, time)
    y = epochs.events[:, 2]  # event codes (13 or 23)
    
    # Labels are already mapped by MNE (0=standard, 1=target)
    y_mapped = y
    
    print(f"{sub}: {X.shape[0]} epochs saved.")
    
    np.save(os.path.join(OUT_DIR, f"{sub}_X.npy"), X)
    np.save(os.path.join(OUT_DIR, f"{sub}_y.npy"), y_mapped)
    
    return X, y_mapped

# Process all subjects
all_X, all_y = [], []
for sub in subjects:
    out = preprocess_subject(sub)
    if out:
        X, y = out
        all_X.append(X)
        all_y.append(y)

# Combine all subjects
if len(all_X) > 0:
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    
    np.save(os.path.join(OUT_DIR, "all_X.npy"), X_all)
    np.save(os.path.join(OUT_DIR, "all_y.npy"), y_all)
    
    print("\n=== FINAL DATASET ===")
    print("X:", X_all.shape)
    print("y:", y_all.shape)
    print("Class distribution:", np.bincount(y_all))
    print("Saved to", OUT_DIR)
else:
    print("No subjects processed.")