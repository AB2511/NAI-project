import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data/erp_core_p3b"
SAVE_DIR = "figures/erp_core"
os.makedirs(SAVE_DIR, exist_ok=True)

SUBJECTS = [s for s in os.listdir(DATA_DIR) if s.startswith("sub-")]
SUBJECTS = sorted(SUBJECTS)

# ERP parameters - Remapped event codes
EVENT_ID = {"target": 1, "standard": 2}
TMIN, TMAX = -0.2, 0.8  # -200ms to 800ms
BASELINE = (None, 0)     # baseline from -200ms to 0ms

CHANNELS_OF_INTEREST = ["Pz", "CPz", "Cz"]  # classical P3b channels

def load_subject(sub):
    """ Load subject EEG data + events. """
    path = f"{DATA_DIR}/{sub}/eeg/{sub}_task-P3_eeg.set"
    raw = mne.io.read_raw_eeglab(path, preload=True)
    
    # Event extraction - filter only stimulus events
    events_tsv = f"{DATA_DIR}/{sub}/eeg/{sub}_task-P3_events.tsv"
    ev = pd.read_csv(events_tsv, sep="\t")
    
    # Filter only stimulus events (not responses)
    stim_events = ev[ev["trial_type"] == "stimulus"].copy()
    
    # Create events array with proper format
    events = np.zeros((len(stim_events), 3), dtype=int)
    events[:, 0] = (stim_events["onset"].values * raw.info["sfreq"]).astype(int)  # onset in samples
    
    # Remap event codes: 12 = target (1), others = standard (2)
    event_codes = stim_events["value"].values.astype(int)
    events[:, 2] = np.where(event_codes == 12, 1, 2)  # 1=target, 2=standard
    
    return raw, events

def compute_erp(raw, events):
    """ Epoch and compute ERPs for target + standard. """
    epochs = mne.Epochs(
        raw,
        events,
        event_id=EVENT_ID,
        tmin=TMIN,
        tmax=TMAX,
        baseline=BASELINE,
        preload=True
    )
    
    evoked_standard = epochs["standard"].average()
    evoked_target = epochs["target"].average()
    evoked_diff = mne.combine_evoked([evoked_target, evoked_standard], weights=[1, -1])
    
    return evoked_standard, evoked_target, evoked_diff

def plot_single_subject(sub):
    """ Plot ERP for Pz/CPz/Cz for a single subject. """
    print(f"Processing {sub}...")
    raw, events = load_subject(sub)
    ev_std, ev_tgt, ev_diff = compute_erp(raw, events)
    
    fig, ax = plt.subplots(len(CHANNELS_OF_INTEREST), 1, figsize=(8, 10))
    
    for i, ch in enumerate(CHANNELS_OF_INTEREST):
        ax[i].plot(ev_std.times * 1000, ev_std.data[ev_std.ch_names.index(ch)], label="Standard", color="blue")
        ax[i].plot(ev_tgt.times * 1000, ev_tgt.data[ev_tgt.ch_names.index(ch)], label="Target", color="red")
        ax[i].plot(ev_diff.times * 1000, ev_diff.data[ev_diff.ch_names.index(ch)], label="Difference", color="green")
        
        ax[i].axvline(0, color="black", linestyle="--")
        ax[i].axvline(300, color="purple", linestyle="--", alpha=0.5)
        ax[i].axvline(500, color="purple", linestyle="--", alpha=0.5)
        
        ax[i].set_title(f"{sub} – {ch}")
        ax[i].set_xlabel("Time (ms)")
        ax[i].set_ylabel("µV")
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{sub}_erp.png", dpi=300)
    plt.close()

def plot_group_average():
    """ Plot group-level ERP (averaged across subjects). """
    print("Computing group average ERPs...")
    
    ev_std_list = []
    ev_tgt_list = []
    
    for sub in SUBJECTS:
        raw, events = load_subject(sub)
        ev_std, ev_tgt, _ = compute_erp(raw, events)
        ev_std_list.append(ev_std)
        ev_tgt_list.append(ev_tgt)
    
    group_std = mne.combine_evoked(ev_std_list, weights="equal")
    group_tgt = mne.combine_evoked(ev_tgt_list, weights="equal")
    group_diff = mne.combine_evoked([group_tgt, group_std], weights=[1, -1])
    
    # Save plots
    for ch in CHANNELS_OF_INTEREST:
        plt.figure(figsize=(8, 5))
        plt.plot(group_std.times * 1000, group_std.data[group_std.ch_names.index(ch)], label="Standard", color="blue")
        plt.plot(group_tgt.times * 1000, group_tgt.data[group_tgt.ch_names.index(ch)], label="Target", color="red")
        plt.plot(group_diff.times * 1000, group_diff.data[group_diff.ch_names.index(ch)], label="Difference", color="green")
        
        plt.axvline(0, color="black", linestyle="--")
        plt.axvline(300, color="purple", linestyle="--", alpha=0.5)
        plt.axvline(500, color="purple", linestyle="--", alpha=0.5)
        
        plt.title(f"Group Average ERP – {ch}")
        plt.xlabel("Time (ms)")
        plt.ylabel("µV")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/group_erp_{ch}.png", dpi=300)
        plt.close()
    
    print("Group ERP plots saved!")

if __name__ == "__main__":
    print(f"Found {len(SUBJECTS)} subjects.")
    
    for sub in SUBJECTS:
        plot_single_subject(sub)
    
    plot_group_average()
    
    print("DONE – All ERP plots generated!")