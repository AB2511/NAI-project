"""
Publication-grade topomap pipeline for ERP CORE P3b

Saves subject-level and group-level topographic maps for:
- Target average
- Standard average  
- Difference (Target - Standard)

Features:
- Uses BioSemi64 montage (as in ERP CORE)
- Finds P3b peak in a user-specified window (default 300-500 ms)
- Saves high-resolution PNGs (300 DPI)
- Graceful fallbacks if Pz is missing (averages parietal channels)
- Group-level grand average maps and subject-level maps

Usage:
python topomap_p3b.py --data-root data/erp_core_p3b --out-dir results/topomaps

Requirements:
- mne, numpy, matplotlib
- dataset arranged as: data/erp_core_p3b/sub-XXX/eeg/sub-XXX_task-P3_eeg.set
"""

import argparse
from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def find_peak_time(evoked, picks, tmin=0.3, tmax=0.5):
    """Return latency (s) of maximal absolute mean over picks in given window."""
    times = evoked.times
    start, stop = times.searchsorted([tmin, tmax])
    if stop <= start:
        # fallback - use closest indices
        start = max(0, start - 1)
        stop = min(len(times) - 1, stop + 1)
    
    data = evoked.data[picks, start:stop]
    # mean across chosen channels then find time index
    mean_ts = np.abs(data).mean(axis=0)
    peak_idx = mean_ts.argmax()
    return times[start + peak_idx]

def pick_parietal_channels(info):
    """Preferred single channel Pz else CPz/Cz, otherwise average of common parietal set"""
    # Return list of channel names that exist in info
    prefer = ["Pz", "CPz", "Cz"]
    for ch in prefer:
        if ch in info['ch_names']:
            return [ch]
    
    # fallback average of parietal cluster
    cluster = ["P3", "P4", "Pz", "CPz", "POz", "P7", "P8"]
    present = [c for c in cluster if c in info['ch_names']]
    if len(present) == 0:
        # final fallback: use all EEG channels
        picks = mne.pick_types(info, eeg=True)
        return [info['ch_names'][i] for i in picks]
    return present

def plot_and_save_topomap(evoked, t, out_path, title, vmin=None, vmax=None, cmap=None):
    """Plot and save a single topomap at time t"""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=300)
    
    # Get data at time point
    time_idx = evoked.time_as_index(t)
    data = evoked.data[:, time_idx]
    
    # Plot topomap with correct parameters
    im, _ = mne.viz.plot_topomap(
        data, 
        evoked.info, 
        axes=ax,
        show=False
    )
    
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def process_subject(sub_dir, args):
    """Process a single subject and return evoked responses"""
    eeg_dir = sub_dir / 'eeg'
    set_files = list(eeg_dir.glob('*.set'))
    if len(set_files) == 0:
        raise FileNotFoundError(f'No .set file found in {eeg_dir}')
    
    set_file = set_files[0]
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    
    # set BioSemi64 montage - be more careful about channel matching
    try:
        montage = mne.channels.make_standard_montage('biosemi64')
        # Only set montage for channels that exist in both
        montage_ch_names = set(montage.ch_names)
        raw_ch_names = set(raw.ch_names)
        common_channels = montage_ch_names.intersection(raw_ch_names)
        
        if len(common_channels) > 10:  # Only if we have enough channels
            raw.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"Warning: Could not set montage: {e}")
        pass
    
    # Load events from TSV file (BIDS format)
    events_tsv = list(eeg_dir.glob('*events.tsv'))
    if not events_tsv:
        raise FileNotFoundError(f'No events.tsv file found in {eeg_dir}')
    
    tsv_file = events_tsv[0]
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Filter only stimulus events (not responses)
    stim_events = df[df["trial_type"] == "stimulus"].copy()
    
    # Create events array with proper format
    events = np.zeros((len(stim_events), 3), dtype=int)
    events[:, 0] = (stim_events["onset"].values * raw.info["sfreq"]).astype(int)  # onset in samples
    
    # Remap event codes: 12 = target (1), others = standard (2)
    event_codes = stim_events["value"].values.astype(int)
    events[:, 2] = np.where(event_codes == 12, 1, 2)  # 1=target, 2=standard
    
    # Drop problematic channels (EOG, etc.)
    bad_channels = ['HEOG_left', 'HEOG_right', 'VEOG_lower', 'VEOG_upper']
    raw.info['bads'].extend([ch for ch in bad_channels if ch in raw.ch_names])
    
    # Create epochs
    tmin, tmax = -0.2, 0.8
    picks = mne.pick_types(raw.info, eeg=True, eog=False, exclude='bads')
    epochs = mne.Epochs(
        raw, events, 
        event_id={"target": 1, "standard": 2}, 
        tmin=tmin, tmax=tmax,
        baseline=(-0.2, 0.0), 
        preload=True, 
        picks=picks, 
        verbose=False
    )
    
    # Re-reference to average
    epochs.set_eeg_reference('average', verbose=False)
    
    # Compute evokeds
    ev_target = epochs["target"].average()
    ev_standard = epochs["standard"].average()
    ev_diff = mne.combine_evoked([ev_target, ev_standard], weights=[1, -1])
    
    return ev_target, ev_standard, ev_diff

def main():
    parser = argparse.ArgumentParser(description='Topomap pipeline for ERP CORE P3b')
    parser.add_argument('--data-root', type=str, default='data/erp_core_p3b', 
                       help='Root folder with sub-XXX folders')
    parser.add_argument('--out-dir', type=str, default='results/topomaps', 
                       help='Output folder for figures')
    parser.add_argument('--tmin', type=float, default=0.3, 
                       help='Window start (s) to search P3b peak')
    parser.add_argument('--tmax', type=float, default=0.5, 
                       help='Window end (s) to search P3b peak')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    subs = sorted([d for d in data_root.iterdir() if d.name.startswith('sub-')])
    subject_evokeds = []
    
    print(f'Found {len(subs)} subjects. Processing...')
    
    for sub in subs:
        print('Processing', sub.name)
        try:
            ev_t, ev_s, ev_d = process_subject(sub, args)
            
            subject_evokeds.append({
                'sub': sub.name, 
                'target': ev_t, 
                'standard': ev_s, 
                'diff': ev_d
            })
            
            # determine peak time using parietal picks
            picks_names = pick_parietal_channels(ev_t.info)
            picks = [ev_t.ch_names.index(ch) for ch in picks_names]
            peak_t = find_peak_time(ev_d, picks, tmin=args.tmin, tmax=args.tmax)
            
            # Save subject-level figures
            subj_out = out_dir / sub.name
            subj_out.mkdir(parents=True, exist_ok=True)
            
            # three maps: at detected peak, and at fixed 0.3 and 0.4s
            times_to_plot = [peak_t, 0.3, 0.4]
            names = ['diff_peak', 'diff_300ms', 'diff_400ms']
            
            for t, nm in zip(times_to_plot, names):
                fpath = subj_out / f'{sub.name}_{nm}.png'
                title = f'{sub.name} Diff @ {int(t*1000)} ms (Target-Standard)'
                plot_and_save_topomap(ev_d, t, fpath, title)
            
            # Also save target and standard maps at peak
            plot_and_save_topomap(
                ev_t, peak_t, 
                subj_out / f'{sub.name}_target_peak.png', 
                f'{sub.name} Target @ {int(peak_t*1000)} ms'
            )
            plot_and_save_topomap(
                ev_s, peak_t, 
                subj_out / f'{sub.name}_standard_peak.png', 
                f'{sub.name} Standard @ {int(peak_t*1000)} ms'
            )
            
        except Exception as e:
            print(f'ERROR processing {sub.name}:', e)
    
    # Group-level maps
    if len(subject_evokeds) == 0:
        print('No subject evokeds available - nothing to group average')
        return
    
    # create lists of evokeds
    target_list = [d['target'] for d in subject_evokeds]
    standard_list = [d['standard'] for d in subject_evokeds]
    diff_list = [d['diff'] for d in subject_evokeds]
    
    ga_target = mne.grand_average(target_list)
    ga_standard = mne.grand_average(standard_list)
    ga_diff = mne.grand_average(diff_list)
    
    # find group peak time
    picks_names = pick_parietal_channels(ga_diff.info)
    picks = [ga_diff.ch_names.index(ch) for ch in picks_names]
    group_peak = find_peak_time(ga_diff, picks, tmin=args.tmin, tmax=args.tmax)
    
    print(f'Group peak at: {group_peak:.3f}s ({int(group_peak*1000)}ms)')
    
    grp_out = out_dir / 'group'
    grp_out.mkdir(parents=True, exist_ok=True)
    
    # Group topomaps at peak
    plot_and_save_topomap(
        ga_diff, group_peak, 
        grp_out / f'group_diff_peak_{int(group_peak*1000)}ms.png', 
        f'Group Diff @ {int(group_peak*1000)} ms'
    )
    plot_and_save_topomap(
        ga_target, group_peak, 
        grp_out / f'group_target_peak_{int(group_peak*1000)}ms.png', 
        f'Group Target @ {int(group_peak*1000)} ms'
    )
    plot_and_save_topomap(
        ga_standard, group_peak, 
        grp_out / f'group_standard_peak_{int(group_peak*1000)}ms.png', 
        f'Group Standard @ {int(group_peak*1000)} ms'
    )
    
    # also fixed windows 300 and 400 ms
    for t in [0.3, 0.4]:
        plot_and_save_topomap(
            ga_diff, t, 
            grp_out / f'group_diff_{int(t*1000)}ms.png', 
            f'Group Diff @ {int(t*1000)} ms'
        )
    
    print('Done. Topomaps saved to', out_dir)

if __name__ == '__main__':
    main()