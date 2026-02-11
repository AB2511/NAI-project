#!/usr/bin/env python3
"""
Preprocess ERP CORE P3B (EEGLAB .set/.fdt BIDS layout).

Usage:
    python src/preprocess_fix_p3b.py \
        --input data/raw_p3b \
        --output data/processed_p3b \
        --resample 256 \
        --lowpass 30 \
        --highpass 0.1 \
        --epoch_tmin -0.2 \
        --epoch_tmax 0.8 \
        --reject_uv 150e-6 \
        --balance undersample

Outputs (per subject):
    data/processed_p3b/sub-XXX_epochs.npy    -> shape (n_epochs, n_channels, n_times)
    data/processed_p3b/sub-XXX_labels.npy    -> 0/1 labels (0=non-target, 1=target)
    data/processed_p3b/metadata.json         -> summary metadata
"""

import os
import argparse
import json
import glob
from pathlib import Path

import numpy as np
import mne
import pandas as pd
from tqdm import tqdm


def find_events_tsv(subject_dir):
    # return path to events.tsv (BIDS naming: sub-XXX_task-YYY_events.tsv)
    candidates = list(Path(subject_dir).rglob('*events.tsv'))
    return candidates[0] if candidates else None


def read_event_labels(events_tsv):
    """
    Return events array compatible with mne.make_fixed_length_events? No.
    We'll read events.tsv (BIDS) and return (sample, 0, code) for mne.events
    But easier: use mne.find_events on raw (EEGLAB .set already has events).
    Here, return a mapping for label extraction: use pandas to read trial_type or value.
    """
    try:
        df = pd.read_csv(events_tsv, sep='\t')
        return df
    except Exception:
        return None


def infer_target_codes(events_df):
    """
    Given events dataframe, try to infer which event value corresponds to 'target' and which to 'standard'.
    For P300 oddball paradigm, targets are typically the minority stimulus class.
    Returns: (colname, target_values_set)
    """
    if events_df is None:
        return None, None

    # Filter to only stimulus events (not responses)
    if 'trial_type' in events_df.columns:
        stim_df = events_df[events_df['trial_type'] == 'stimulus'].copy()
    else:
        stim_df = events_df.copy()
    
    if len(stim_df) == 0:
        return None, None

    # Look for value column with stimulus codes
    if 'value' in stim_df.columns:
        vals = stim_df['value'].astype(str)
        counts = vals.value_counts()
        
        # In P300 oddball, targets are typically values ending in 5 (15, 25, 35, 45, 55)
        # or the minority class
        target_candidates = [v for v in counts.index if str(v).endswith('5')]
        if target_candidates:
            return 'value', set(target_candidates)
        
        # Fallback: minority class is target
        if len(counts) >= 2:
            minority = counts.index[-1]  # least frequent
            return 'value', set([str(minority)])
        else:
            return 'value', set([counts.index[0]])
    
    return None, None


def main(args):
    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    subj_dirs = sorted([p for p in inp.iterdir() if p.is_dir() and p.name.startswith('sub-')])
    metadata = {
        'n_subjects': len(subj_dirs),
        'subjects': {},
        'params': {
            'resample': args.resample,
            'lowpass': args.lowpass,
            'highpass': args.highpass,
            'epoch_tmin': args.epoch_tmin,
            'epoch_tmax': args.epoch_tmax,
            'reject_uv': args.reject_uv
        }
    }

    for sdir in tqdm(subj_dirs, desc='Subjects'):
        subj = sdir.name
        eeg_glob = list((sdir).rglob('*.set'))
        if not eeg_glob:
            print(f"[WARN] No .set found for {subj}, skipping.")
            continue
        set_path = eeg_glob[0]
        print(f"\nProcessing {subj}: {set_path}")

        # read raw (EEGLAB)
        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose='ERROR')
        except Exception as e:
            print(f"[ERROR] cannot read {set_path}: {e}")
            continue

        # drop non-eeg channels if present
        picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False, meg=False)
        raw.pick(picks_eeg)

        # resample
        if args.resample is not None and raw.info['sfreq'] != args.resample:
            raw.resample(args.resample, npad='auto')

        # filter
        raw.load_data()
        raw.filter(l_freq=args.highpass, h_freq=args.lowpass)

        # set average reference
        raw.set_eeg_reference('average', projection=False)

        # -----------------------------------------------------
        # LOAD EVENTS FROM events.tsv INSTEAD OF EEGLAB ANNOTS
        # -----------------------------------------------------
        events_tsv = find_events_tsv(sdir)
        if not events_tsv:
            print(f"[ERROR] No events.tsv found for {subj}. Skipping.")
            continue
            
        # Read events.tsv
        events_df = pd.read_csv(events_tsv, sep='\t')
        
        # Filter to stimulus events only
        stim_df = events_df[events_df['trial_type'] == 'stimulus'].copy()
        if len(stim_df) == 0:
            print(f"[ERROR] No stimulus events found for {subj}. Skipping.")
            continue
        
        # Convert onsets to samples
        onsets = (stim_df['onset'].values * raw.info['sfreq']).astype(int)
        codes = stim_df['value'].astype(int).values
        
        # Build events array (N, 3) for MNE
        events = np.column_stack([onsets, np.zeros_like(codes), codes])
        
        # Identify targets and standards
        # Rule: values ending in "5" (15, 25, 35, 45, 55) are targets
        unique_codes = np.unique(codes)
        targets = [v for v in unique_codes if str(v).endswith('5')]
        standards = [v for v in unique_codes if not str(v).endswith('5')]
        
        print(f"[INFO] Loaded events from TSV: {events_tsv}")
        print(f"[INFO] Target codes: {targets}")
        print(f"[INFO] Standard codes: {standards}")
        print(f"[INFO] Total events: {len(events)}")
        
        if len(targets) == 0:
            print(f"[WARN] No target codes found for {subj}.")
            target_vals = set()
        else:
            target_vals = set([str(t) for t in targets])
        
        # Create code to value mapping for later use
        code_to_val = {code: str(code) for code in unique_codes}

        # Create epochs using the events from TSV
        try:
            epochs = mne.Epochs(raw, events=events,
                                event_id=None,
                                tmin=args.epoch_tmin,
                                tmax=args.epoch_tmax,
                                baseline=(args.epoch_tmin, 0.0),
                                preload=True,
                                reject=None,
                                verbose='ERROR')
        except Exception as e:
            print(f"[ERROR] Could not create epochs for {subj}: {e}")
            continue

        # Convert event codes to labels and determine target/non-target
        labels = []
        for ev in epochs.events:
            code = ev[2]
            val = code_to_val.get(code, str(code))
            labels.append(str(val))

        labels = np.array(labels)

        # Determine numeric labels 0/1: 1 = target, 0 = non-target
        if target_vals:
            mask_target = np.array([str(l) in target_vals for l in labels])
        else:
            # fallback: minority class as target
            vals, counts = np.unique(labels, return_counts=True)
            if len(vals) > 1:
                minority = vals[np.argmin(counts)]
                mask_target = (labels == minority)
            else:
                mask_target = np.zeros_like(labels, dtype=bool)
        numeric_labels = mask_target.astype(int)

        # basic artifact rejection: peak-to-peak threshold (uV)
        reject_uv = args.reject_uv
        picks = mne.pick_types(epochs.info, eeg=True)
        # apply drop_bad based on peak-to-peak
        try:
            epochs.drop_bad(reject=dict(eeg=reject_uv))
        except Exception:
            # fallback: manual rejection by amplitude
            data = epochs.get_data()
            peak2peak = data.ptp(axis=2)
            good = (peak2peak < (reject_uv * 1e6)) if False else (peak2peak < (reject_uv))
            # above fallback probably won't be used; keep simple
            pass

        # collect final data and labels after drop
        final_data = epochs.get_data()  # n_epochs x n_channels x n_times
        final_labels = numeric_labels[epochs.selection]  # keep only non-dropped epochs

        # optional balancing
        if args.balance == 'undersample':
            # undersample majority class
            from sklearn.utils import resample
            idx_pos = np.where(final_labels == 1)[0]
            idx_neg = np.where(final_labels == 0)[0]
            if len(idx_pos) == 0:
                print(f"[WARN] No target epochs for {subj}.")
                chosen_idx = np.arange(final_labels.shape[0])
            else:
                n_pos = len(idx_pos)
                idx_neg_down = resample(idx_neg, replace=False, n_samples=n_pos, random_state=42)
                chosen_idx = np.concatenate([idx_pos, idx_neg_down])
        else:
            chosen_idx = np.arange(final_labels.shape[0])

        final_data = final_data[chosen_idx]
        final_labels = final_labels[chosen_idx]

        # Save
        subj_out_epochs = out / f"{subj}_epochs.npy"
        subj_out_labels = out / f"{subj}_labels.npy"
        np.save(subj_out_epochs, final_data)
        np.save(subj_out_labels, final_labels)

        metadata['subjects'][subj] = {
            'n_epochs': int(final_data.shape[0]),
            'n_channels': int(final_data.shape[1]),
            'n_times': int(final_data.shape[2]),
            'sfreq': float(epochs.info['sfreq'])
        }

    # create time vector
    # get one epoch file to compute times
    example = next(out.glob('*_epochs.npy'), None)
    if example:
        ex = np.load(example, mmap_mode='r')
        n_times = ex.shape[2]
        sf = metadata['params']['resample'] or None
        if sf is None:
            # try to read sf from metadata first subject
            if len(metadata['subjects']) > 0:
                sf = list(metadata['subjects'].values())[0]['sfreq']
            else:
                sf = 256
        tmin = metadata['params']['epoch_tmin'] if 'epoch_tmin' in metadata['params'] else args.epoch_tmin
        times = np.linspace(tmin, tmin + (n_times-1)/sf, n_times).tolist()
        metadata['time_vector'] = times

    # write metadata file
    with open(out / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nPreprocessing complete. Saved results to:", out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to raw BIDS-style folder (data/raw_p3b)')
    parser.add_argument('--output', required=True, help='Path to output processed folder')
    parser.add_argument('--resample', type=int, default=256)
    parser.add_argument('--lowpass', type=float, default=30.0)
    parser.add_argument('--highpass', type=float, default=0.1)
    parser.add_argument('--epoch_tmin', type=float, default=-0.2)
    parser.add_argument('--epoch_tmax', type=float, default=0.8)
    parser.add_argument('--reject_uv', type=float, default=150e-6, help='peak-to-peak rejection threshold in Volts (default 150 uV)')
    parser.add_argument('--balance', choices=['none', 'undersample'], default='none')
    args = parser.parse_args()
    main(args)