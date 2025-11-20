#!/usr/bin/env python3
"""
paper_analysis_p3b.py

Combined publication-grade analysis for ERP P3b project.

Provides subcommands:
- erp_plot: subject + group ERP plots and difference waves
- topomap: subject + group topographic maps (fixed times and peak)
- cluster: cluster-based permutation tests on difference waves

Usage examples:
python src/paper_analysis_p3b.py erp_plot --processed data/processed_p3b --out figures/erp --chan Pz --baseline -0.1 0 --tmin -0.1 --tmax 0.6
python src/paper_analysis_p3b.py topomap --processed data/processed_p3b --raw-root data/erp_core_p3b --out figures/topomaps --times 0.30 0.35 0.40 --peak-window 0.30 0.50
python src/paper_analysis_p3b.py cluster --processed data/processed_p3b --out results/stats --n-perm 1000 --tmin 0.1 --tmax 0.6

Author: Generated for Anjali (research pipeline)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.stats import permutation_cluster_test
from scipy import stats

# Reproducibility
RNG_SEED = 42
np.random.seed(RNG_SEED)

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_processed_subject(processed_dir, subject):
    """Load a subject's processed arrays saved as sub-XXX_X.npy and sub-XXX_y.npy
    Returns X (n_epochs, n_chan, n_times), y (n_epochs,)
    """
    x_path = Path(processed_dir) / f"{subject}_X.npy"
    y_path = Path(processed_dir) / f"{subject}_y.npy"
    
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing processed files for {subject}: {x_path} or {y_path}")
    
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y

def list_subjects(processed_dir):
    # find files like sub-001_X.npy
    files = sorted(Path(processed_dir).glob('sub-*_X.npy'))
    subjects = [f.stem.split('_')[0] for f in files]
    return subjects

def make_times(n_times, sfreq=1024, tmin=-0.2):
    # default compatible with your earlier pipeline; attempts to infer dt if needed
    # If n_times known and sfreq known, generate times array
    dt = 1.0 / sfreq
    times = np.arange(0, n_times) * dt + tmin
    return times

def erp_plot(args):
    processed = Path(args.processed)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    
    subjects = list_subjects(processed)
    print(f'Found {len(subjects)} subjects: {subjects[0]} ... {subjects[-1]}')
    
    per_subject = {}
    
    for sub in subjects:
        print(f'Processing {sub}...')
        X, y = load_processed_subject(processed, sub)
        n_epochs, n_chan, n_times = X.shape
        
        # infer sfreq from stored metadata? assume 1024 if not provided
        sfreq = getattr(args, 'sfreq', 1024)
        times = make_times(n_times, sfreq=sfreq, tmin=args.tmin)
        
        # pick channel index for args.chan
        # load a canonical montage mapping if available via mne
        # we assume channel order is consistent across subjects and matches your saved montage
        ch_names = None
        # try to guess channel names from a saved file if provided
        # otherwise require user to supply index via --chan-index
        if args.chan_index is not None:
            ch_idx = int(args.chan_index)
        else:
            # try map common name to index by searching a saved channels file
            # fallback to center channel index
            ch_idx = n_chan // 2
            print(f'Warning: channel index not specified; defaulting to {ch_idx}')
        
        # compute ERPs
        target_idx = y == 1
        standard_idx = y == 0
        
        if target_idx.sum() == 0:
            raise RuntimeError(f'No targets found for {sub}')
        
        erp_target = X[target_idx].mean(axis=0)  # shape (n_chan, n_times)
        erp_standard = X[standard_idx].mean(axis=0)
        diff = erp_target - erp_standard
        
        # compute CI across trials (bootstrap percentiles)
        def bootstrap_ci(data, axis=0, n_boot=1000, ci=95):
            # returns mean, lower, upper
            mean = data.mean(axis=axis)
            boot_means = []
            n = data.shape[0]
            for _ in range(n_boot):
                idx = np.random.randint(0, n, n)
                boot_means.append(data[idx].mean(axis=axis))
            boot = np.stack(boot_means, axis=0)
            lo = np.percentile(boot, (100 - ci) / 2.0, axis=0)
            hi = np.percentile(boot, 100 - (100 - ci) / 2.0, axis=0)
            return mean, lo, hi
        
        # extract channel waveform
        ch_wave_t, lo_t, hi_t = bootstrap_ci(X[target_idx][:, ch_idx, :], axis=0, n_boot=300)
        ch_wave_s, lo_s, hi_s = bootstrap_ci(X[standard_idx][:, ch_idx, :], axis=0, n_boot=300)
        diff_mean = ch_wave_t - ch_wave_s
        
        # smoothing optional
        if args.smooth_ms and args.smooth_ms > 0:
            from scipy.ndimage import gaussian_filter1d
            sigma_samples = (args.smooth_ms / 1000.0) * sfreq
            ch_wave_t = gaussian_filter1d(ch_wave_t, sigma=sigma_samples)
            ch_wave_s = gaussian_filter1d(ch_wave_s, sigma=sigma_samples)
            diff_mean = gaussian_filter1d(diff_mean, sigma=sigma_samples)
            lo_t = gaussian_filter1d(lo_t, sigma=sigma_samples)
            hi_t = gaussian_filter1d(hi_t, sigma=sigma_samples)
            lo_s = gaussian_filter1d(lo_s, sigma=sigma_samples)
            hi_s = gaussian_filter1d(hi_s, sigma=sigma_samples)
        
        # find peak in window
        tmin_idx = np.searchsorted(times, args.peak_window[0])
        tmax_idx = np.searchsorted(times, args.peak_window[1])
        peak_idx = np.argmax(diff_mean[tmin_idx:tmax_idx]) + tmin_idx
        peak_time = times[peak_idx]
        peak_amp = diff_mean[peak_idx]
        
        per_subject[sub] = dict(
            peak_time=float(peak_time), 
            peak_amp=float(peak_amp), 
            n_target=int(target_idx.sum())
        )
        
        # plot subject ERP
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
        ax.fill_between(times, lo_t, hi_t, alpha=0.2, label='Target 95% CI', color='C0')
        ax.fill_between(times, lo_s, hi_s, alpha=0.2, label='Standard 95% CI', color='C1')
        ax.plot(times, ch_wave_t, label='Target', color='C0')
        ax.plot(times, ch_wave_s, label='Standard', color='C1')
        ax.plot(times, diff_mean, label='Difference (T-S)', color='C2')
        ax.axvline(0, color='k', linestyle='--')
        ax.axvline(args.peak_window[0], color='gray', linestyle=':')
        ax.axvline(args.peak_window[1], color='gray', linestyle=':')
        ax.axvline(peak_time, color='C2', linestyle='--', label=f'Peak {int(peak_time*1000)} ms')
        ax.set_xlim(args.tmin, args.tmax)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (uV)')
        ax.legend(loc='upper right')
        fig.savefig(out_dir / f'{sub}_erp_chan{ch_idx}.png', bbox_inches='tight')
        plt.close(fig)
    
    # group average
    print('Computing group averages...')
    
    # stack subject-level difference waves at chosen channel
    all_diff = []
    all_target = []
    all_standard = []
    times = None
    
    for sub in subjects:
        X, y = load_processed_subject(processed, sub)
        n_epochs, n_chan, n_times = X.shape
        sfreq = getattr(args, 'sfreq', 1024)
        times = make_times(n_times, sfreq=sfreq, tmin=args.tmin)
        ch_idx = args.chan_index or (n_chan // 2)
        
        targ = X[y == 1][:, ch_idx, :]
        std = X[y == 0][:, ch_idx, :]
        
        all_target.append(targ.mean(axis=0))
        all_standard.append(std.mean(axis=0))
        all_diff.append(targ.mean(axis=0) - std.mean(axis=0))
    
    all_target = np.stack(all_target, axis=0)
    all_standard = np.stack(all_standard, axis=0)
    all_diff = np.stack(all_diff, axis=0)
    
    ga_target = all_target.mean(axis=0)
    ga_standard = all_standard.mean(axis=0)
    ga_diff = all_diff.mean(axis=0)
    
    # group CI across subjects
    ga_t_lo = np.percentile(all_target, 2.5, axis=0)
    ga_t_hi = np.percentile(all_target, 97.5, axis=0)
    ga_s_lo = np.percentile(all_standard, 2.5, axis=0)
    ga_s_hi = np.percentile(all_standard, 97.5, axis=0)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    ax.fill_between(times, ga_t_lo, ga_t_hi, alpha=0.2, color='C0')
    ax.fill_between(times, ga_s_lo, ga_s_hi, alpha=0.2, color='C1')
    ax.plot(times, ga_target, color='C0', label='Group Target')
    ax.plot(times, ga_standard, color='C1', label='Group Standard')
    ax.plot(times, ga_diff, color='C2', label='Group Difference')
    
    # peak
    tmin_idx = np.searchsorted(times, args.peak_window[0])
    tmax_idx = np.searchsorted(times, args.peak_window[1])
    peak_idx = np.argmax(ga_diff[tmin_idx:tmax_idx]) + tmin_idx
    peak_time = times[peak_idx]
    ax.axvline(peak_time, color='C2', linestyle='--', label=f'Group peak {int(peak_time*1000)} ms')
    
    ax.set_xlim(args.tmin, args.tmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (uV)')
    ax.legend()
    fig.savefig(out_dir / 'group_erp.png', bbox_inches='tight')
    plt.close(fig)
    
    # save per_subject summary
    summary_path = Path(args.out).parent / 'results' / 'erp_peak_summary.json'
    ensure_dir(summary_path.parent)
    with open(summary_path, 'w') as fh:
        json.dump(per_subject, fh, indent=2)
    
    print('ERP plotting done. Group + subject figures saved.')

def load_channel_info(raw_root, subject):
    """Load channel names and positions from BIDS data"""
    if raw_root is None:
        return None, None
    
    channels_file = Path(raw_root) / subject / 'eeg' / f'{subject}_task-P3_channels.tsv'
    electrodes_file = Path(raw_root) / subject / 'eeg' / f'{subject}_task-P3_electrodes.tsv'
    
    if not channels_file.exists() or not electrodes_file.exists():
        return None, None
    
    # Read channel names and types
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for channel info loading")
        return None, None
    
    channels_df = pd.read_csv(channels_file, sep='\t')
    electrodes_df = pd.read_csv(electrodes_file, sep='\t')
    
    # Filter to EEG channels only (exclude EOG)
    eeg_channels = channels_df[channels_df['type'] == 'EEG']['name'].tolist()
    
    # Get positions for EEG channels
    eeg_electrodes = electrodes_df[electrodes_df['name'].isin(eeg_channels)]
    
    return eeg_channels, eeg_electrodes

def topomap(args):
    processed = Path(args.processed)
    raw_root = Path(args.raw_root) if args.raw_root else None
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    
    subjects = list_subjects(processed)
    all_evokeds = []
    sfreq = getattr(args, 'sfreq', 1024)
    
    for sub in subjects:
        try:
            X, y = load_processed_subject(processed, sub)
        except FileNotFoundError:
            print(f'Skipping {sub} (missing)')
            continue
        
        n_epochs, n_chan, n_times = X.shape
        times = make_times(n_times, sfreq=sfreq, tmin=args.tmin)
        
        # Try to load original channel information
        ch_names, electrodes_df = load_channel_info(raw_root, sub)
        
        if ch_names is not None and len(ch_names) >= n_chan:
            # Use original channel names (first n_chan channels)
            ch_names = ch_names[:n_chan]
            info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        else:
            # Fallback to generic names
            ch_names = [f'EEG{i:02d}' for i in range(n_chan)]
            info = mne.create_info(ch_names, sfreq, ch_types='eeg')
        
        # average
        evoked_t = mne.EvokedArray(X[y == 1].mean(axis=0), info, tmin=times[0])
        evoked_s = mne.EvokedArray(X[y == 0].mean(axis=0), info, tmin=times[0])
        evoked_diff = mne.combine_evoked([evoked_t, evoked_s], weights=[1, -1])
        
        # Set electrode positions if available
        if electrodes_df is not None and not args.skip_montage:
            try:
                import pandas as pd
                # Create custom montage from electrode positions
                pos_dict = {}
                for _, row in electrodes_df.iterrows():
                    if row['name'] in ch_names and pd.notna(row['x']):
                        pos_dict[row['name']] = [row['x'], row['y'], row['z']]
                
                if pos_dict:
                    montage = mne.channels.make_dig_montage(pos_dict, coord_frame='head')
                    evoked_diff.set_montage(montage, on_missing='ignore')
                else:
                    print(f'No valid electrode positions found for {sub}')
                    continue
            except Exception as e:
                print(f'Custom montage failed for {sub}: {e}')
                continue
        elif args.montage and not args.skip_montage:
            try:
                mont = mne.channels.make_standard_montage(args.montage)
                evoked_diff.set_montage(mont, match_case=False, on_missing='ignore')
            except Exception as e:
                print(f'Standard montage failed for {sub}: {e}')
                continue
        
        # find peak within window
        tmin_idx = np.searchsorted(times, args.peak_window[0])
        tmax_idx = np.searchsorted(times, args.peak_window[1])
        data = evoked_diff.data
        peak_sample = tmin_idx + np.argmax(np.mean(data[:, tmin_idx:tmax_idx], axis=0))
        peak_time = evoked_diff.times[peak_sample]
        
        # times to plot: provided list + peak
        times_to_plot = list(args.times) + [peak_time]
        
        subj_out = out_dir / 'subjects'
        ensure_dir(subj_out)
        
        for tt in times_to_plot:
            try:
                fig = evoked_diff.plot_topomap(tt, show=False)
                # evoked.plot_topomap returns a figure or list
                if isinstance(fig, list):
                    f = fig[0]
                else:
                    f = fig
                savepath = subj_out / f'{sub}_diff_{int(tt*1000)}ms.png'
                f.savefig(savepath, dpi=300)
                plt.close(f)
            except Exception as e:
                print(f'Warning: Could not create topomap for {sub} @ {tt}s: {e}')
        
        all_evokeds.append(evoked_diff)
    
    # group average
    if len(all_evokeds) > 0:
        grp = mne.grand_average(all_evokeds)
        grp_out = out_dir / 'group'
        ensure_dir(grp_out)
        
        times_to_plot = list(args.times)
        
        # compute group peak
        tmin_idx = np.searchsorted(grp.times, args.peak_window[0])
        tmax_idx = np.searchsorted(grp.times, args.peak_window[1])
        peak_sample = tmin_idx + np.argmax(np.mean(grp.data[:, tmin_idx:tmax_idx], axis=0))
        group_peak = grp.times[peak_sample]
        times_to_plot.append(group_peak)
        
        for tt in times_to_plot:
            try:
                fig = grp.plot_topomap(tt, show=False)
                if isinstance(fig, list):
                    f = fig[0]
                else:
                    f = fig
                f.savefig(grp_out / f'group_diff_{int(tt*1000)}ms.png', dpi=300)
                plt.close(f)
            except Exception as e:
                print('Warning (group):', e)
    
    print('Topomap generation done.')

def cluster_test(args):
    processed = Path(args.processed)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    
    subjects = list_subjects(processed)
    data_list = []  # subjects x (n_epochs, n_chan, n_times)
    sfreq = getattr(args, 'sfreq', 1024)
    
    for sub in subjects:
        X, y = load_processed_subject(processed, sub)
        # restrict to channel set (e.g., parietal channels) if provided
        # here we use all channels
        data_list.append((X, y))
    
    # build arrays: for each subject, compute difference per trial then average across trials
    # For cluster test we want subjects x channels x times array of difference waves
    subject_diffs = []
    
    for X, y in data_list:
        targ = X[y == 1]
        std = X[y == 0]
        
        # if too few trials, skip
        if targ.shape[0] < 3 or std.shape[0] < 3:
            # pad with nan
            print('Warning: subject with low trials -> skipping for cluster test')
            continue
        
        ev_t = targ.mean(axis=0)  # n_chan x n_times
        ev_s = std.mean(axis=0)
        diff = ev_t - ev_s
        subject_diffs.append(diff)
    
    subject_diffs = np.stack(subject_diffs, axis=0)  # n_subject x n_chan x n_times
    n_subjects, n_chan, n_times = subject_diffs.shape
    print('Cluster test on array:', subject_diffs.shape)
    
    # Convert to shape (n_subjects * n_chan, n_times) or use MNE cluster-based tools
    # We'll run a time-only cluster test averaged over a channel ROI (parietal) by default,
    # and a full channel-time permutation for visualization using permutation_cluster_test
    
    # Example ROI average: let user pass channels via arg. If None -> average across all channels
    roi_idx = None
    if args.roi_chan is not None:
        # try match channel names in saved info? We don't have channel names, so allow numeric list
        roi_idx = [int(x) for x in args.roi_chan.split(',')]
        roi_data = subject_diffs[:, roi_idx, :].mean(axis=1)  # n_sub x n_times
    else:
        roi_data = subject_diffs.mean(axis=1)  # average across channels -> n_sub x n_times
    
    # We want to test H0: mean across subjects is zero -> one-sample test
    X = roi_data
    
    # permutation cluster_test expects lists of arrays for two-sample; for one-sample use X - 0
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        [X], n_permutations=int(args.n_perm), tail=0, n_jobs=1, buffer_size=None
    )
    
    print('Found clusters:', len(clusters))
    
    # Save results
    outp = out_dir / 'cluster_perm_results.npz'
    np.savez(outp, T_obs=T_obs, clusters=clusters, p_values=cluster_p_values)
    
    # Visualization: plot T_obs with significant clusters masked
    times = make_times(n_times, sfreq=sfreq, tmin=args.tmin)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=300)
    
    # Handle T_obs shape - it might be 1D or 2D
    if T_obs.ndim == 1:
        t_plot = T_obs
    else:
        t_plot = T_obs.mean(axis=0) if T_obs.shape[0] > 1 else T_obs.squeeze()
    
    ax.plot(times, t_plot)
    
    sig_times = np.zeros_like(times, dtype=bool)
    for cl, p in zip(clusters, cluster_p_values):
        if p <= 0.05:
            cl_idx = cl[0]
            sig_times[cl_idx] = True
    
    ax.fill_between(times, ax.get_ylim()[0], ax.get_ylim()[1], 
                   where=sig_times, color='red', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('T')
    fig.savefig(out_dir / 'cluster_Tobs.png', bbox_inches='tight')
    plt.close(fig)
    
    print('Cluster test done. Results saved to', outp)

def main():
    parser = argparse.ArgumentParser(description='Publication-grade P3b analysis toolkit')
    sub = parser.add_subparsers(dest='command')
    
    p_erp = sub.add_parser('erp_plot')
    p_erp.add_argument('--processed', required=True)
    p_erp.add_argument('--out', required=True)
    p_erp.add_argument('--chan', default='Pz')
    p_erp.add_argument('--chan-index', type=int, default=None, 
                      help='channel index if names not available')
    p_erp.add_argument('--sfreq', type=float, default=1024)
    p_erp.add_argument('--baseline', nargs=2, type=float, default=[-0.1, 0.0])
    p_erp.add_argument('--tmin', type=float, default=-0.1)
    p_erp.add_argument('--tmax', type=float, default=0.6)
    p_erp.add_argument('--peak-window', nargs=2, type=float, default=[0.3, 0.5])
    p_erp.add_argument('--smooth-ms', type=float, default=0)
    
    p_top = sub.add_parser('topomap')
    p_top.add_argument('--processed', required=True)
    p_top.add_argument('--raw-root', required=False, default=None)
    p_top.add_argument('--out', required=True)
    p_top.add_argument('--montage', default='biosemi64')
    p_top.add_argument('--skip-montage', action='store_true', 
                      help='Skip montage setting (use for processed data without channel positions)')
    p_top.add_argument('--times', nargs='+', type=float, default=[0.3, 0.35, 0.4])
    p_top.add_argument('--peak-window', nargs=2, type=float, default=[0.3, 0.5])
    p_top.add_argument('--sfreq', type=float, default=1024)
    p_top.add_argument('--tmin', type=float, default=-0.2)
    
    p_clust = sub.add_parser('cluster')
    p_clust.add_argument('--processed', required=True)
    p_clust.add_argument('--out', required=True)
    p_clust.add_argument('--n-perm', type=int, default=500)
    p_clust.add_argument('--sfreq', type=float, default=1024)
    p_clust.add_argument('--tmin', type=float, default=-0.2)
    p_clust.add_argument('--tmax', type=float, default=0.8)
    p_clust.add_argument('--roi-chan', type=str, default=None, 
                        help='comma separated channel indices to form ROI')
    
    args = parser.parse_args()
    
    if args.command == 'erp_plot':
        erp_plot(args)
    elif args.command == 'topomap':
        topomap(args)
    elif args.command == 'cluster':
        cluster_test(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()