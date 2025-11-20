# Publication-Grade P3b Analysis Guide

This guide explains how to use the `src/paper_analysis_p3b.py` script for comprehensive ERP analysis.

## Overview

The script provides three main analysis functions:
- **erp_plot**: Individual and group ERP plots with statistics
- **topomap**: Topographic maps at specific time points
- **cluster**: Cluster-based permutation testing

## Prerequisites

- Processed data in `.npy` format (from your preprocessing pipeline)
- Original BIDS data (for electrode positions in topomaps)
- Required packages: `mne`, `matplotlib`, `numpy`, `scipy`, `pandas`

## Usage Examples

### 1. ERP Plotting

Generate publication-ready ERP plots with bootstrap confidence intervals:

```bash
python src/paper_analysis_p3b.py erp_plot \
    --processed data/processed_p3b \
    --out figures/erp \
    --chan-index 12 \
    --tmin -0.1 \
    --tmax 0.6 \
    --peak-window 0.3 0.5 \
    --smooth-ms 5
```

**Key parameters:**
- `--chan-index`: Channel index for plotting (e.g., 12 for Pz-like channel)
- `--peak-window`: Time window for peak detection (in seconds)
- `--smooth-ms`: Optional smoothing (milliseconds)

**Outputs:**
- Individual subject ERP plots: `figures/erp/sub-XXX_erp_chanYY.png`
- Group average plot: `figures/erp/group_erp.png`
- Peak summary: `figures/results/erp_peak_summary.json`

### 2. Topographic Maps

Create topomaps with proper electrode positions:

```bash
python src/paper_analysis_p3b.py topomap \
    --processed data/processed_p3b \
    --raw-root data/erp_core_p3b \
    --out figures/topomaps \
    --times 0.30 0.35 0.39 \
    --peak-window 0.30 0.50
```

**Key parameters:**
- `--raw-root`: Path to original BIDS data (for electrode positions)
- `--times`: Specific time points for topomaps (in seconds)
- `--skip-montage`: Use if electrode positions unavailable

**Outputs:**
- Subject topomaps: `figures/topomaps/subjects/sub-XXX_diff_YYYms.png`
- Group topomaps: `figures/topomaps/group/group_diff_YYYms.png`

### 3. Statistical Testing

Run cluster-based permutation tests:

```bash
python src/paper_analysis_p3b.py cluster \
    --processed data/processed_p3b \
    --out results/stats \
    --n-perm 1000 \
    --tmin -0.1 \
    --tmax 0.6 \
    --roi-chan 12,13,14
```

**Key parameters:**
- `--n-perm`: Number of permutations (1000+ for final analysis)
- `--roi-chan`: Comma-separated channel indices for ROI analysis
- `--tmin/tmax`: Time window for testing

**Outputs:**
- Statistical results: `results/stats/cluster_perm_results.npz`
- Visualization: `results/stats/cluster_Tobs.png`

## Data Requirements

### Processed Data Format
- Files named: `sub-XXX_X.npy` (epochs data) and `sub-XXX_y.npy` (labels)
- Shape: X = (n_epochs, n_channels, n_timepoints), y = (n_epochs,)
- Labels: 1 = target, 0 = standard

### Original Data (for topomaps)
- BIDS-formatted EEG data
- Required files per subject:
  - `sub-XXX_task-P3_channels.tsv`
  - `sub-XXX_task-P3_electrodes.tsv`

## Tips for Publication

1. **ERP Plots**: Use `--smooth-ms 5` for cleaner publication figures
2. **Topomaps**: Include both fixed times and peak-detected times
3. **Statistics**: Use `--n-perm 1000` or higher for final results
4. **Channel Selection**: Channel 12 typically corresponds to Pz in your data

## Troubleshooting

- **No topomaps generated**: Check `--raw-root` path or use `--skip-montage`
- **Time out of range**: Adjust `--times` to be within your data range (≤0.399s)
- **No clusters found**: Normal if no significant effects; check data quality
- **Memory issues**: Reduce `--n-perm` for testing, increase for final analysis

## Output Summary

After running all analyses, you'll have:
- Publication-ready ERP figures with statistics
- Topographic maps showing spatial distribution
- Statistical test results for significance claims
- JSON summaries for reproducible reporting