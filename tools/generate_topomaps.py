#!/usr/bin/env python3
"""
Generate topomaps without MNE dependency
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('.').resolve()
OUTDIR = ROOT / "figures" / "final"

def save(fig, name):
    png = OUTDIR / f"{name}.png"
    svg = OUTDIR / f"{name}.svg"
    fig.savefig(png, bbox_inches='tight', dpi=300)
    fig.savefig(svg, bbox_inches='tight')
    plt.close(fig)
    print("Saved:", png, svg)

# Create simple topomaps using scatter plots
def create_simple_topomap(data, title, filename):
    # Create a simple 2D representation of EEG channels
    # Standard 10-20 system approximate positions
    positions = np.array([
        [0.0, 0.8], [0.3, 0.7], [-0.3, 0.7],  # Fz, F3, F4
        [0.0, 0.5], [0.4, 0.4], [-0.4, 0.4],  # Cz, C3, C4
        [0.0, 0.2], [0.3, 0.1], [-0.3, 0.1],  # Pz, P3, P4
        [0.0, -0.1], [0.2, -0.2], [-0.2, -0.2], # Oz, O1, O2
        [0.6, 0.6], [-0.6, 0.6], [0.6, 0.0], [-0.6, 0.0], # T7, T8, etc.
        [0.6, -0.3], [-0.6, -0.3], [0.1, 0.6], [-0.1, 0.6],
        [0.2, 0.3], [-0.2, 0.3], [0.1, 0.0], [-0.1, 0.0],
        [0.15, -0.05], [-0.15, -0.05]
    ])
    
    # Ensure we have the right number of positions
    if len(positions) < len(data):
        # Add more positions if needed
        extra_positions = np.random.uniform(-0.7, 0.7, (len(data) - len(positions), 2))
        positions = np.vstack([positions, extra_positions])
    
    positions = positions[:len(data)]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Create head outline
    head_circle = plt.Circle((0, 0), 0.8, fill=False, color='black', linewidth=2)
    ax.add_patch(head_circle)
    
    # Add nose
    nose_x = [0, 0]
    nose_y = [0.8, 0.9]
    ax.plot(nose_x, nose_y, 'k-', linewidth=2)
    
    # Plot data as colored circles
    scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                        c=data, s=200, cmap='RdBu_r', 
                        vmin=-np.max(np.abs(data)), vmax=np.max(np.abs(data)))
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Amplitude (µV)')
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    
    save(fig, filename)

# Generate topomaps
try:
    topo_300 = np.load(ROOT / "results" / "topomaps" / "group_diff_300ms.npy")
    create_simple_topomap(topo_300, "P300 Difference Topomap (300ms)", "topomap_group_300ms")
    
    topo_peak = np.load(ROOT / "results" / "topomaps" / "group_diff_peakms.npy")
    create_simple_topomap(topo_peak, "P300 Difference Topomap (Peak)", "topomap_group_peak")
    
    print("✅ Topomaps generated successfully!")
    
except Exception as e:
    print(f"❌ Error generating topomaps: {e}")

# Also create a ML comparison bar chart
try:
    import pandas as pd
    df = pd.read_csv(ROOT / "results" / "ml_baselines" / "loso_baselines_summary.csv")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df['method'], df['auc'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    # Add value labels on bars
    for bar, auc in zip(bars, df['auc']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom')
    
    ax.set_ylabel('AUC Score')
    ax.set_title('ML Method Comparison (LOSO Cross-Validation)')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save(fig, "ml_comparison_barplot")
    print("✅ ML comparison chart generated!")
    
except Exception as e:
    print(f"❌ Error generating ML comparison: {e}")