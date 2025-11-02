#!/usr/bin/env python3
"""
plot_validation_results.py

Create visualization plots from validation results.
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from pathlib import Path
import json

def plot_rmse_vs_condition(summary_csv: Path, output_dir: Path):
    """Scatter plot: RMSE vs Condition Number."""
    df = pd.read_csv(summary_csv)
    
    # Remove duplicates (keep best of each algorithm)
    df = df.drop_duplicates(subset=['Algorithm', 'N Dipoles', 'Condition Number'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by algorithm
    algorithms = df['Algorithm'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    for algo, color in zip(algorithms, colors):
        algo_data = df[df['Algorithm'] == algo]
        ax.scatter(algo_data['Condition Number'], 
                  algo_data['RMSE Mean (μV)'] / 1000,  # Convert to mV
                  label=f'Algorithm {algo}',
                  s=100, alpha=0.7, color=color)
    
    ax.set_xlabel('Condition Number (κ)', fontsize=12)
    ax.set_ylabel('RMSE (mV)', fontsize=12)
    ax.set_title('Reconstruction Error vs Condition Number', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_vs_condition.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'rmse_vs_condition.png'}")


def plot_rmse_vs_ndipoles(summary_csv: Path, output_dir: Path):
    """Scatter plot: RMSE vs Number of Dipoles."""
    df = pd.read_csv(summary_csv)
    df = df.drop_duplicates(subset=['Algorithm', 'N Dipoles', 'Condition Number'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    algorithms = df['Algorithm'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    for algo, color in zip(algorithms, colors):
        algo_data = df[df['Algorithm'] == algo]
        ax.scatter(algo_data['N Dipoles'], 
                  algo_data['RMSE Mean (μV)'] / 1000,
                  label=f'Algorithm {algo}',
                  s=100, alpha=0.7, color=color)
    
    ax.set_xlabel('Number of Dipoles Selected', fontsize=12)
    ax.set_ylabel('RMSE (mV)', fontsize=12)
    ax.set_title('Reconstruction Error vs Dipole Count', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_vs_ndipoles.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'rmse_vs_ndipoles.png'}")


def plot_algorithm_ranking(summary_csv: Path, output_dir: Path):
    """Bar chart: Algorithm ranking by RMSE."""
    df = pd.read_csv(summary_csv)
    
    # Keep only best result per algorithm
    best_per_algo = df.loc[df.groupby('Algorithm')['RMSE Mean (μV)'].idxmin()]
    best_per_algo = best_per_algo.sort_values('RMSE Mean (μV)')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(range(len(best_per_algo)), 
                   best_per_algo['RMSE Mean (μV)'] / 1000,
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(best_per_algo))))
    
    ax.set_yticks(range(len(best_per_algo)))
    
    # Label with algorithm + dipole count + condition number
    labels = [f"{row['Algorithm']} (n={row['N Dipoles']}, κ={row['Condition Number']:.1f})"
              for _, row in best_per_algo.iterrows()]
    ax.set_yticklabels(labels, fontsize=10)
    
    ax.set_xlabel('RMSE (mV)', fontsize=12)
    ax.set_title('Best Performance per Algorithm', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'algorithm_ranking.png'}")


def plot_spatial_errors(spatial_csv: Path, summary_csv: Path, output_dir: Path):
    """Heatmap: Regional errors by algorithm."""
    df = pd.read_csv(spatial_csv)
    
    # Get best result per algorithm
    summary_df = pd.read_csv(summary_csv)
    best_per_algo = summary_df.loc[summary_df.groupby('Algorithm')['RMSE Mean (μV)'].idxmin()]
    best_results = best_per_algo['Result Name'].tolist()
    
    # Filter spatial data to best results only
    df_best = df[df['Result'].isin(best_results)]
    
    # Pivot: rows=regions, cols=algorithms
    pivot = df_best.pivot_table(
        index='Region',
        columns='Algorithm',
        values='Mean Error (μV)',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(pivot.values / 1000, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0, fontsize=10)
    
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    
    # Add values in cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j] / 1000
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}',
                       ha="center", va="center",
                       color="black" if val < 4 else "white",
                       fontsize=8)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Brain Region', fontsize=12)
    ax.set_title('Regional Reconstruction Errors (mV)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean Error (mV)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_errors_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / 'spatial_errors_heatmap.png'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot validation results')
    parser.add_argument('--validation-dir', type=Path, default=Path('validation_reports'),
                       help='Directory with validation results')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for plots (default: same as validation-dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.validation_dir
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_csv = args.validation_dir / 'validation_summary.csv'
    spatial_csv = args.validation_dir / 'spatial_analysis.csv'
    
    print("Generating validation plots...")
    
    if summary_csv.exists():
        plot_rmse_vs_condition(summary_csv, args.output_dir)
        plot_rmse_vs_ndipoles(summary_csv, args.output_dir)
        plot_algorithm_ranking(summary_csv, args.output_dir)
    else:
        print(f"ERROR: {summary_csv} not found")
    
    if spatial_csv.exists():
        plot_spatial_errors(spatial_csv, summary_csv, args.output_dir)
    else:
        print(f"ERROR: {spatial_csv} not found")
    
    print("\nDone! Generated:")
    print(f"  - rmse_vs_condition.png")
    print(f"  - rmse_vs_ndipoles.png")
    print(f"  - algorithm_ranking.png")
    print(f"  - spatial_errors_heatmap.png")


if __name__ == '__main__':
    main()
