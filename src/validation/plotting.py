"""
plotting.py

Graph generation for validation results.
All graphs in PNG format with direct labels for clarity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


def set_plot_style():
    """Configure matplotlib for clean, readable plots."""
    plt.style.use('default')
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 12


def plot_probe_comparison_bars(
    probe_names: List[str],
    V_solver: np.ndarray,
    V_full: np.ndarray,
    V_subset: np.ndarray,
    case_name: str,
    output_path: Path
):
    """
    Bar chart comparing V_solver, V_full, V_subset for each probe.
    Side-by-side grouped bars.
    """
    set_plot_style()
    
    n_probes = len(probe_names)
    x = np.arange(n_probes)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Bars
    ax.bar(x - width, V_solver * 1e6, width, label='Solver (Ground Truth)', 
           color='#2ca02c', alpha=0.8)
    ax.bar(x, V_full * 1e6, width, label='F_full (All 36 dipoles)',
           color='#1f77b4', alpha=0.8)
    ax.bar(x + width, V_subset * 1e6, width, label='F_subset (Selected dipoles)',
           color='#ff7f0e', alpha=0.8)
    
    ax.set_xlabel('Probe')
    ax.set_ylabel('Voltage (μV)')
    ax.set_title(f'Probe Measurements Comparison - {case_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(probe_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_bars(
    probe_names: List[str],
    errors_solver_subset: np.ndarray,
    regions: List[str],
    case_name: str,
    output_path: Path
):
    """
    Bar chart of |Solver - F_subset| errors per probe.
    Color-coded by region, sorted by error magnitude.
    """
    set_plot_style()
    
    # Sort by error magnitude
    abs_errors = np.abs(errors_solver_subset)
    sort_idx = np.argsort(abs_errors)[::-1]
    
    sorted_probes = [probe_names[i] for i in sort_idx]
    sorted_errors = abs_errors[sort_idx] * 1e6  # μV
    sorted_regions = [regions[i] for i in sort_idx]
    
    # Color map by region
    unique_regions = list(set(sorted_regions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))
    region_colors = {region: colors[i] for i, region in enumerate(unique_regions)}
    bar_colors = [region_colors[r] for r in sorted_regions]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars = ax.bar(range(len(sorted_probes)), sorted_errors, color=bar_colors, alpha=0.8)
    
    ax.set_xlabel('Probe')
    ax.set_ylabel('|Error| (μV)')
    ax.set_title(f'Reconstruction Errors (|Solver - F_subset|) - {case_name}')
    ax.set_xticks(range(len(sorted_probes)))
    ax.set_xticklabels(sorted_probes, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    handles = [plt.Rectangle((0,0),1,1, color=region_colors[r]) for r in unique_regions]
    ax.legend(handles, unique_regions, loc='upper right', ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_comparison(
    probe_names: List[str],
    V_solver: np.ndarray,
    V_subset: np.ndarray,
    regions: List[str],
    case_name: str,
    output_path: Path
):
    """
    Scatter plot: Solver (x) vs F_subset (y).
    Diagonal line = perfect match.
    Points labeled with probe abbreviations.
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Convert to μV
    x = V_solver * 1e6
    y = V_subset * 1e6
    
    # Color by region
    unique_regions = list(set(regions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))
    region_colors = {region: colors[i] for i, region in enumerate(unique_regions)}
    
    # Plot points
    for i, (xi, yi, probe, region) in enumerate(zip(x, y, probe_names, regions)):
        ax.scatter(xi, yi, color=region_colors[region], s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add abbreviated label
        label = probe[:3] if len(probe) > 3 else probe
        ax.annotate(label, (xi, yi), xytext=(3, 3), textcoords='offset points',
                   fontsize=7, alpha=0.8)
    
    # Diagonal line
    all_vals = np.concatenate([x, y])
    lim_min, lim_max = all_vals.min(), all_vals.max()
    margin = (lim_max - lim_min) * 0.1
    ax.plot([lim_min - margin, lim_max + margin], [lim_min - margin, lim_max + margin],
           'k--', alpha=0.5, linewidth=1, label='Perfect match')
    
    ax.set_xlabel('Solver (Ground Truth) [μV]')
    ax.set_ylabel('F_subset (Reconstruction) [μV]')
    ax.set_title(f'Solver vs F_subset Correlation - {case_name}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=region_colors[r],
                         markersize=8, alpha=0.7, markeredgecolor='black', markeredgewidth=0.5)
              for r in unique_regions]
    ax.legend(handles + [plt.Line2D([0], [0], color='k', linestyle='--')],
             unique_regions + ['Perfect match'], loc='upper left', ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rmse_comparison(
    solver_vs_full_rmse: float,
    solver_vs_subset_rmse: float,
    full_vs_subset_rmse: float,
    output_path: Path
):
    """
    Bar chart comparing RMSE across all test cases.
    Shows: Solver-Full, Solver-Subset, Full-Subset
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Solver vs\nF_full', 'Solver vs\nF_subset', 'F_full vs\nF_subset']
    rmse_values = [solver_vs_full_rmse * 1e6, solver_vs_subset_rmse * 1e6, full_vs_subset_rmse * 1e6]
    colors = ['#2ca02c', '#d62728', '#9467bd']
    
    bars = ax.bar(categories, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('RMSE (μV)')
    ax.set_title('RMSE Comparison (Mean across all test cases)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_regional_heatmap(
    test_case_names: List[str],
    regional_data: List[Dict[str, Dict[str, float]]],
    output_path: Path
):
    """
    Heatmap: Regions (rows) × Test cases (cols).
    Color = mean error in that region for that case.
    """
    set_plot_style()
    
    # Extract region names
    all_regions = set()
    for case_data in regional_data:
        all_regions.update(case_data.keys())
    regions = sorted(all_regions)
    
    # Build matrix
    matrix = np.zeros((len(regions), len(test_case_names)))
    for j, case_data in enumerate(regional_data):
        for i, region in enumerate(regions):
            if region in case_data:
                matrix[i, j] = case_data[region]['mean_error'] * 1e6  # μV
            else:
                matrix[i, j] = np.nan
    
    fig, ax = plt.subplots(figsize=(max(10, len(test_case_names) * 0.6), max(6, len(regions) * 0.4)))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    ax.set_xticks(range(len(test_case_names)))
    ax.set_xticklabels(test_case_names, rotation=45, ha='right')
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions)
    
    # Add values in cells
    for i in range(len(regions)):
        for j in range(len(test_case_names)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > np.nanmax(matrix) * 0.6 else 'black'
                ax.text(j, i, f'{val:.0f}',
                       ha='center', va='center', color=text_color, fontsize=7)
    
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Brain Region')
    ax.set_title('Regional Errors (μV): Solver vs F_subset')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean |Error| (μV)', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_algorithm_comparison_boxplot(
    algorithm_results: Dict[str, List[float]],
    output_path: Path
):
    """
    Box plot showing RMSE distribution per algorithm.
    Shows variability across different S matrices.
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    algorithms = sorted(algorithm_results.keys())
    data = [algorithm_results[algo] for algo in algorithms]
    
    bp = ax.boxplot(data, labels=algorithms, patch_artist=True,
                    showmeans=True, meanline=True)
    
    # Color boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('RMSE (μV)')
    ax.set_xlabel('Algorithm')
    ax.set_title('RMSE Distribution by Algorithm (Solver vs F_subset)')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_best_per_algorithm_bars(
    algorithm_names: List[str],
    best_rmse: List[float],
    output_path: Path
):
    """
    Bar chart showing best RMSE achieved by each algorithm.
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by RMSE
    sorted_idx = np.argsort(best_rmse)
    sorted_algos = [algorithm_names[i] for i in sorted_idx]
    sorted_rmse = [best_rmse[i] for i in sorted_idx]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_algos)))
    bars = ax.barh(range(len(sorted_algos)), sorted_rmse, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_yticks(range(len(sorted_algos)))
    ax.set_yticklabels(sorted_algos)
    ax.set_xlabel('Best RMSE (μV)')
    ax.set_title('Best Performance per Algorithm (Solver vs F_subset)')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_rmse)):
        ax.text(val + max(sorted_rmse) * 0.02, i, f'{val:.1f}',
               va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
