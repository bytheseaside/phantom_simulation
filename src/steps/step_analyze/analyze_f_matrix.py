#!/usr/bin/env python3
"""
Analyze the F-matrix to identify optimal dipole configurations for simulations.

Features:
- Compute column correlations to assess unit stimuli pattern similarity.
- Calculate column norms to evaluate signal strength.
- Analyze matrix rank, condition number, and singular values.
- Generate heatmaps for visualizing correlations.

Usage:
  python analyze_f_matrix.py --matrix f_matrix_complete/F_matrix.npy --n 8 --out analysis_results

Options:
  --matrix <path>       Path to the F-matrix file in .npy format.
  --out <directory>     Directory to save analysis results (default: analysis_results).
  --abs-corr            Use absolute values for correlation heatmaps (default: True).
  --no-abs-corr         Use signed correlations for heatmaps.
  --annotate            Annotate heatmap cells with numeric values (default: True).
  --no-annotate         Do not annotate heatmap cells.
  --fmt <format>        Format string for numeric annotations (default: "{:.2f}").
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import colors as mcolors
from itertools import combinations
import re

def compute_correlation_matrix(f_matrix):
    """
    Compute pairwise correlation between columns (dipoles).
    Returns correlation matrix where C[i,j] = correlation between dipole i and j.
    """
    corr_matrix = np.corrcoef(f_matrix, rowvar=False)
    return corr_matrix


def compute_column_norms(f_matrix):
    """Compute L2 norm of each column (signal strength)."""
    return np.linalg.norm(f_matrix, axis=0)



def analyze_matrix_properties(f_matrix):
    """Compute rank, condition number, singular values.

    Uses NumPy helpers for rank and condition-number computation so the
    decisions follow NumPy's robust defaults (tolerances tied to machine
    precision and matrix shape). Still compute and return the singular
    values for diagnostics.
    """
    # Compute singular values (kept for reporting)
    _, s, _ = np.linalg.svd(f_matrix, full_matrices=False)

    # Rank computed via NumPy helper (uses a robust SVD-based tolerance)
    rank = int(np.linalg.matrix_rank(f_matrix))

    # Condition number
    try:
        cond = float(np.linalg.cond(f_matrix))
    except Exception:
        cond = np.inf

    return {
        'rank': rank,
        'condition_number': cond,
        'singular_values': s.tolist(),
        'max_singular_value': float(s[0]) if s.size else 0.0,
        'min_singular_value': float(s[-1]) if s.size else 0.0
    }


def save_correlation_heatmap(corr_matrix, case_names, output_path: Path, *, abs_val: bool = True, annotate: bool = True, fmt: str = "{:.2f}"):
    """Save correlation matrix as heatmap.

    Additional features:
    - abs_val: plot absolute value of correlations (0..1) instead of signed (-1..1)
    - annotate: put numeric values in each visible cell
    - fmt: format string for the numeric annotations
    """

    # Parameters are accepted via function arguments (see signature above).
    # Prepare plotting matrix
    mat = np.array(corr_matrix, copy=True)
    if abs_val:
        mat = np.abs(mat)
        vmin, vmax = 0.0, 1.0
        cmap = plt.get_cmap('Blues')
    else:
        vmin, vmax = -1.0, 1.0
        cmap = plt.get_cmap('RdBu_r')

    nmat = mat.shape[0]
    tri_i, tri_j = np.tril_indices(nmat, -1)
    mat[tri_i, tri_j] = 0.0

    _, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xlabel('Dipole Case', fontsize=10)
    ax.set_ylabel('Dipole Case', fontsize=10)
    ax.set_title('Dipole Correlation Matrix', fontsize=14)

    n = len(case_names)
    # Set ticks
    ax.set_xticks(range(n))
    ax.set_xticklabels(case_names, rotation=45, fontsize=6)
    ax.xaxis.set_ticks_position('top')
    ax.set_yticks(range(n))
    ax.set_yticklabels(case_names, rotation=45, fontsize=6)
    ax.yaxis.set_ticks_position('right')

    # Annotate numeric values on the visible cells
    if annotate:
        for i in range(n):
            for j in range(n):
                # Skip lower triangle if masked
                if j < i:
                    continue
                try:
                    val = mat[i, j]
                except Exception:
                    val = corr_matrix[i, j]
                if np.ma.is_masked(val) or (isinstance(val, float) and np.isnan(val)):
                    continue
                ax.text(j, i, fmt.format(float(val)), ha='center', va='center', fontsize=6, color='black')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation' + (' (abs)' if abs_val else ''), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def extract_electrodes(dipole_name):
    """Extract electrode numbers from dipole name (e.g., 'e1e2' -> [1, 2])."""
    matches = re.findall(r'e(\d+)', dipole_name)
    return [int(m) for m in matches]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix', type=Path, required=True, help='Path to F_matrix.npy')
    ap.add_argument('--out', type=Path, default=Path('analysis_results'), 
                    help='Output directory')
    ap.add_argument('--abs-corr', dest='abs_corr', action='store_true', help='Use absolute value of correlations for heatmap')
    ap.add_argument('--no-abs-corr', dest='abs_corr', action='store_false', help='Do not use absolute value of correlations for heatmap')
    ap.set_defaults(abs_corr=True)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--annotate', dest='annotate', action='store_true', help='Annotate cells with numeric values (default)')
    grp.add_argument('--no-annotate', dest='annotate', action='store_false', help='Do not annotate cells')
    ap.set_defaults(annotate=True)
    ap.add_argument('--fmt', dest='fmt', default='{:.2f}', help='Format string for numeric annotations, e.g. "{:.2f}")')

    args = ap.parse_args()
    
    if not args.matrix.exists():
        print(f"ERROR: Matrix file not found: {args.matrix}")
        return
    
    # Load matrix
    print(f"Loading F-matrix from {args.matrix}...")
    f_matrix = np.load(args.matrix)
    print(f"Matrix shape: {f_matrix.shape[0]} probes × {f_matrix.shape[1]} dipoles")

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    
    # 1. Compute matrix properties
    print("\nComputing matrix properties...")
    props = analyze_matrix_properties(f_matrix)
    print(f"  Rank: {props['rank']}")
    print(f"  Condition number: {props['condition_number']:.2e}")
    
    # 2. Compute correlation matrix
    print("\nComputing correlation matrix...")
    corr_matrix = compute_correlation_matrix(f_matrix)

    # Save correlation heatmap
    case_names = [f"e{i}e{j}" for i in range(1, 10) for j in range(i + 1, 10)] # ORDERED LIST OF DIPOLES
    corr_heatmap_path = args.out / "correlation_heatmap.png"
    save_correlation_heatmap(corr_matrix, case_names, corr_heatmap_path, abs_val=args.abs_corr, annotate=args.annotate, fmt=args.fmt)
    print(f"  Saved: {corr_heatmap_path}")
            
    # 3. Save results
    results = {
        'matrix_properties': props,
    }
    
    results_path = args.out / "analysis_results.json"
    with results_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")
    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()
