#!/usr/bin/env python3
"""
Analyze the F-matrix to identify optimal dipole configurations for simulations.

Features:
- Compute column correlations to assess unit stimuli pattern similarity.
- Calculate column norms to evaluate signal strength.
- Analyze matrix rank, condition number, and singular values.
- Generate heatmaps for visualizing correlations.

Usage:
  python analyze_f_matrix.py --matrix f_matrix_complete/F_matrix.npy --out analysis_results

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
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_correlation_matrix(f_matrix):
    """Compute pairwise correlation between columns (cases)."""
    return np.corrcoef(f_matrix, rowvar=False)


def compute_probe_statistics(f_matrix, probe_names=None, case_names=None):
    """Compute row-wise statistics (mean, std, CV) for each probe across all cases.
    
    Parameters:
        f_matrix: (n_probes, n_cases) array
        probe_names: list of probe names
        case_names: list of case names
    
    Returns:
        dict with probe statistics and summary metrics
    """
    n_probes, n_cases = f_matrix.shape
    
    if probe_names is None:
        probe_names = [f"probe_{i}" for i in range(n_probes)]
    
    if case_names is None:
        case_names = [f"case_{i}" for i in range(n_cases)]
    
    probe_stats = []
    for i in range(n_probes):
        row = f_matrix[i, :]
        mean_val = float(np.mean(row))
        std_val = float(np.std(row))
        # Coefficient of variation: std / |mean| (handle near-zero means)
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else np.inf
        
        # Per-case values as a dict: {case_name: value}
        values_per_case = {case_names[j]: float(row[j]) for j in range(n_cases)}
        
        probe_stats.append({
            'probe_name': probe_names[i],
            'mean': mean_val,
            'std': std_val,
            'cv': float(cv) if not np.isinf(cv) else None,
            'values_per_case': values_per_case
        })
    
    # Summary metrics across all probes
    stds = [p['std'] for p in probe_stats]
    cvs = [p['cv'] for p in probe_stats if p['cv'] is not None]
    
    summary = {
        'mean_of_stds': float(np.mean(stds)),
        'std_of_stds': float(np.std(stds)),
        'mean_of_cvs': float(np.mean(cvs)) if cvs else None,
        'max_std': float(np.max(stds)),
        'min_std': float(np.min(stds)),
    }
    
    return {
        'per_probe': probe_stats,
        'summary': summary
    }


def analyze_matrix_properties(f_matrix):
    """Compute rank, condition number, and singular values."""
    _, s, _ = np.linalg.svd(f_matrix, full_matrices=False)
    rank = int(np.linalg.matrix_rank(f_matrix))
    cond = float(np.linalg.cond(f_matrix))

    return {
        'rank': rank,
        'condition_number': cond,
        'singular_values': s.tolist(),
        'max_singular_value': float(s[0]) if s.size else 0.0,
        'min_singular_value': float(s[-1]) if s.size else 0.0
    }


def save_correlation_heatmap(corr_matrix, case_names, output_path: Path, *, abs_val: bool = True, annotate: bool = True, fmt: str = "{:.2f}"):
    """Save correlation matrix as heatmap with optional absolute values and annotations."""
    # Prepare plotting matrix
    mat = np.array(corr_matrix, copy=True)
    if abs_val:
        mat = np.abs(mat)
        vmin, vmax = 0.0, 1.0
        cmap = plt.get_cmap('Greens')
    else:
        vmin, vmax = -1.0, 1.0
        cmap = plt.get_cmap('PRGn')

    n_cases = mat.shape[0]
    tri_i, tri_j = np.triu_indices(n_cases, 1)
    mat[tri_i, tri_j] = 0.0

    fig_size = max(24, n_cases * 1.5)
    _, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=300)

    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xlabel('Case', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_ylabel('Case', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_title('Case Correlation Matrix', fontsize=32, fontweight='bold', pad=30)

    n_labels = min(n_cases, len(case_names))
    
    tick_fontsize = 22
    
    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(case_names[:n_labels], rotation=45, ha='right',
                       fontsize=tick_fontsize)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    
    ax.set_yticks(range(n_labels))
    ax.set_yticklabels(case_names[:n_labels], rotation=45, ha='right', fontsize=tick_fontsize)
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('left')

    cell_fontsize = 18
    if annotate:
        import matplotlib.patheffects as path_effects
        for i in range(n_cases):
            for j in range(i + 1):  # Lower triangle only
                val = mat[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, fmt.format(float(val)), ha='center', va='center', 
                                  fontsize=cell_fontsize, fontweight='bold', color='black')
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=2, foreground='white'),
                        path_effects.Normal()
                    ])

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation' + (' (abs)' if abs_val else ''), fontsize=24, fontweight='bold')
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


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
    print(f"Matrix shape: {f_matrix.shape[0]} probes × {f_matrix.shape[1]} cases")

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

    # Load case names and probe names from metadata.json (created by build_f_matrix.py)
    metadata_path = args.matrix.parent / "metadata.json"
    probe_names = None
    if metadata_path.exists():
        with metadata_path.open('r') as f:
            metadata = json.load(f)
            case_names = metadata.get('cases', [f"case_{i}" for i in range(f_matrix.shape[1])])
            # Try to get probe names from metadata (may not exist in older versions)
            probe_names = metadata.get('probe_names', None)
    else:
        print(f"  WARNING: metadata.json not found at {metadata_path}, using generic case names")
        case_names = [f"case_{i}" for i in range(f_matrix.shape[1])]
    
    # Save correlation heatmap
    corr_heatmap_path = args.out / "correlation_heatmap.svg"
    save_correlation_heatmap(corr_matrix, case_names, corr_heatmap_path, abs_val=args.abs_corr, annotate=args.annotate, fmt=args.fmt)
    print(f"  Saved: {corr_heatmap_path}")
    
    # 3. Compute probe statistics (row-wise analysis)
    print("\nComputing probe statistics...")
    probe_stats = compute_probe_statistics(f_matrix, probe_names, case_names)
    print(f"  Mean of probe stds: {probe_stats['summary']['mean_of_stds']:.3e}")
    if probe_stats['summary']['mean_of_cvs'] is not None:
        print(f"  Mean of probe CVs: {probe_stats['summary']['mean_of_cvs']:.3f}")
                
    # 4. Save results
    results = {
        'matrix_properties': props,
        'probe_statistics': probe_stats,
    }
    
    results_path = args.out / "analysis_results.json"
    with results_path.open('w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")
    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()
