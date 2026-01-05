#!/usr/bin/env python3
"""
Analyze a forward transfer matrix F for regularized pseudoinverse computation.

Metrics computed:
- Matrix properties: rank, condition number, singular value spectrum
- Case correlation: pairwise column correlations (redundancy analysis)
- Probe statistics: mean, std, CV for each row (spatial discriminability)
- Regularization analysis: condition number vs lambda, L-curve for test reconstructions

Usage:
  python analyze_transfer_matrix.py --matrix F.npy --out results/
  python analyze_transfer_matrix.py --matrix F.npy --files case1.npy case2.npy --out results/

Outputs:
  - matrix_properties.json: rank, condition number, singular values
  - correlation_heatmap.svg: case correlation matrix
  - singular_values.svg: singular value spectrum
  - lambda_analysis.svg: condition number and L-curve plots
  - analysis_report.json: full report with all metrics
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


def compute_correlation_matrix(F: np.ndarray) -> np.ndarray:
    """Compute pairwise correlation between columns (cases)."""
    return np.corrcoef(F, rowvar=False)


def compute_correlation_stats(corr_matrix: np.ndarray) -> dict:
    """Compute summary statistics of off-diagonal correlations."""
    n = corr_matrix.shape[0]
    # Extract lower triangle (excluding diagonal)
    tril_idx = np.tril_indices(n, k=-1)
    off_diag = corr_matrix[tril_idx]
    
    return {
        'mean': float(np.mean(off_diag)),
        'std': float(np.std(off_diag)),
        'min': float(np.min(off_diag)),
        'max': float(np.max(off_diag)),
        'rms': float(np.sqrt(np.mean(off_diag**2))),
        'n_pairs': len(off_diag),
    }


def compute_probe_statistics(F: np.ndarray, probe_names: list = None) -> dict:
    """Compute row-wise statistics (mean, std, CV) for each probe."""
    n_probes, n_cases = F.shape
    
    if probe_names is None:
        probe_names = [f"probe_{i}" for i in range(n_probes)]
    
    probe_stats = []
    for i in range(n_probes):
        row = F[i, :]
        mean_val = float(np.mean(row))
        std_val = float(np.std(row))
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else float('inf')
        
        probe_stats.append({
            'probe_name': probe_names[i],
            'mean': mean_val,
            'std': std_val,
            'cv': float(cv) if not np.isinf(cv) else None,
        })
    
    # Summary
    cvs = [p['cv'] for p in probe_stats if p['cv'] is not None]
    stds = [p['std'] for p in probe_stats]
    
    return {
        'per_probe': probe_stats,
        'summary': {
            'mean_cv': float(np.mean(cvs)) if cvs else None,
            'min_cv': float(np.min(cvs)) if cvs else None,
            'max_cv': float(np.max(cvs)) if cvs else None,
            'mean_std': float(np.mean(stds)),
        }
    }


def compute_matrix_properties(F: np.ndarray) -> dict:
    """Compute rank, condition number, and singular values."""
    _, s, _ = np.linalg.svd(F, full_matrices=False)
    rank = int(np.linalg.matrix_rank(F))
    cond = float(np.linalg.cond(F))
    
    return {
        'shape': list(F.shape),
        'n_probes': F.shape[0],
        'n_cases': F.shape[1],
        'rank': rank,
        'condition_number': cond,
        'singular_values': s.tolist(),
        'max_singular_value': float(s[0]) if s.size else 0.0,
        'min_singular_value': float(s[-1]) if s.size else 0.0,
        'sv_ratio': float(s[0] / s[-1]) if s.size and s[-1] > 0 else float('inf'),
    }


def compute_regularized_condition(F: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Compute condition number of (F^T F + lambda^2 I) for each lambda."""
    FTF = F.T @ F
    n = FTF.shape[0]
    I = np.eye(n)
    
    conds = []
    for lam in lambdas:
        A = FTF + (lam ** 2) * I
        conds.append(np.linalg.cond(A))
    
    return np.array(conds)


def compute_regularized_inverse(F: np.ndarray, lam: float, L: np.ndarray = None) -> np.ndarray:
    """
    Compute regularized pseudoinverse: B = (F^T F + lambda^2 L^T L)^{-1} F^T
    
    If L is None, uses L = I (Tikhonov regularization).
    """
    n_cases = F.shape[1]
    if L is None:
        L = np.eye(n_cases)
    
    FTF = F.T @ F
    LTL = L.T @ L
    A = FTF + (lam ** 2) * LTL
    
    return np.linalg.solve(A, F.T)


def compute_lcurve_point(F: np.ndarray, y: np.ndarray, lam: float, L: np.ndarray = None):
    """
    Compute one point on the L-curve.
    
    Returns:
        (residual_norm, solution_norm, x_lambda)
    """
    B = compute_regularized_inverse(F, lam, L)
    x_lambda = B @ y
    
    residual = F @ x_lambda - y
    residual_norm = np.linalg.norm(residual)
    solution_norm = np.linalg.norm(x_lambda)
    
    return residual_norm, solution_norm, x_lambda


def compute_lcurve(F: np.ndarray, y: np.ndarray, lambdas: np.ndarray, L: np.ndarray = None):
    """
    Compute L-curve data for a range of lambdas.
    
    Returns:
        dict with residual_norms, solution_norms arrays
    """
    residual_norms = []
    solution_norms = []
    
    for lam in lambdas:
        res_norm, sol_norm, _ = compute_lcurve_point(F, y, lam, L)
        residual_norms.append(res_norm)
        solution_norms.append(sol_norm)
    
    return {
        'lambdas': lambdas.tolist(),
        'residual_norms': residual_norms,
        'solution_norms': solution_norms,
    }


def find_lcurve_corner(residual_norms, solution_norms, lambdas):
    """
    Find the corner of the L-curve using maximum curvature.
    
    Returns the index and lambda value at the corner.
    """
    # Use log-log coordinates
    log_res = np.log10(np.array(residual_norms) + 1e-16)
    log_sol = np.log10(np.array(solution_norms) + 1e-16)
    
    # Compute curvature using finite differences
    n = len(lambdas)
    if n < 3:
        return 0, lambdas[0]
    
    # First and second derivatives
    dx = np.gradient(log_res)
    dy = np.gradient(log_sol)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature: kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^{3/2}
    curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-16)**1.5
    
    # Find maximum curvature (corner)
    corner_idx = np.argmax(curvature)
    
    return corner_idx, lambdas[corner_idx]


def reconstruction_error(x_true: np.ndarray, x_reconstructed: np.ndarray) -> dict:
    """Compute reconstruction error metrics."""
    diff = x_true - x_reconstructed
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    rel_error = float(np.linalg.norm(diff) / (np.linalg.norm(x_true) + 1e-16))
    correlation = float(np.corrcoef(x_true.flatten(), x_reconstructed.flatten())[0, 1])
    
    return {
        'mse': mse,
        'rmse': rmse,
        'relative_error': rel_error,
        'correlation': correlation,
    }


# =============================================================================
# Plotting functions
# =============================================================================

def plot_correlation_heatmap(corr_matrix: np.ndarray, case_names: list, output_path: Path,
                             annotate: bool = False, fmt: str = "{:.2f}"):
    """Save correlation matrix as heatmap."""
    n = corr_matrix.shape[0]
    
    # Lower triangle only
    mat = corr_matrix.copy()
    tri_i, tri_j = np.triu_indices(n, 1)
    mat[tri_i, tri_j] = np.nan
    
    fig_size = max(8, n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=150)
    
    im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    
    ax.set_xlabel('Case', fontsize=12, fontweight='bold')
    ax.set_ylabel('Case', fontsize=12, fontweight='bold')
    ax.set_title('Case Correlation Matrix', fontsize=14, fontweight='bold')
    
    if n <= 20:
        ax.set_xticks(range(n))
        ax.set_xticklabels(case_names, rotation=90, ha='center', fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(case_names, fontsize=8)
    
    if annotate and n <= 12:
        for i in range(n):
            for j in range(i + 1):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, fmt.format(val), ha='center', va='center', fontsize=7)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_singular_values(singular_values: list, output_path: Path):
    """Plot singular value spectrum."""
    s = np.array(singular_values)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Linear scale
    ax1 = axes[0]
    ax1.plot(range(1, len(s) + 1), s, 'b.-', markersize=8)
    ax1.set_xlabel('Index', fontsize=12)
    ax1.set_ylabel('Singular Value', fontsize=12)
    ax1.set_title('Singular Values (Linear)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2 = axes[1]
    ax2.semilogy(range(1, len(s) + 1), s, 'b.-', markersize=8)
    ax2.set_xlabel('Index', fontsize=12)
    ax2.set_ylabel('Singular Value', fontsize=12)
    ax2.set_title('Singular Values (Log)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_lambda_analysis(lambdas: np.ndarray, conds: np.ndarray, 
                         lcurve_data: dict, corner_idx: int, corner_lambda: float,
                         output_path: Path):
    """Plot lambda analysis: condition number and L-curve."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Condition number vs lambda
    ax1 = axes[0]
    ax1.loglog(lambdas, conds, 'b.-')
    ax1.axvline(corner_lambda, color='r', linestyle='--', label=f'Corner λ={corner_lambda:.2e}')
    ax1.set_xlabel('λ', fontsize=12)
    ax1.set_ylabel('Condition Number', fontsize=12)
    ax1.set_title('Regularized System Conditioning', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. L-curve
    ax2 = axes[1]
    res_norms = lcurve_data['residual_norms']
    sol_norms = lcurve_data['solution_norms']
    ax2.loglog(res_norms, sol_norms, 'b.-')
    ax2.loglog(res_norms[corner_idx], sol_norms[corner_idx], 'ro', markersize=10, 
               label=f'Corner λ={corner_lambda:.2e}')
    ax2.set_xlabel('||Fx - y|| (Residual Norm)', fontsize=12)
    ax2.set_ylabel('||x|| (Solution Norm)', fontsize=12)
    ax2.set_title('L-Curve', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Error metrics vs lambda (if x_true was provided)
    ax3 = axes[2]
    ax3.semilogx(lambdas, res_norms, 'b.-', label='Residual ||Fx-y||')
    ax3.set_xlabel('λ', fontsize=12)
    ax3.set_ylabel('Residual Norm', fontsize=12)
    ax3.axvline(corner_lambda, color='r', linestyle='--', label=f'Corner λ={corner_lambda:.2e}')
    ax3.set_title('Residual vs λ', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstruction_test(lambdas: np.ndarray, errors: list, corner_lambda: float, output_path: Path):
    """Plot reconstruction error vs lambda for multiple test cases."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, err_data in enumerate(errors):
        rel_errors = [e['relative_error'] for e in err_data]
        ax.semilogx(lambdas, rel_errors, '.-', label=f'Test case {i+1}', alpha=0.7)
    
    ax.axvline(corner_lambda, color='r', linestyle='--', linewidth=2, label=f'Corner λ={corner_lambda:.2e}')
    ax.set_xlabel('λ', fontsize=12)
    ax.set_ylabel('Relative Error ||x_λ - x_true|| / ||x_true||', fontsize=12)
    ax.set_title('Reconstruction Error vs Regularization', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze forward transfer matrix for regularized inversion.')
    parser.add_argument('--matrix', type=Path, required=True, help='Path to F matrix (.npy)')
    parser.add_argument('--files', nargs='+', default=None, help='Original file paths for case names')
    parser.add_argument('--out', type=Path, default=Path('analysis_results'), help='Output directory')
    parser.add_argument('--n-test-cases', type=int, default=5, help='Number of random test cases for L-curve')
    parser.add_argument('--lambda-min', type=float, default=1e-6, help='Minimum lambda for sweep')
    parser.add_argument('--lambda-max', type=float, default=1e2, help='Maximum lambda for sweep')
    parser.add_argument('--n-lambdas', type=int, default=50, help='Number of lambda values in sweep')
    
    args = parser.parse_args()
    
    if not args.matrix.exists():
        print(f"ERROR: Matrix file not found: {args.matrix}")
        return
    
    # Load matrix
    print(f"Loading F matrix from {args.matrix}...")
    F = np.load(args.matrix)
    n_probes, n_cases = F.shape
    print(f"Matrix shape: {n_probes} probes × {n_cases} cases")
    
    # Extract case names
    if args.files:
        case_names = [Path(f).stem for f in args.files]
    else:
        case_names = [f"case_{i}" for i in range(n_cases)]
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # 1. Matrix properties
    # ==========================================================================
    print("\n1. Computing matrix properties...")
    props = compute_matrix_properties(F)
    print(f"   Shape: {props['shape']}")
    print(f"   Rank: {props['rank']} (max possible: {min(n_probes, n_cases)})")
    print(f"   Condition number: {props['condition_number']:.2e}")
    print(f"   SV ratio (max/min): {props['sv_ratio']:.2e}")
    
    # Plot singular values
    sv_path = args.out / "singular_values.svg"
    plot_singular_values(props['singular_values'], sv_path)
    print(f"   Saved: {sv_path}")
    
    # ==========================================================================
    # 2. Case correlation
    # ==========================================================================
    print("\n2. Computing case correlations...")
    corr_matrix = compute_correlation_matrix(F)
    corr_stats = compute_correlation_stats(corr_matrix)
    print(f"   Mean correlation: {corr_stats['mean']:.3f}")
    print(f"   RMS correlation: {corr_stats['rms']:.3f}")
    print(f"   Min/Max: {corr_stats['min']:.3f} / {corr_stats['max']:.3f}")
    
    # Plot correlation heatmap
    corr_path = args.out / "correlation_heatmap.svg"
    plot_correlation_heatmap(corr_matrix, case_names, corr_path, annotate=(n_cases <= 12))
    print(f"   Saved: {corr_path}")
    
    # ==========================================================================
    # 3. Probe statistics
    # ==========================================================================
    print("\n3. Computing probe statistics...")
    probe_stats = compute_probe_statistics(F)
    print(f"   Mean CV across probes: {probe_stats['summary']['mean_cv']:.2f}")
    print(f"   Min/Max CV: {probe_stats['summary']['min_cv']:.2f} / {probe_stats['summary']['max_cv']:.2f}")
    
    # ==========================================================================
    # 4. Lambda analysis
    # ==========================================================================
    print("\n4. Analyzing regularization parameter λ...")
    lambdas = np.logspace(np.log10(args.lambda_min), np.log10(args.lambda_max), args.n_lambdas)
    
    # Condition number vs lambda
    print("   Computing condition numbers...")
    conds = compute_regularized_condition(F, lambdas)
    
    # Generate random test cases for L-curve
    print(f"   Generating {args.n_test_cases} random test cases...")
    np.random.seed(42)  # Reproducibility
    test_cases = []
    all_errors = []
    
    for i in range(args.n_test_cases):
        x_true = np.random.uniform(-5, 5, n_cases)
        y = F @ x_true
        test_cases.append({'x_true': x_true, 'y': y})
    
    # Compute L-curve for first test case
    print("   Computing L-curve...")
    lcurve_data = compute_lcurve(F, test_cases[0]['y'], lambdas)
    corner_idx, corner_lambda = find_lcurve_corner(
        lcurve_data['residual_norms'], 
        lcurve_data['solution_norms'], 
        lambdas
    )
    print(f"   L-curve corner at λ = {corner_lambda:.2e}")
    
    # Compute reconstruction errors for all test cases
    print("   Computing reconstruction errors...")
    for tc in test_cases:
        errors_for_case = []
        for lam in lambdas:
            _, _, x_lambda = compute_lcurve_point(F, tc['y'], lam)
            err = reconstruction_error(tc['x_true'], x_lambda)
            errors_for_case.append(err)
        all_errors.append(errors_for_case)
    
    # Find optimal lambda based on average reconstruction error
    avg_rel_errors = np.mean([[e['relative_error'] for e in errs] for errs in all_errors], axis=0)
    opt_idx = np.argmin(avg_rel_errors)
    opt_lambda = lambdas[opt_idx]
    print(f"   Optimal λ (min avg error): {opt_lambda:.2e}")
    print(f"   Relative error at optimal λ: {avg_rel_errors[opt_idx]:.4f}")
    
    # Plot lambda analysis
    lambda_path = args.out / "lambda_analysis.svg"
    plot_lambda_analysis(lambdas, conds, lcurve_data, corner_idx, corner_lambda, lambda_path)
    print(f"   Saved: {lambda_path}")
    
    # Plot reconstruction errors
    recon_path = args.out / "reconstruction_error.svg"
    plot_reconstruction_test(lambdas, all_errors, opt_lambda, recon_path)
    print(f"   Saved: {recon_path}")
    
    # ==========================================================================
    # 5. Save full report
    # ==========================================================================
    print("\n5. Saving analysis report...")
    report = {
        'matrix_file': str(args.matrix),
        'matrix_properties': props,
        'correlation_statistics': corr_stats,
        'probe_statistics': probe_stats,
        'regularization_analysis': {
            'lambda_range': [args.lambda_min, args.lambda_max],
            'n_lambdas': args.n_lambdas,
            'lcurve_corner_lambda': float(corner_lambda),
            'optimal_lambda_by_error': float(opt_lambda),
            'optimal_relative_error': float(avg_rel_errors[opt_idx]),
            'condition_at_optimal': float(conds[opt_idx]),
        },
        'recommendations': {
            'suggested_lambda': float(opt_lambda),
            'notes': []
        }
    }
    
    # Add recommendations
    if props['rank'] < min(n_probes, n_cases):
        report['recommendations']['notes'].append(
            f"Matrix is rank-deficient (rank={props['rank']} < {min(n_probes, n_cases)}). "
            "Consider removing redundant cases."
        )
    
    if corr_stats['rms'] > 0.5:
        report['recommendations']['notes'].append(
            f"High case correlation (RMS={corr_stats['rms']:.3f}). "
            "Cases may be too similar; consider subset selection."
        )
    
    if props['condition_number'] > 1e6:
        report['recommendations']['notes'].append(
            f"Very high condition number ({props['condition_number']:.2e}). "
            "Regularization is essential."
        )
    
    report_path = args.out / "analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   Saved: {report_path}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Matrix: {n_probes} probes × {n_cases} cases")
    print(f"Rank: {props['rank']}")
    print(f"Condition number: {props['condition_number']:.2e}")
    print(f"Case correlation (RMS): {corr_stats['rms']:.3f}")
    print(f"Probe discrimination (mean CV): {probe_stats['summary']['mean_cv']:.2f}")
    print(f"Suggested λ: {opt_lambda:.2e}")
    print(f"Expected relative error: {avg_rel_errors[opt_idx]:.4f} ({avg_rel_errors[opt_idx]*100:.2f}%)")
    print("="*60)


if __name__ == '__main__':
    main()
