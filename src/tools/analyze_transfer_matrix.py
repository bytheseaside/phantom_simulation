#!/usr/bin/env python3
"""
Metrics computed:
- Matrix properties: rank, condition number, singular value spectrum
- Case correlation: pairwise column correlations (redundancy analysis)
- Probe statistics: mean, std, CV for each row (spatial discriminability)
- Regularization analysis: GCV and L-curve methods for lambda selection

Usage:
  python analyze_transfer_matrix.py --matrix F.npy --out results/
  python analyze_transfer_matrix.py --matrix F.npy --case-names E1 E2 E3 --out results/
  python analyze_transfer_matrix.py --matrix F.npy --L-matrix L.npy --out results/

Outputs:
  - correlation_heatmap.svg: case correlation matrix
  - singular_values.svg: singular value spectrum
  - gcv_curve.svg: GCV score vs lambda (cherab-inversion)
  - lcurve.svg: L-curve plot (cherab-inversion)
  - reconstruction_error.svg: reconstruction error vs lambda
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

# cherab-inversion for GCV and L-curve based lambda selection
from cherab.inversion import GCV, Lcurve, compute_svd

# 10-20 International System EEG probe names (21 probes)
PROBE_NAMES_10_20 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'Oz', 'O2', 'FPz'
]


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
    """
    Compute row-wise statistics (mean, std, CV) for each measurement probe.
    
    CV (Coefficient of Variation) = std/|mean| measures how much each probe
    discriminates between different cases. Higher CV = better discriminability.
    """
    n_probes, n_cases = F.shape
    
    # Default to 10-20 system names if not provided and shape matches
    if probe_names is None:
        if n_probes == len(PROBE_NAMES_10_20):
            probe_names = PROBE_NAMES_10_20.copy()
        else:
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
    """Compute rank, condition number, singular values, and pseudoinverse properties."""
    _, s, _ = np.linalg.svd(F, full_matrices=False)
    rank = int(np.linalg.matrix_rank(F))
    cond = float(np.linalg.cond(F))
    
    # Compute Moore-Penrose pseudoinverse (non-regularized)
    F_pinv = np.linalg.pinv(F)
    cond_pinv = float(np.linalg.cond(F_pinv))
    
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
        'pseudoinverse': {
            'shape': list(F_pinv.shape),
            'condition_number': cond_pinv,
            'frobenius_norm': float(np.linalg.norm(F_pinv, 'fro')),
        }
    }


def compute_regularized_inverse(F: np.ndarray, lam: float, L: np.ndarray = None) -> np.ndarray:
    """
    Compute regularized pseudoinverse: B = (F^T F + λ² L^T L)^{-1} F^T
    
    WHAT B IS:
    B is the regularized pseudoinverse matrix. Given measurements y = F @ x_true,
    the reconstructed solution is: x_reconstructed = B @ y
    
    This solves the Tikhonov-regularized least squares problem:
        x_λ = argmin ||Fx - y||² + λ²||Lx||²
    
    If L is None, uses L = I (standard Tikhonov regularization).
    
    WHY solve() INSTEAD OF inv():
    We use np.linalg.solve(A, F.T) instead of np.linalg.inv(A) @ F.T.
    Both give the same mathematical result, but solve() uses LU decomposition
    which is numerically more stable.
    """
    n_cases = F.shape[1]
    if L is None:
        L = np.eye(n_cases)
    
    FTF = F.T @ F
    LTL = L.T @ L
    A = FTF + (lam ** 2) * LTL
    
    return np.linalg.solve(A, F.T)


def compute_regularization_cherab(F: np.ndarray, y: np.ndarray, L: np.ndarray = None):
    """
    Compute GCV and L-curve optimal lambdas using cherab-inversion library.
    
    This uses scipy.optimize.basinhopping to find the global minimum/maximum
    of the respective criteria.
    
    Args:
        F: Forward matrix (m x n)
        y: Measurement vector (m,)
        L: Regularization matrix (n x n). If None, uses identity.
           The Hessian H = L^T @ L is computed internally.
    
    Returns:
        dict with 'gcv_lambda', 'lcurve_lambda', 'gcv_solver', 'lcurve_solver'
    """
    n = F.shape[1]
    
    # Compute Hessian H = L^T @ L from regularization matrix L
    if L is None:
        H = np.eye(n)
    else:
        H = L.T @ L
    
    # Compute generalized SVD (shared between GCV and L-curve)
    s, u, basis = compute_svd(F, H)
    
    # Create GCV solver and solve
    gcv_solver = GCV(s, u, basis, data=y)
    gcv_solution, gcv_status = gcv_solver.solve()
    
    # Create L-curve solver and solve
    lcurve_solver = Lcurve(s, u, basis, data=y)
    lcurve_solution, lcurve_status = lcurve_solver.solve()

    return {
        'gcv_lambda': gcv_solver.lambda_opt,
        'lcurve_lambda': lcurve_solver.lambda_opt,
        'gcv_solver': gcv_solver,
        'lcurve_solver': lcurve_solver,
        'gcv_solution': gcv_solution,
        'lcurve_solution': lcurve_solution,
    }


def reconstruction_error(x_true: np.ndarray, x_reconstructed: np.ndarray) -> dict:  
    """
    Compute reconstruction error metrics.
    
    The 'correlation' metric is Pearson correlation coefficient between x_true
    and x_reconstructed. A value of 1 indicates perfect linear relationship
    (vectors point in same direction), 0 means uncorrelated, -1 means anticorrelated.
    This measures shape similarity independent of scale.
    """
    diff = x_true - x_reconstructed
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    
    x_true_norm = np.linalg.norm(x_true)
    rel_error = float(np.linalg.norm(diff) / x_true_norm) if x_true_norm > 0 else float('inf')
    
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
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    # Use diverging purple-to-green colormap (PiYG)
    im = ax.imshow(mat, cmap='PiYG', vmin=-1, vmax=1, aspect='equal')
    
    ax.set_xlabel('Case', fontsize=12)
    ax.set_ylabel('Case', fontsize=12)
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
    
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_singular_values(singular_values: list, output_path: Path):
    """Plot singular value spectrum."""
    s = np.array(singular_values)
    n = len(s)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Linear scale
    ax1 = axes[0]
    ax1.bar(range(1, n + 1), s, color='steelblue', edgecolor='navy', alpha=0.8)
    ax1.set_xlabel(f'σ₁ (largest) → σ{n} (smallest)', fontsize=12)
    ax1.set_ylabel('Singular Value', fontsize=12)
    ax1.set_title('Singular Values (Linear)', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(1, n + 1))
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Log scale
    ax2 = axes[1]
    ax2.bar(range(1, n + 1), s, color='steelblue', edgecolor='navy', alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel(f'σ₁ (largest) → σ{n} (smallest)', fontsize=12)
    ax2.set_ylabel('Singular Value', fontsize=12)
    ax2.set_title('Singular Values (Log)', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(1, n + 1))
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_reconstruction_errors(x_lambdas: np.ndarray, y_errors: list, 
                                gcv_lambdas: list, lcurve_lambdas: list,
                                empirical_lambda: float, output_path: Path,
                                mark_valid_range: bool = True, lambda_min_valid: float = None, 
                                lambda_max_valid: float = None):
    """
    Plot absolute reconstruction error vs lambda - one subplot per test case.
    
    Args:
        x_lambdas: Array of lambda values used in sweep
        y_errors: List of error lists (one per test case), each containing dicts with 'rmse'
        gcv_lambdas: List of GCV lambdas (one per test case)
        lcurve_lambdas: List of L-curve lambdas (one per test case)
        empirical_lambda: Lambda that minimizes avg reconstruction error (requires x_true)
        output_path: Path to save figure
        mark_valid_range: If True, shade the valid lambda range (σ_min² to σ_max²)
        lambda_min_valid: Lower bound of valid lambda range
        lambda_max_valid: Upper bound of valid lambda range
    """
    n_cases = len(y_errors)
    
    # Determine grid layout
    n_cols = min(3, n_cases)
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), 
                             constrained_layout=True, squeeze=False)
    axes = axes.flatten()
    
    for i, err_data in enumerate(y_errors):
        ax = axes[i]
        
        # Use RMSE (absolute error) instead of relative error
        abs_errors = [e['rmse'] for e in err_data]
        ax.semilogx(x_lambdas, abs_errors, 'b-', linewidth=1.5, alpha=0.8)
        
        # Shade valid lambda range
        if mark_valid_range and lambda_min_valid is not None and lambda_max_valid is not None:
            ax.axvspan(lambda_min_valid, lambda_max_valid, alpha=0.1, color='gray')
        
        # Mark this test case's GCV lambda
        ax.axvline(gcv_lambdas[i], color='g', linestyle='-', linewidth=1.5, 
                   label=f'GCV: {gcv_lambdas[i]:.1e}')
        
        # Mark this test case's L-curve lambda
        ax.axvline(lcurve_lambdas[i], color='purple', linestyle='-.', linewidth=1.5, 
                   label=f'L-curve: {lcurve_lambdas[i]:.1e}')
        
        # Mark empirical optimal (same for all - based on average)
        ax.axvline(empirical_lambda, color='orange', linestyle='--', linewidth=1.5, 
                   label=f'Empirical: {empirical_lambda:.1e}')
        
        ax.set_xlabel('λ', fontsize=10)
        ax.set_ylabel('RMSE', fontsize=10)
        ax.set_title(f'Test Case {i+1}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(n_cases, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Reconstruction Error (RMSE) vs Regularization Parameter', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=300)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze forward transfer matrix for regularized inversion.')
    parser.add_argument('--matrix', type=Path, required=True, help='Path to F matrix (.npy)')
    parser.add_argument('--case-names', nargs='+', default=None, 
                        help='Case names (e.g., E1 E2 E3 or E1_R E1_T E2_R). If not provided, uses case_0, case_1, etc.')
    parser.add_argument('--L-matrix', type=Path, default=None, 
                        help='Path to regularization matrix L (.npy). Default: identity matrix')
    parser.add_argument('--out', type=Path, default=Path('analysis_results'), help='Output directory')
    
    args = parser.parse_args()
    
    if not args.matrix.exists():
        print(f"ERROR: Matrix file not found: {args.matrix}")
        return
    
    # Load matrix
    print(f"Loading F matrix from {args.matrix}...")
    F = np.load(args.matrix)
    n_probes, n_cases = F.shape
    print(f"Matrix shape: {n_probes} probes × {n_cases} cases")
    
    # Load regularization matrix if provided
    L = None
    if args.L_matrix is not None:
        if not args.L_matrix.exists():
            print(f"ERROR: L matrix file not found: {args.L_matrix}")
            return
        L = np.load(args.L_matrix)
        print(f"Using custom regularization matrix L from {args.L_matrix}")
        print(f"L shape: {L.shape}")
    else:
        print("Using identity matrix for regularization (standard Tikhonov)")
    
    # Extract case names
    if args.case_names:
        case_names = args.case_names
        if len(case_names) != n_cases:
            print(f"WARNING: Number of case names ({len(case_names)}) doesn't match matrix columns ({n_cases})")
            # Pad or truncate
            if len(case_names) < n_cases:
                case_names = case_names + [f"case_{i}" for i in range(len(case_names), n_cases)]
            else:
                case_names = case_names[:n_cases]
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
    print(f"   Condition number (F): {props['condition_number']:.2e}")
    print(f"   SV ratio (max/min): {props['sv_ratio']:.2e}")
    print(f"   Pseudoinverse (F⁺):")
    print(f"      Shape: {props['pseudoinverse']['shape']}")
    print(f"      Condition number (κ): {props['pseudoinverse']['condition_number']:.2e}")
    print(f"      Frobenius norm: {props['pseudoinverse']['frobenius_norm']:.2e}")
    
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
    # 4. Lambda analysis (GCV and L-curve based regularization parameter selection)
    # ==========================================================================
    print("\n4. Analyzing regularization parameter λ using cherab-inversion...")
        
    # Use columns of F as test cases
    # Each column F[:, j] = F @ e_j where e_j is unit vector (electrode j active)
    # So x_true = e_j (known!) and y = F[:, j] (no noise)
    print(f"   Using {n_cases} columns of F as test cases (y = F[:, j], x_true = e_j)...")
    test_cases = []
    all_errors = []
    reg_results = []  # Store cherab regularization results
    
    # Store test data: (x_true, y) pairs where x_true = unit vector, y = F column
    test_data = []  # (x_true, y) pairs for error computation
    
    for j in range(n_cases):
        # x_true is unit vector with 1 at position j
        x_true = np.zeros(n_cases)
        x_true[j] = 1.0
        # y is the j-th column of F (no noise)
        y = F[:, j].copy()
        
        test_data.append((x_true, y))
    
    # Compute GCV and L-curve using cherab-inversion (uses optimization, not grid search)
    print(f"   Computing GCV and L-curve for {n_cases} cases (using cherab-inversion)...")
    for j, (x_true, y) in enumerate(test_data):
        reg_result = compute_regularization_cherab(F, y, L)
        reg_results.append(reg_result)
        # Store only the lambda pair
        test_cases.append({
            'case_name': case_names[j],
            'gcv_lambda': reg_result['gcv_lambda'],
            'lcurve_lambda': reg_result['lcurve_lambda'],
        })
        print(f"      {case_names[j]}: GCV λ = {reg_result['gcv_lambda']:.2e}, L-curve λ = {reg_result['lcurve_lambda']:.2e}")
    
    # Average lambdas across test cases (geometric mean)
    gcv_lambdas = [r['gcv_lambda'] for r in reg_results]
    lcurve_lambdas = [r['lcurve_lambda'] for r in reg_results]

    # Compute reconstruction errors for all test cases
    print("   Computing reconstruction errors...")

    # Define lambda grid: from σ_min² to σ_max² (cherab's valid range) plus external intervals
    s_values = np.array(props['singular_values'])
    s_min = s_values[-1]  # Smallest singular value
    s_max = s_values[0]   # Largest singular value
    lambda_min_valid = s_min ** 2
    lambda_max_valid = s_max ** 2
    
    # Create lambda sweep: include valid range plus external intervals for comparison
    n_lambdas = 300
    lambdas_internal = np.logspace(np.log10(lambda_min_valid), np.log10(lambda_max_valid), n_lambdas // 2)
    lambdas_external_low = np.logspace(np.log10(lambda_min_valid) - 2, np.log10(lambda_min_valid) - 0.1, n_lambdas // 4)
    lambdas_external_high = np.logspace(np.log10(lambda_max_valid) + 0.1, np.log10(lambda_max_valid) + 2, n_lambdas // 4)
    lambdas = np.concatenate([lambdas_external_low, lambdas_internal, lambdas_external_high])
    lambdas = np.sort(np.unique(lambdas))
    
    print(f"   Lambda range: {lambdas[0]:.2e} to {lambdas[-1]:.2e}")
    print(f"   Valid range (σ_min² to σ_max²): {lambda_min_valid:.2e} to {lambda_max_valid:.2e}")
    
    for i, (x_true, y) in enumerate(test_data):
        errors_for_case = []
        
        for lam in lambdas:
            B = compute_regularized_inverse(F, lam, L)
            x_lambda = B @ y
            err = reconstruction_error(x_true, x_lambda)
            errors_for_case.append(err)
        
        all_errors.append(errors_for_case)
    
    # Find lambda that minimizes average reconstruction error (for validation - requires knowing x_true)
    avg_rel_errors = np.mean([[e['relative_error'] for e in errs] for errs in all_errors], axis=0)
    min_error_idx = np.argmin(avg_rel_errors)
    empirical_lambda = lambdas[min_error_idx]
    empirical_error_value = avg_rel_errors[min_error_idx]
    print(f"   Empirical λ (validation only, requires x_true): {empirical_lambda:.2e}")
    print(f"   Relative error at empirical λ: {empirical_error_value:.4f}")
    
    
    # Use cherab-inversion's built-in plotting for first test case
    print("   Generating plots using cherab-inversion...")
    
    lcurve_solver = reg_results[0]['lcurve_solver']
    gcv_solver = reg_results[0]['gcv_solver']
    
    # Plot L-curve with annotated scatter points (library's built-in feature)
    lcurve_path = args.out / "lcurve.svg"
    fig_lcurve, ax_lcurve = lcurve_solver.plot_L_curve(scatter_plot=5, scatter_annotate=True)
    fig_lcurve.tight_layout()
    fig_lcurve.savefig(lcurve_path, dpi=300)
    plt.close(fig_lcurve)
    print(f"   Saved: {lcurve_path}")
    
    # Plot L-curve curvature
    curvature_path = args.out / "lcurve_curvature.svg"
    fig_curv, ax_curv = lcurve_solver.plot_curvature()
    fig_curv.tight_layout()
    fig_curv.savefig(curvature_path, dpi=300)
    plt.close(fig_curv)
    print(f"   Saved: {curvature_path}")
    
    # Plot GCV curve
    gcv_path = args.out / "gcv_curve.svg"
    fig_gcv, ax_gcv = gcv_solver.plot_gcv()
    fig_gcv.tight_layout()
    fig_gcv.savefig(gcv_path, dpi=300)
    plt.close(fig_gcv)
    print(f"   Saved: {gcv_path}")
    
    # Plot reconstruction errors (one subplot per test case, with per-case lambdas)
    recon_path = args.out / "reconstruction_error.svg"
    plot_reconstruction_errors(lambdas, all_errors, gcv_lambdas, lcurve_lambdas, 
                              empirical_lambda, recon_path, mark_valid_range=True,
                              lambda_min_valid=lambda_min_valid, lambda_max_valid=lambda_max_valid)
    print(f"   Saved: {recon_path}")
    
    # ==========================================================================
    # 5. Save full report
    # ==========================================================================
    print("\n5. Saving analysis report...")
    
    report = {
        'matrix_file': str(args.matrix),
        'L_matrix_file': str(args.L_matrix) if args.L_matrix else None,
        'case_names': case_names,
        'matrix_properties': props,
        'correlation_statistics': corr_stats,
        'probe_statistics': probe_stats,
        'regularization_analysis': {
            'n_cases': n_cases,
            # GCV method (cherab-inversion)
            'gcv_lambdas_per_test': gcv_lambdas,
            # L-curve method (cherab-inversion)
            'lcurve_lambdas_per_test': lcurve_lambdas,
            # Empirical lambda (for validation only - requires knowing x_true)
            'empirical_lambda': float(empirical_lambda),
        },
        'test_cases': test_cases,
    }
        
    report_path = args.out / "analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   Saved: {report_path}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Matrix: {n_probes} probes × {n_cases} cases")
    print(f"Rank: {props['rank']}")
    print(f"Condition number (F): {props['condition_number']:.2e}")
    print(f"Pseudoinverse condition number (F⁺): {props['pseudoinverse']['condition_number']:.2e}")
    print(f"Case correlation (RMS): {corr_stats['rms']:.3f}")
    print(f"Probe discrimination (mean CV): {probe_stats['summary']['mean_cv']:.2f}")
    print(f"")
    print(f"Lambda Selection ({n_cases} cases = F columns, cherab-inversion):")
    print(f"  GCV method:")
    print(f"    Per-case: {', '.join([f'{l:.2e}' for l in gcv_lambdas])}")
    print(f"  L-curve method:")
    print(f"    Per-case: {', '.join([f'{l:.2e}' for l in lcurve_lambdas])}")
    print(f"  Empirical (validation): {empirical_lambda:.2e}  (error: {empirical_error_value*100:.2f}%)")
    print(f"")
    print("="*60)


if __name__ == '__main__':
    main()