#!/usr/bin/env python3
"""
Validate pseudoinverse reconstruction quality using test cases.

This script:
1. Loads a forward matrix F and computes its pseudoinverse F⁺
2. For each test case (NPY file with measurements y):
   - Reconstructs x using: x_reconstructed = F⁺ @ y
   - Compares x_reconstructed to x_true (from manifest)
   - Computes error metrics

Usage:
  python validate_pseudoinverse_reconstruction.py \
    --F run_phantom/F.npy \
    --manifest run_phantom/manifest.json \
    --test-dir run_phantom/cases \
    --out validation_results/

Outputs:
  - validation_report.json: Full reconstruction error analysis
  - reconstruction_rmse.svg: RMSE per test case
  - reconstruction_relative_error.svg: Relative error per test case
  - summary_comparison.svg: Combined true vs reconstructed comparison
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

# Component keys (0-based, lexicographic order)
DOF_KEYS = [f'e{i}_T' for i in range(1, 10)]

# Global for deleted indices
deleted_indices = []


def extract_x_true(case_data: dict) -> np.ndarray:
    """Extract ground truth x vector from manifest case."""
    dirichlet = case_data['dirichlet']
    values_dict = {item['name']: item['value'] for item in dirichlet}
    
    keys = DOF_KEYS.copy()
    if deleted_indices:
        keys = [k for i, k in enumerate(keys) if i not in deleted_indices]
    
    values = [values_dict[key] for key in keys]
    return np.array(values)


def compute_errors(x_true: np.ndarray, x_recon: np.ndarray) -> Dict:
    """Compute error metrics between true and reconstructed x."""
    diff = x_true - x_recon
    
    rmse = float(np.sqrt(np.mean(diff**2)))
    x_norm = np.linalg.norm(x_true)
    rel_error = float(np.linalg.norm(diff) / x_norm) if x_norm > 0 else float('inf')
    
    # Correlation (if more than 1 component)
    if len(x_true) > 1:
        corr = float(np.corrcoef(x_true, x_recon)[0, 1])
    else:
        corr = 1.0 if np.isclose(x_true[0], x_recon[0]) else 0.0
    
    return {
        'rmse': rmse,
        'relative_error': rel_error,
        'correlation': corr if not np.isnan(corr) else 0.0,
    }


def add_noise(y: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add Gaussian noise to measurement vector.
    
    Args:
        y: Clean measurement vector
        noise_level: Noise as percentage of signal std (e.g., 5 for 5%)
        rng: NumPy random generator
    
    Returns:
        Noisy measurement vector
    """
    if noise_level <= 0:
        return y
    sigma = (noise_level / 100.0) * np.std(y)
    noise = rng.normal(0, sigma, size=y.shape)
    return y + noise


def validate_reconstruction(F: np.ndarray, test_files: List[Path], manifest: Dict,
                            noise_level: float = 0.0, seed: int = None, lambda_reg: float = None) -> List[Dict]:
    """Validate reconstruction for all test cases."""
    global deleted_indices

    n_cases = F.shape[1]
    rng = np.random.default_rng(seed)

    # Choose reconstruction method
    if lambda_reg is not None:
        # Regularized pseudoinverse with identity L
        FTF = F.T @ F
        LTL = np.eye(n_cases)
        A = FTF + (lambda_reg ** 2) * LTL
        F_pinv = np.linalg.solve(A, F.T)
        cond_F = np.linalg.cond(A)
        print(f"Using regularized pseudoinverse with lambda={lambda_reg}")
    else:
        # Standard pseudoinverse
        F_pinv = np.linalg.pinv(F)
        cond_F = np.linalg.cond(F)
        print(f"Using standard pseudoinverse")

    print(f"F shape: {F.shape}")
    print(f"F⁺ shape: {F_pinv.shape}")
    print(f"κ(F): {cond_F:.2e}")
    if noise_level > 0:
        print(f"Noise level: {noise_level}%")

    results = []
    manifest_cases = {c['name']: c for c in manifest['cases']}

    for test_file in test_files:
        name = test_file.stem

        if name not in manifest_cases:
            print(f"  SKIP: {name} (not in manifest)")
            continue

        y_clean = np.load(test_file)
        y = add_noise(y_clean, noise_level, rng)  # Add noise to measurements
        x_recon = F_pinv @ y
        x_true = extract_x_true(manifest_cases[name])

        errors = compute_errors(x_true, x_recon)

        results.append({
            'name': name,
            'x_true': x_true.tolist(),
            'x_recon': x_recon.tolist(),
            'errors': errors,
        })

        print(f"  {name}: RMSE={errors['rmse']:.4f}, RelErr={errors['relative_error']*100:.1f}%")

    return results, cond_F


def plot_errors(results: List[Dict], out_dir: Path):
    """Plot error bar charts."""
    names = [r['name'] for r in results]
    rmse = [r['errors']['rmse'] for r in results]
    rel_err = [r['errors']['relative_error'] * 100 for r in results]
    
    # RMSE plot
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.bar(names, rmse, color='#5b9bd5')
    ax.set_xlabel('Test Case')
    ax.set_ylabel('RMSE')
    ax.set_title('Reconstruction Error (RMSE)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / 'reconstruction_rmse.svg')
    plt.close()
    
    # Relative error plot
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.bar(names, rel_err, color='#ed7d31')
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title('Reconstruction Relative Error')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / 'reconstruction_relative_error.svg')
    plt.close()


def plot_comparison(results: List[Dict], out_dir: Path):
    """Plot true vs reconstructed comparison for all cases."""
    n_cases = len(results)
    if n_cases == 0:
        return
    
    n_components = len(results[0]['x_true'])
    
    # Collect all true and reconstructed values
    all_true = np.array([r['x_true'] for r in results])
    all_recon = np.array([r['x_recon'] for r in results])
    
    # Global scatter: true vs reconstructed
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.scatter(all_true.flatten(), all_recon.flatten(), alpha=0.6, s=30, color='#5b9bd5')
    
    # Perfect line
    lims = [min(all_true.min(), all_recon.min()), max(all_true.max(), all_recon.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
    
    ax.set_xlabel('True Value')
    ax.set_ylabel('Reconstructed Value')
    ax.set_title('True vs Reconstructed (All Components)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(out_dir / 'true_vs_reconstructed_scatter.svg')
    plt.close()
    
    # Heatmap comparison (cases x components)
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=150)
    
    names = [r['name'] for r in results]
    
    # Get component labels
    keys = DOF_KEYS.copy()
    if deleted_indices:
        keys = [k for i, k in enumerate(keys) if i not in deleted_indices]
    
    vmin = min(all_true.min(), all_recon.min())
    vmax = max(all_true.max(), all_recon.max())
    
    im1 = axes[0].imshow(all_true, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('True Values')
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('Test Case')
    axes[0].set_xticks(range(n_components))
    axes[0].set_xticklabels(keys, rotation=45, ha='right')
    axes[0].set_yticks(range(n_cases))
    axes[0].set_yticklabels(names)
    
    im2 = axes[1].imshow(all_recon, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('Reconstructed Values')
    axes[1].set_xlabel('Component')
    axes[1].set_ylabel('Test Case')
    axes[1].set_xticks(range(n_components))
    axes[1].set_xticklabels(keys, rotation=45, ha='right')
    axes[1].set_yticks(range(n_cases))
    axes[1].set_yticklabels(names)
    
    fig.colorbar(im2, ax=axes, shrink=0.6, label='Value')
    plt.tight_layout()
    plt.savefig(out_dir / 'comparison_heatmap.svg')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Validate pseudoinverse reconstruction.')
    parser.add_argument('--F', type=Path, required=True, help='Forward matrix F (.npy)')
    parser.add_argument('--manifest', type=Path, required=True, help='manifest.json')
    parser.add_argument('--test-dir', type=Path, default=None, help='Directory with test NPY files')
    parser.add_argument('--test-files', nargs='+', type=Path, default=None, help='Test NPY files')
    parser.add_argument('--out', type=Path, default=Path('validation_results'), help='Output directory')
    parser.add_argument('--deleted', nargs='*', type=int, default=None, help='0-based deleted column indices')
    parser.add_argument('--noise', type=float, default=0.0,
                        help='Noise level as %% of signal std (e.g., 5 for 5%% noise)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible noise')
    parser.add_argument('--lambda', type=float, default=None,
                        help='Regularization parameter lambda. If not provided, uses pseudoinverse.')
    
    args = parser.parse_args()
    
    global deleted_indices
    deleted_indices = args.deleted if args.deleted else []
    
    # Get test files
    if args.test_files:
        test_files = args.test_files
    elif args.test_dir:
        test_files = sorted(args.test_dir.glob('*.npy'))
    else:
        print("ERROR: Provide --test-dir or --test-files")
        return
    
    print(f"Found {len(test_files)} test files")
    if deleted_indices:
        print(f"Deleted columns: {deleted_indices}")
    
    # Load data
    F = np.load(args.F)
    with open(args.manifest) as f:
        manifest = json.load(f)
    
    print(f"\nManifest has {len(manifest['cases'])} cases")
    
    # Run validation
    print("\n" + "="*50)
    print("VALIDATING RECONSTRUCTION")
    print("="*50)
    
    # 'lambda' is a reserved keyword, so use getattr
    lambda_value = getattr(args, 'lambda')
    results, cond_F = validate_reconstruction(F, test_files, manifest,
                                               noise_level=args.noise, seed=args.seed,
                                               lambda_reg=lambda_value)
    
    if not results:
        print("\nERROR: No matching cases found!")
        return
    
    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Compute summary
    all_rmse = [r['errors']['rmse'] for r in results]
    all_rel = [r['errors']['relative_error'] for r in results]
    
    summary = {
        'n_cases': len(results),
        'condition_number': float(cond_F),
        'noise_level_percent': args.noise,
        'deleted_columns': deleted_indices,
        'rmse_mean': float(np.mean(all_rmse)),
        'rmse_std': float(np.std(all_rmse)),
        'rel_error_mean': float(np.mean(all_rel)),
        'rel_error_std': float(np.std(all_rel)),
    }
    
    # Save report
    report = {'summary': summary, 'results': results}
    with open(args.out / 'validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Plot
    plot_errors(results, args.out)
    plot_comparison(results, args.out)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Cases validated: {len(results)}")
    print(f"κ(F): {cond_F:.2e}")
    if args.noise > 0:
        print(f"Noise: {args.noise}%")
    print(f"RMSE: {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
    print(f"Relative Error: {summary['rel_error_mean']*100:.2f}% ± {summary['rel_error_std']*100:.2f}%")
    print(f"\nOutputs saved to: {args.out}")


if __name__ == '__main__':
    main()
