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
  # Using test cases in a directory
  python validate_pseudoinverse_reconstruction.py \
    --F run_phantom/9dof/F.npy \
    --test-dir run_phantom/9dof/cases \
    --manifest run_phantom/9dof/manifest.json \
    --dof 9 \
    --out validation_results/

  # Or specify individual test files
  python validate_pseudoinverse_reconstruction.py \
    --F run_phantom/9dof/F.npy \
    --test-files run_phantom/9dof/cases/9dof-e1-tr.npy run_phantom/9dof/cases/9dof-e2-tr.npy \
    --manifest run_phantom/9dof/manifest.json \
    --dof 9

Outputs:
  - validation_report.json: Full reconstruction error analysis
  - reconstruction_errors.svg: Bar plot of errors per test case
  - correlation_plot.svg: Scatter plot of x_true vs x_reconstructed
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple

# Component keys for 9dof mode (0-based, lexicographic)
DOF9_KEYS = [f'e{i}_T' for i in range(1, 10)]


def load_manifest(manifest_path: Path, dof: int) -> Dict:
    """Load and parse manifest JSON."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


def extract_x_true_from_manifest(case_data: dict, dof: int) -> np.ndarray:
    """
    Extract ground truth x vector from manifest case.
    
    [e1_T, e2_T, ..., e9_T]
        9 electrodes, only T component varies (R = -T, S = 0)
    """
    dirichlet = case_data['dirichlet']
    values_dict = {item['name']: item['value'] for item in dirichlet}
    keys = DOF9_KEYS.copy()
    global deleted_indices
    if deleted_indices:
        keys = [k for i, k in enumerate(keys) if i not in deleted_indices]
    values = []
    for key in keys:
        if key in values_dict:
            values.append(values_dict[key])
        else:
            raise ValueError(f"Missing key {key} in manifest")
    x_true = np.array(values)
    if len(x_true) != len(keys):
        raise ValueError(f"Expected {len(keys)} components but got {len(x_true)} from manifest")
    return x_true


def compute_reconstruction_errors(x_true: np.ndarray, x_recon: np.ndarray) -> Dict:
    """Compute various error metrics between true and reconstructed x."""
    diff = x_true - x_recon
    
    # Absolute errors
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    max_abs_error = float(np.max(np.abs(diff)))
    
    # Relative errors
    x_true_norm = np.linalg.norm(x_true)
    rel_error = float(np.linalg.norm(diff) / x_true_norm) if x_true_norm > 0 else float('inf')
    rel_rmse = float(rmse / x_true_norm) if x_true_norm > 0 else float('inf')
    
    # Correlation
    if len(x_true) > 1:
        correlation = float(np.corrcoef(x_true.flatten(), x_recon.flatten())[0, 1])
    else:
        correlation = 1.0 if np.isclose(x_true[0], x_recon[0]) else 0.0
    
    # Cosine similarity (normalized dot product)
    cos_sim = float(np.dot(x_true, x_recon) / (np.linalg.norm(x_true) * np.linalg.norm(x_recon)))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_abs_error': max_abs_error,
        'relative_error': rel_error,
        'relative_rmse': rel_rmse,
        'correlation': correlation,
        'cosine_similarity': cos_sim,
    }


def validate_reconstruction(F: np.ndarray, test_files: List[Path], 
                           manifest: Dict, dof: int) -> List[Dict]:
    """
    Validate reconstruction for all test cases.
    
    Returns list of dicts with results for each test case.
    """
    # Remove deleted columns/components from F
    global deleted_indices
    if deleted_indices:
        F = np.delete(F, deleted_indices, axis=1)
    # Compute pseudoinverse
    F_pinv = np.linalg.pinv(F)
    print(f"Pseudoinverse F⁺ shape: {F_pinv.shape}")
    print(f"Condition number κ(F): {np.linalg.cond(F):.2e}")
    print(f"Condition number κ(F⁺): {np.linalg.cond(F_pinv):.2e}")
    results = []
    manifest_cases = manifest['cases']
    for test_file in test_files:
        print(f"\nProcessing: {test_file.name}")
        y = np.load(test_file)
        if deleted_indices:
            y = np.delete(y, deleted_indices)
        x_recon = F_pinv @ y
        file_stem = test_file.stem
        x_true = None
        matched_case = None
        for case in manifest_cases:
            if case.get('name') == file_stem:
                matched_case = case
                x_true = extract_x_true_from_manifest(case, dof)
                break
        if x_true is None:
            print(f"  WARNING: No matching manifest case found for {test_file.name}")
            continue
        print(f"  Matched to manifest case: {matched_case['name']}")
        errors = compute_reconstruction_errors(x_true, x_recon)
        results.append({
            'test_file': str(test_file),
            'test_name': test_file.stem,
            'manifest_case': matched_case['name'],
            'x_true': x_true.tolist(),
            'x_reconstructed': x_recon.tolist(),
            'y_measured': y.tolist(),
            'errors': errors,
        })
        print(f"  RMSE: {errors['rmse']:.4f}")
        print(f"  Relative error: {errors['relative_error']*100:.2f}%")
        print(f"  Correlation: {errors['correlation']:.4f}")
    return results


def plot_reconstruction_errors(results: List[Dict], output_path: Path):
    """Plot bar chart of reconstruction errors."""
    test_names = [r['test_name'] for r in results]
    rmse_values = [r['errors']['rmse'] for r in results]
    rel_errors = [r['errors']['relative_error'] * 100 for r in results]

    # RMSE plot
    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(test_names, rmse_values, color='#6fa8dc', alpha=0.85)
    plt.xlabel('Test Case', fontsize=13)
    plt.ylabel('RMSE', fontsize=13)
    plt.title('Reconstruction Error (RMSE)', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path.parent / 'reconstruction_rmse.svg', dpi=300)
    plt.close()

    # Relative error plot
    plt.figure(figsize=(10, 5), dpi=300)
    plt.bar(test_names, rel_errors, color='#f6b26b', alpha=0.85)
    plt.xlabel('Test Case', fontsize=13)
    plt.ylabel('Relative Error (%)', fontsize=13)
    plt.title('Reconstruction Error (Relative)', fontsize=15, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path.parent / 'reconstruction_relative_error.svg', dpi=300)
    plt.close()


def plot_correlation_scatter(results: List[Dict], output_path: Path):
    """Plot scatter plots of x_true vs x_reconstructed for each test case."""
    # For each test case, plot x_true and x_reconstructed as grouped bars
    for result in results:
        x_true = np.array(result['x_true'])
        x_recon = np.array(result['x_reconstructed'])
        n = len(x_true)
        indices = np.arange(1, n + 1)
        width = 0.35
        plt.figure(figsize=(max(8, n), 5), dpi=300)
        plt.bar(indices - width/2, x_true, width, label='True', color='#6fa8dc', alpha=0.85)
        plt.bar(indices + width/2, x_recon, width, label='Reconstructed', color='#f6b26b', alpha=0.85)
        plt.xlabel('Component (Electrode Index)', fontsize=13)
        plt.ylabel('Value', fontsize=13)
        plt.title(f"{result['test_name']}\nTrue vs Reconstructed Components", fontsize=15, fontweight='bold')
        plt.xticks(indices, [str(i) for i in indices], fontsize=10)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        out_path = output_path.parent / f"{result['test_name']}_components.svg"
        plt.savefig(out_path, dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Validate pseudoinverse reconstruction using test cases.'
    )
    parser.add_argument('--F', type=Path, required=True, 
                       help='Path to forward matrix F (.npy)')
    parser.add_argument('--manifest', type=Path, required=True,
                       help='Path to manifest.json with ground truth')
    parser.add_argument('--test-dir', type=Path, default=None,
                       help='Directory containing test case NPY files')
    parser.add_argument('--test-files', nargs='+', type=Path, default=None,
                       help='Specific test case NPY files')
    parser.add_argument('--out', type=Path, default=Path('validation_results'),
                       help='Output directory')
    parser.add_argument('--deleted', nargs='*', type=int, default=None,
                       help='0-based indices of deleted columns/components in F')
    args = parser.parse_args()
    # Set global deleted_indices
    global deleted_indices
    deleted_indices = args.deleted if args.deleted else []
    dof = 9
    # ...existing code...


if __name__ == '__main__':
    main()
