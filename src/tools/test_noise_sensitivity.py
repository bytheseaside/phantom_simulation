#!/usr/bin/env python3
"""
matrix_tests_noise_only.py
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def rel_error(x_true, x_hat, eps=1e-12):
    """L2 relative error: ||x - x_hat||_2 / ||x||_2"""
    num = np.linalg.norm(x_true - x_hat, axis=0)
    den = np.linalg.norm(x_true, axis=0)
    return num / (den + eps)

def generate_cases(Ne, n_dense, n_sparse, seed=0, signed_sparse=False):
    rng = np.random.default_rng(seed)
    
    # Dense: Standard Normal
    x_dense = rng.normal(0.0, 1.0, size=(Ne, n_dense))

    # Sparse: 1-2 non-zero entries
    x_sparse = np.zeros((Ne, n_sparse))
    for j in range(n_sparse):
        k = rng.choice([1, 2])
        idx = rng.choice(Ne, size=k, replace=False)
        vals = np.ones(k)
        if signed_sparse:
            vals *= rng.choice([-1.0, 1.0], size=k)
        x_sparse[idx, j] = vals

    return x_dense, x_sparse

def summarize(err: np.ndarray):
    return {
        "N": err.size,
        "mean": np.mean(err),
        "median": np.median(err),
        "std": np.std(err, ddof=1) if err.size > 1 else 0.0,
        "p95": np.percentile(err, 95),
        "max": np.max(err),
    }

def solve_noise_only(A, B, X, mu, sigma, noise_seed):
    """Reconstruct using pinv(A)."""
    y = A @ X
    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(mu, sigma, size=y.shape)
    y_noisy = y + noise

    x_hat = B @ y_noisy
    err = rel_error(X, x_hat)
    
    # Calculate Noise-to-Signal ratio for context
    nsr = np.mean(np.linalg.norm(noise, axis=0)) / np.mean(np.linalg.norm(y, axis=0))
    return err, nsr

def _fmt(x: float) -> str:
    return f"{x: .3e}" if (abs(x) < 1e-3 or abs(x) >= 1e3) else f"{x: .6f}"

def print_summary_block(A_shape, mu, sigma, results):
    Nm, Ne = A_shape
    print("\n" + "=" * 85)
    print(f" MATRIX RECONSTRUCTION REPORT | A: {Nm}x{Ne}")
    print(f" Noise Config: μ={_fmt(mu)}, σ={_fmt(sigma)}")
    print("-" * 85)

    headers = ["Family", "N", "Mean Err", "Median", "StdDev", "95th %", "Max"]
    col_widths = [10, 6, 12, 12, 12, 12, 12]
    
    header_line = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * 85)

    for name, st in results.items():
        row = [
            name, 
            str(st["N"]), 
            _fmt(st["mean"]), 
            _fmt(st["median"]), 
            _fmt(st["std"]), 
            _fmt(st["p95"]), 
            _fmt(st["max"])
        ]
        print("".join(f"{str(item):<{w}}" for item, w in zip(row, col_widths)))
    print("=" * 85 + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A_path", type=Path, help="Path to A matrix (.npy)")
    ap.add_argument("--n-dense", type=int, default=250)
    ap.add_argument("--n-sparse", type=int, default=250)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=1e-2)
    ap.add_argument("--logy", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("matrix_test_noise_only.png"))
    args = ap.parse_args()

    # Load and Precompute pinv
    A = np.load(args.A_path)
    B = np.linalg.pinv(A)
    Nm, Ne = A.shape

    # Generate signal families
    x_dense, x_sparse = generate_cases(Ne, args.n_dense, args.n_sparse)

    # Solve
    err_dense, nsr_d = solve_noise_only(A, B, x_dense, args.mu, args.sigma, 101)
    err_sparse, nsr_s = solve_noise_only(A, B, x_sparse, args.mu, args.sigma, 202)

    stats = {
        "Dense": summarize(err_dense),
        "Sparse": summarize(err_sparse)
    }

    # Print Report
    print_summary_block(A.shape, args.mu, args.sigma, stats)

    # ---- Prettier Plotting ----
    plt.rcParams.update({'font.size': 10, 'axes.grid': True})
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # Plot lines with slight transparency and markers for clarity
    ax.plot(err_dense, label=f"Dense (Avg NSR: {_fmt(nsr_d)})", color='#1f77b4', alpha=0.7, lw=1.5)
    ax.plot(err_sparse, label=f"Sparse (Avg NSR: {_fmt(nsr_s)})", color='#ff7f0e', alpha=0.7, lw=1.5)

    # Add a horizontal line for the mean of each
    ax.axhline(stats["Dense"]["mean"], color='#1f77b4', ls='--', alpha=0.5)
    ax.axhline(stats["Sparse"]["mean"], color='#ff7f0e', ls='--', alpha=0.5)

    ax.set_xlabel("Test Case Index", fontweight='bold')
    ax.set_ylabel(r"Relative Error $\frac{||x - \hat{x}||_2}{||x||_2}$", fontweight='bold')
    ax.set_title(f"Reconstruction Error\nMatrix Shape: {Nm}x{Ne} | $\sigma$={args.sigma}", pad=15)
    
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    
    if args.logy:
        ax.set_yscale("log")

    # Legend placement BELOW
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, 
               bbox_to_anchor=(0.5, 0.02), fontsize=9)

    # Adjust layout to make room for bottom legend
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    fig.savefig(args.out)
    print(f"Saved figure to: {args.out}")

if __name__ == "__main__":
    main()