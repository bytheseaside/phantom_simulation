#!/usr/bin/env python3
"""
Column selection algorithms for F matrix reduction.

Selects a subset of columns (cases/antennas) to improve matrix conditioning
while maintaining sufficient degrees of freedom.

Algorithms:
- condition: Greedy removal minimizing condition number
- correlation: Greedy removal of most correlated column
- coherence: Greedy removal based on Gram matrix / mutual coherence
- qr: QR decomposition with column pivoting (selects most independent)

Usage:
  python select_columns.py --matrix F.npy --algorithm condition --min-cols 6 --out results/
  python select_columns.py --matrix F.npy --names e1 e2 e3 e4 e5 e6 e7 e8 e9 --algorithm correlation
  python select_columns.py --matrix F.npy --algorithm qr --min-cols 8

Outputs:
  - selection_results.json: indices, names, metrics history
  - F_reduced.npy: reduced matrix with selected columns
  - selection_history.svg: plot of metrics vs columns removed
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# Metric computation helpers
# =============================================================================

def compute_condition_number(F: np.ndarray) -> float:
    """Compute condition number of matrix."""
    return float(np.linalg.cond(F))


def compute_correlation_matrix(F: np.ndarray) -> np.ndarray:
    """Compute pairwise Pearson correlation between columns."""
    return np.corrcoef(F, rowvar=False)


def compute_coherence_matrix(F: np.ndarray) -> np.ndarray:
    """
    Compute normalized Gram matrix (coherence matrix).
    
    A = F^T F (Gram matrix)
    A_hat[i,j] = A[i,j] / sqrt(A[i,i] * A[j,j])
    
    This gives cosine similarity between columns.
    """
    A = F.T @ F
    diag = np.diag(A)
    # Avoid division by zero
    diag_sqrt = np.sqrt(np.maximum(diag, 1e-16))
    A_hat = A / np.outer(diag_sqrt, diag_sqrt)
    return A_hat


def compute_metrics(F: np.ndarray) -> dict:
    """Compute all relevant metrics for current matrix state."""
    corr = compute_correlation_matrix(F)
    coh = compute_coherence_matrix(F)
    n = F.shape[1]
    
    # Off-diagonal elements
    tril_idx = np.tril_indices(n, k=-1)
    corr_offdiag = np.abs(corr[tril_idx])
    coh_offdiag = np.abs(coh[tril_idx])
    
    # SVD
    _, singular_values, _ = np.linalg.svd(F, full_matrices=False)
    
    return {
        'n_cols': n,
        'condition_number': compute_condition_number(F),
        'rank': int(np.linalg.matrix_rank(F)),
        'singular_values': singular_values.tolist(),
        'max_singular_value': float(singular_values[0]) if len(singular_values) > 0 else 0.0,
        'min_singular_value': float(singular_values[-1]) if len(singular_values) > 0 else 0.0,
        'max_correlation': float(np.max(corr_offdiag)) if len(corr_offdiag) > 0 else 0.0,
        'mean_correlation': float(np.mean(corr_offdiag)) if len(corr_offdiag) > 0 else 0.0,
        'max_coherence': float(np.max(coh_offdiag)) if len(coh_offdiag) > 0 else 0.0,
        'mean_coherence': float(np.mean(coh_offdiag)) if len(coh_offdiag) > 0 else 0.0,
    }


# =============================================================================
# Selection algorithms
# =============================================================================

def greedy_condition(F: np.ndarray, min_cols: int, names: list) -> dict:
    """
    Greedy column removal to minimize condition number.
    
    At each step, remove the column whose removal results in the
    lowest condition number.
    """
    n_cols = F.shape[1]
    current_indices = list(range(n_cols))
    current_names = list(names)
    removed = []
    history = []
    
    # Initial state
    F_current = F.copy()
    history.append(compute_metrics(F_current))
    initial_cond = history[0]['condition_number']
    
    while len(current_indices) > min_cols:
        # Start with current condition number - only remove if it improves or maintains κ
        best_cond = initial_cond
        best_to_remove = None
        
        # Try removing each column
        for i in range(len(current_indices)):
            # Create F without column i
            mask = np.ones(len(current_indices), dtype=bool)
            mask[i] = False
            F_test = F_current[:, mask]
            
            cond = compute_condition_number(F_test)
            if cond < best_cond:
                best_cond = cond
                best_to_remove = i
        
        # Remove the best column
        removed_idx = current_indices[best_to_remove]
        removed_name = current_names[best_to_remove]
        removed.append({'index': removed_idx, 'name': removed_name})
        
        # Update current state
        mask = np.ones(len(current_indices), dtype=bool)
        mask[best_to_remove] = False
        F_current = F_current[:, mask]
        current_indices = [idx for j, idx in enumerate(current_indices) if mask[j]]
        current_names = [name for j, name in enumerate(current_names) if mask[j]]
        
        # Record metrics
        history.append(compute_metrics(F_current))
    
    # Build selected list
    selected = [{'index': idx, 'name': name} for idx, name in zip(current_indices, current_names)]
    
    return {
        'algorithm': 'greedy_condition',
        'selected': selected,
        'removed': removed,
        'F_reduced': F_current,
        'history': history,
        'final_metrics': history[-1],
    }


def greedy_correlation(F: np.ndarray, min_cols: int, names: list) -> dict:
    """
    Greedy column removal based on mean correlation.
    
    At each step, remove the column with highest mean absolute
    correlation with other columns.
    """
    n_cols = F.shape[1]
    current_indices = list(range(n_cols))
    current_names = list(names)
    removed = []
    history = []
    
    F_current = F.copy()
    history.append(compute_metrics(F_current))
    
    while len(current_indices) > min_cols:
        corr = compute_correlation_matrix(F_current)
        n = corr.shape[0]
        
        # Compute mean absolute correlation for each column (excluding diagonal)
        mean_corr = np.zeros(n)
        for i in range(n):
            others = [corr[i, j] for j in range(n) if j != i]
            mean_corr[i] = np.mean(np.abs(others))
        
        # Remove column with highest mean correlation
        worst_idx = int(np.argmax(mean_corr))
        
        removed_idx = current_indices[worst_idx]
        removed_name = current_names[worst_idx]
        removed.append({'index': removed_idx, 'name': removed_name})
        
        # Update
        mask = np.ones(n, dtype=bool)
        mask[worst_idx] = False
        F_current = F_current[:, mask]
        current_indices = [idx for j, idx in enumerate(current_indices) if mask[j]]
        current_names = [name for j, name in enumerate(current_names) if mask[j]]
        
        history.append(compute_metrics(F_current))
    
    # Build selected list
    selected = [{'index': idx, 'name': name} for idx, name in zip(current_indices, current_names)]
    
    return {
        'algorithm': 'greedy_correlation',
        'selected': selected,
        'removed': removed,
        'F_reduced': F_current,
        'history': history,
        'final_metrics': history[-1],
    }


def greedy_coherence(F: np.ndarray, min_cols: int, names: list) -> dict:
    """
    Greedy column removal based on Gram matrix coherence.
    
    At each step, find the pair with maximum coherence and remove
    the one with higher overall mean coherence.
    """
    n_cols = F.shape[1]
    current_indices = list(range(n_cols))
    current_names = list(names)
    removed = []
    history = []
    
    F_current = F.copy()
    history.append(compute_metrics(F_current))
    
    while len(current_indices) > min_cols:
        coh = compute_coherence_matrix(F_current)
        n = coh.shape[0]
        
        # Set diagonal to 0 to ignore self-coherence
        np.fill_diagonal(coh, 0)
        
        # Find pair with maximum coherence
        max_idx = np.unravel_index(np.argmax(np.abs(coh)), coh.shape)
        i, j = max_idx
        
        # Between i and j, remove the one with higher mean coherence to others
        mean_coh_i = np.mean(np.abs(coh[i, :]))
        mean_coh_j = np.mean(np.abs(coh[j, :]))
        worst_idx = i if mean_coh_i >= mean_coh_j else j
        
        removed_idx = current_indices[worst_idx]
        removed_name = current_names[worst_idx]
        removed.append({'index': removed_idx, 'name': removed_name})
        
        # Update
        mask = np.ones(n, dtype=bool)
        mask[worst_idx] = False
        F_current = F_current[:, mask]
        current_indices = [idx for k, idx in enumerate(current_indices) if mask[k]]
        current_names = [name for k, name in enumerate(current_names) if mask[k]]
        
        history.append(compute_metrics(F_current))
    
    # Build selected list
    selected = [{'index': idx, 'name': name} for idx, name in zip(current_indices, current_names)]
    
    return {
        'algorithm': 'greedy_coherence',
        'selected': selected,
        'removed': removed,
        'F_reduced': F_current,
        'history': history,
        'final_metrics': history[-1],
    }


def qr_pivoting(F: np.ndarray, min_cols: int, names: list) -> dict:
    """
    Column selection via QR decomposition with pivoting.
    
    The pivoting order ranks columns by linear independence.
    Select the first `min_cols` in pivot order.
    """
    n_cols = F.shape[1]
    
    # QR with column pivoting
    Q, R, P = linalg.qr(F, pivoting=True)
    
    # P contains the pivot order: P[i] = original column index at position i
    # Select first min_cols columns in pivot order
    selected_pivot_positions = list(range(min_cols))
    selected_indices = [int(P[i]) for i in selected_pivot_positions]
    selected_names = [names[i] for i in selected_indices]
    
    # Removed = everything not selected, in the order they'd be removed
    # (last in pivot order = first to remove)
    removed_pivot_positions = list(range(n_cols - 1, min_cols - 1, -1))
    removed_indices = [int(P[i]) for i in removed_pivot_positions]
    removed_names = [names[i] for i in removed_indices]
    
    # Build selected and removed lists
    selected = [{'index': idx, 'name': name} for idx, name in zip(selected_indices, selected_names)]
    removed = [{'index': idx, 'name': name} for idx, name in zip(removed_indices, removed_names)]
    
    # Build reduced matrix
    F_reduced = F[:, selected_indices]
    
    # Build history (for each step of "removal")
    history = []
    for k in range(n_cols, min_cols - 1, -1):
        cols_to_keep = [int(P[i]) for i in range(k)]
        F_temp = F[:, cols_to_keep]
        history.append(compute_metrics(F_temp))
    
    return {
        'algorithm': 'qr_pivoting',
        'selected': selected,
        'removed': removed,
        'F_reduced': F_reduced,
        'history': history,
        'final_metrics': history[-1],
        'pivot_order': P.tolist(),
    }


def greedy_residual(F: np.ndarray, min_cols: int, names: list) -> dict:
    """
    Greedy removal based on column reconstructability.
    
    At each step, remove the column that can be best reconstructed
    from the remaining columns (i.e., contributes least new information).
    """
    n_cols = F.shape[1]
    current_indices = list(range(n_cols))
    current_names = list(names)
    removed = []
    history = []
    
    F_current = F.copy()
    history.append(compute_metrics(F_current))
    
    while len(current_indices) > min_cols:
        n = F_current.shape[1]
        min_residual = float('inf')
        best_to_remove = None
        
        # For each column, see how well it can be reconstructed from others
        for i in range(n):
            # Create F without column i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            F_others = F_current[:, mask]
            col_i = F_current[:, i]
            
            # Least-squares fit: how well can we reconstruct col_i from F_others?
            if F_others.shape[1] > 0:
                coeffs, residuals, _, _ = np.linalg.lstsq(F_others, col_i, rcond=None)
                residual_norm = np.linalg.norm(col_i - F_others @ coeffs)
            else:
                residual_norm = np.linalg.norm(col_i)
            
            # Remove the column with smallest residual (easiest to reconstruct)
            if residual_norm < min_residual:
                min_residual = residual_norm
                best_to_remove = i
        
        removed_idx = current_indices[best_to_remove]
        removed_name = current_names[best_to_remove]
        removed.append({'index': removed_idx, 'name': removed_name})
        
        # Update
        mask = np.ones(n, dtype=bool)
        mask[best_to_remove] = False
        F_current = F_current[:, mask]
        current_indices = [idx for j, idx in enumerate(current_indices) if mask[j]]
        current_names = [name for j, name in enumerate(current_names) if mask[j]]
        
        history.append(compute_metrics(F_current))
    
    selected = [{'index': idx, 'name': name} for idx, name in zip(current_indices, current_names)]
    
    return {
        'algorithm': 'greedy_residual',
        'selected': selected,
        'removed': removed,
        'F_reduced': F_current,
        'history': history,
        'final_metrics': history[-1],
    }


def energy_based(F: np.ndarray, min_cols: int, names: list) -> dict:
    """
    Energy-based selection: keep columns with highest norms.
    
    Columns with larger norms typically contribute more to the forward problem.
    This is a simple, non-iterative approach.
    """
    n_cols = F.shape[1]
    
    # Compute column norms
    col_norms = np.linalg.norm(F, axis=0)
    
    # Sort indices by norm (descending)
    sorted_indices = np.argsort(-col_norms)
    
    # Select top min_cols
    selected_indices = sorted(sorted_indices[:min_cols].tolist())
    selected_names = [names[i] for i in selected_indices]
    
    # Removed are the rest
    removed_indices = sorted(sorted_indices[min_cols:].tolist())
    removed_names = [names[i] for i in removed_indices]
    
    selected = [{'index': idx, 'name': name} for idx, name in zip(selected_indices, selected_names)]
    removed = [{'index': idx, 'name': name} for idx, name in zip(removed_indices, removed_names)]
    
    # Build history (simulate removing in order of increasing norm)
    history = []
    for k in range(n_cols, min_cols - 1, -1):
        cols_to_keep = sorted_indices[:k].tolist()
        F_temp = F[:, cols_to_keep]
        history.append(compute_metrics(F_temp))
    
    F_reduced = F[:, selected_indices]
    
    return {
        'algorithm': 'energy_based',
        'selected': selected,
        'removed': removed,
        'F_reduced': F_reduced,
        'history': history,
        'final_metrics': history[-1],
    }


def hybrid_condition_correlation(F: np.ndarray, min_cols: int, names: list) -> dict:
    """
    Hybrid: combine condition number improvement and correlation reduction.
    
    At each step, compute a score for each column:
    score = alpha * condition_improvement + (1-alpha) * correlation_penalty
    Remove the column with best (lowest) score.
    """
    alpha = 0.5  # Weight between condition (alpha) and correlation (1-alpha)
    
    n_cols = F.shape[1]
    current_indices = list(range(n_cols))
    current_names = list(names)
    removed = []
    history = []
    
    F_current = F.copy()
    history.append(compute_metrics(F_current))
    current_cond = compute_condition_number(F_current)
    
    while len(current_indices) > min_cols:
        n = F_current.shape[1]
        corr = compute_correlation_matrix(F_current)
        
        best_score = float('-inf')  # Now we MAXIMIZE score
        best_to_remove = None
        
        for i in range(n):
            # Condition improvement
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            F_test = F_current[:, mask]
            new_cond = compute_condition_number(F_test)
            cond_improvement = current_cond - new_cond 
            
            # Correlation score (how correlated is this column with others?)
            mean_corr_i = np.mean(np.abs([corr[i, j] for j in range(n) if j != i]))
            
            # Combined score: HIGHER is better to remove
            # - High cond_improvement (removing improves κ) → good to remove
            # - High mean_corr_i (column is redundant) → good to remove
            score = alpha * cond_improvement + (1 - alpha) * mean_corr_i
            
            if score > best_score:
                best_score = score
                best_to_remove = i
        
        removed_idx = current_indices[best_to_remove]
        removed_name = current_names[best_to_remove]
        removed.append({'index': removed_idx, 'name': removed_name})
        
        # Update
        mask = np.ones(n, dtype=bool)
        mask[best_to_remove] = False
        F_current = F_current[:, mask]
        current_indices = [idx for j, idx in enumerate(current_indices) if mask[j]]
        current_names = [name for j, name in enumerate(current_names) if mask[j]]
        current_cond = compute_condition_number(F_current)
        
        history.append(compute_metrics(F_current))
    
    selected = [{'index': idx, 'name': name} for idx, name in zip(current_indices, current_names)]
    
    return {
        'algorithm': 'hybrid_condition_correlation',
        'selected': selected,
        'removed': removed,
        'F_reduced': F_current,
        'history': history,
        'final_metrics': history[-1],
    }


def leverage_scores(F: np.ndarray, min_cols: int, names: list) -> dict:
    """
    SVD-based selection using leverage scores.
    
    Leverage scores measure how much each column influences the dominant
    subspace. Keep columns with highest leverage.
    """
    n_probes, n_cols = F.shape
    
    # SVD
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    
    # Use top min_cols singular vectors to define the subspace
    k = min(min_cols, len(s))
    Vk = Vt[:k, :].T  # (n_cols, k)
    
    # Leverage score for each column: squared row norm of Vk
    leverage = np.sum(Vk**2, axis=1)
    
    # Sort by leverage (descending)
    sorted_indices = np.argsort(-leverage)
    
    # Select top min_cols
    selected_indices = sorted(sorted_indices[:min_cols].tolist())
    selected_names = [names[i] for i in selected_indices]
    
    # Removed
    removed_indices = sorted(sorted_indices[min_cols:].tolist())
    removed_names = [names[i] for i in removed_indices]
    
    selected = [{'index': idx, 'name': name} for idx, name in zip(selected_indices, selected_names)]
    removed = [{'index': idx, 'name': name} for idx, name in zip(removed_indices, removed_names)]
    
    # Build history
    history = []
    for m in range(n_cols, min_cols - 1, -1):
        cols_to_keep = sorted_indices[:m].tolist()
        F_temp = F[:, cols_to_keep]
        history.append(compute_metrics(F_temp))
    
    F_reduced = F[:, selected_indices]
    
    return {
        'algorithm': 'leverage_scores',
        'selected': selected,
        'removed': removed,
        'F_reduced': F_reduced,
        'history': history,
        'final_metrics': history[-1],
        'leverage_scores': leverage.tolist(),
    }


# =============================================================================
# Plotting
# =============================================================================

# =============================================================================
# Plotting functions
# =============================================================================

def plot_selection_history(history: list, algorithm: str, output_dir: Path):
    """
    Plot metrics evolution during column removal.
    Creates separate plots for each metric.
    """
    n_cols = [h['n_cols'] for h in history]
    conds = [h['condition_number'] for h in history]
    max_corrs = [h['max_correlation'] for h in history]
    max_cohs = [h['max_coherence'] for h in history]
    ranks = [h['rank'] for h in history]
    
    # Style settings matching paraview scripts
    FIGSIZE = (10, 6)
    GRID_ALPHA = 0.3
    DPI = 150
    COLOR_PRIMARY = '#1f77b4'
    COLOR_SECONDARY = '#ff7f0e'
    COLOR_TERTIARY = '#2ca02c'
    LINEWIDTH = 2.0
    MARKERSIZE = 6
    
    algo_name = algorithm.replace('_', ' ').title()
    
    # --- Plot 1: Condition Number ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogy(n_cols, conds, 'o-', color=COLOR_PRIMARY, linewidth=LINEWIDTH, 
                markersize=MARKERSIZE, markeredgecolor='white', markeredgewidth=1)
    ax.set_xlabel('Number of Columns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Condition Number (κ)', fontsize=12, fontweight='bold')
    ax.set_title(f'Selection Algorithm: {algo_name} — Matrix Conditioning', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.invert_xaxis()
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f'{algorithm}_condition_number.svg', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Max Correlation ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(n_cols, max_corrs, 'o-', color=COLOR_SECONDARY, linewidth=LINEWIDTH, 
            markersize=MARKERSIZE, markeredgecolor='white', markeredgewidth=1)
    ax.set_xlabel('Number of Columns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max |Correlation|', fontsize=12, fontweight='bold')
    ax.set_title(f'Selection Algorithm: {algo_name} — Case Correlation', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.invert_xaxis()
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f'{algorithm}_correlation.svg', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # --- Plot 3: Max Coherence ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(n_cols, max_cohs, 'o-', color=COLOR_TERTIARY, linewidth=LINEWIDTH, 
            markersize=MARKERSIZE, markeredgecolor='white', markeredgewidth=1)
    ax.set_xlabel('Number of Columns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max |Coherence|', fontsize=12, fontweight='bold')
    ax.set_title(f'Selection Algorithm: {algo_name} — Gram Matrix Coherence', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.invert_xaxis()
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / f'{algorithm}_coherence.svg', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # --- Plot 4: Rank ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(n_cols, ranks, 's-', color='#d62728', linewidth=LINEWIDTH, 
            markersize=MARKERSIZE, markeredgecolor='white', markeredgewidth=1)
    ax.set_xlabel('Number of Columns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rank', fontsize=12, fontweight='bold')
    ax.set_title(f'Selection Algorithm: {algo_name} — Matrix Rank', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.invert_xaxis()
    ax.tick_params(labelsize=11)
    ax.set_ylim(0, max(ranks) + 1)
    plt.tight_layout()
    plt.savefig(output_dir / f'{algorithm}_rank.svg', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # --- Plot 5: Singular Value Span by Iteration ---
    # Shows the spread (min-max) of singular values at each step
    # Goal: visualize how the spectrum compresses/expands as columns are removed
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    sv_min = [h['min_singular_value'] for h in history]
    sv_max = [h['max_singular_value'] for h in history]
    sv_all = [h['singular_values'] for h in history]
    
    # Compute median and quartiles for each iteration
    sv_median = [np.median(sv) for sv in sv_all]
    sv_q25 = [np.percentile(sv, 25) for sv in sv_all]
    sv_q75 = [np.percentile(sv, 75) for sv in sv_all]
    
    # Draw vertical spans (min to max) for each iteration
    for i, nc in enumerate(n_cols):
        # Full range line (min to max)
        ax.plot([nc, nc], [sv_min[i], sv_max[i]], color=COLOR_PRIMARY, 
                linewidth=3, alpha=0.6, solid_capstyle='round')
        # IQR box (25th to 75th percentile)
        ax.plot([nc, nc], [sv_q25[i], sv_q75[i]], color=COLOR_PRIMARY, 
                linewidth=8, alpha=0.4, solid_capstyle='round')
        # Median marker
        ax.plot(nc, sv_median[i], 'o', color=COLOR_SECONDARY, markersize=7, 
                markeredgecolor='white', markeredgewidth=1, zorder=5)
    
    # Connect medians with a line for trend visibility
    ax.plot(n_cols, sv_median, '--', color=COLOR_SECONDARY, linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Number of Columns', fontsize=12, fontweight='bold')
    ax.set_ylabel('Singular Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Selection Algorithm: {algo_name} — Singular Value Spread by Iteration', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.invert_xaxis()
    ax.tick_params(labelsize=11)
    
    # Add legend manually
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLOR_PRIMARY, linewidth=3, alpha=0.6, label='Min–Max range'),
        Line2D([0], [0], color=COLOR_PRIMARY, linewidth=8, alpha=0.4, label='IQR (25–75%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_SECONDARY, 
               markersize=8, label='Median'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{algorithm}_sv_span.svg', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    # --- Plot 6: Final Singular Value Spectrum ---
    # Line plot showing full spectrum of final reduced matrix
    # Goal: check that values descend smoothly (soft slope), not abruptly
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    final_sv = history[-1]['singular_values']
    final_n = history[-1]['n_cols']
    sv_indices = list(range(1, len(final_sv) + 1))
    
    ax.plot(sv_indices, final_sv, 'o-', color=COLOR_PRIMARY, linewidth=LINEWIDTH + 0.5, 
            markersize=MARKERSIZE + 2, markeredgecolor='white', markeredgewidth=1.5)
    
    # Annotate each point with its value
    for i, (x, y) in enumerate(zip(sv_indices, final_sv)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points', 
                    xytext=(0, 10), ha='center', fontsize=9, color='#333333')
    
    # Add condition number annotation
    cond_final = final_sv[0] / final_sv[-1] if final_sv[-1] > 0 else float('inf')
    ax.text(0.98, 0.95, f'κ = {cond_final:.2f}', transform=ax.transAxes, 
            fontsize=12, fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('Singular Value Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Singular Value (σ)', fontsize=12, fontweight='bold')
    ax.set_title(f'Selection Algorithm: {algo_name} — Final Spectrum ({final_n} columns)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=GRID_ALPHA, linestyle='--')
    ax.tick_params(labelsize=11)
    ax.set_xticks(sv_indices)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{algorithm}_sv_final.svg', dpi=DPI, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved 6 plots to: {output_dir}/")


# =============================================================================
# Main
# =============================================================================

ALGORITHMS = {
    'condition': greedy_condition,
    'correlation': greedy_correlation,
    'coherence': greedy_coherence,
    'qr': qr_pivoting,
    'residual': greedy_residual,
    'energy': energy_based,
    'hybrid': hybrid_condition_correlation,
    'leverage': leverage_scores,
}


def main():
    parser = argparse.ArgumentParser(
        description='Select columns from F matrix using various algorithms.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithms:
  condition   - Greedy removal minimizing condition number
  correlation - Greedy removal of most correlated column  
  coherence   - Greedy removal based on Gram matrix coherence
  qr          - QR decomposition with column pivoting
  residual    - Greedy removal of most reconstructable column
  energy      - Keep columns with highest norms
  hybrid      - Combines condition number and correlation
  leverage    - SVD-based leverage score selection

Examples:
  %(prog)s --matrix F.npy --algorithm condition --min-cols 6
  %(prog)s --matrix F.npy --names e1 e2 e3 e4 e5 e6 e7 e8 e9 --algorithm correlation
  %(prog)s --matrix F.npy --algorithm qr --out results/
        """
    )
    parser.add_argument('--matrix', type=Path, required=True, help='Path to F matrix (.npy)')
    parser.add_argument('--names', nargs='+', default=None,
                        help='Names for columns (in same order as matrix columns)')
    parser.add_argument('--algorithm', type=str, required=True, choices=list(ALGORITHMS.keys()),
                        help='Selection algorithm to use')
    parser.add_argument('--min-cols', type=int, default=6,
                        help='Minimum number of columns to keep (default: 6)')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output directory (default: same as matrix)')
    
    args = parser.parse_args()
    
    if not args.matrix.exists():
        print(f"ERROR: Matrix file not found: {args.matrix}")
        return 1
    
    # Load matrix
    print(f"Loading F matrix from {args.matrix}...")
    F = np.load(args.matrix)
    n_probes, n_cols = F.shape
    print(f"Matrix shape: {n_probes} probes × {n_cols} columns")
    
    # Setup names
    if args.names:
        if len(args.names) != n_cols:
            print(f"ERROR: Got {len(args.names)} names but matrix has {n_cols} columns")
            return 1
        names = args.names
    else:
        names = [f"col_{i}" for i in range(n_cols)]
    
    # Validate min_cols
    if args.min_cols < 1:
        print(f"ERROR: min-cols must be at least 1")
        return 1
    if args.min_cols > n_cols:
        print(f"ERROR: min-cols ({args.min_cols}) > number of columns ({n_cols})")
        return 1
    
    # Output directory
    out_dir = args.out or args.matrix.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run algorithm
    print(f"\nRunning {args.algorithm} algorithm...")
    print(f"Target: reduce from {n_cols} to {args.min_cols} columns")
    
    algo_func = ALGORITHMS[args.algorithm]
    result = algo_func(F, args.min_cols, names)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.algorithm}")
    print(f"{'='*60}")
    print(f"Selected {len(result['selected'])} columns:")
    for item in result['selected']:
        print(f"  [{item['index']}] {item['name']}")
    
    print(f"\nRemoved {len(result['removed'])} columns (in order):")
    for item in result['removed']:
        print(f"  [{item['index']}] {item['name']}")
    
    print(f"\nFinal metrics:")
    for key, val in result['final_metrics'].items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4g}")
        else:
            print(f"  {key}: {val}")
    
    # Save reduced matrix
    F_reduced_path = out_dir / f"F_reduced_{args.algorithm}.npy"
    np.save(F_reduced_path, result['F_reduced'])
    print(f"\nSaved reduced matrix: {F_reduced_path}")
    
    # Save results JSON (without numpy array)
    result_json = {k: v for k, v in result.items() if k != 'F_reduced'}
    result_json['matrix_file'] = str(args.matrix)
    result_json['min_cols'] = args.min_cols
    
    json_path = out_dir / f"selection_{args.algorithm}.json"
    with open(json_path, 'w') as f:
        json.dump(result_json, f, indent=2)
    print(f"Saved results: {json_path}")
    
    # Plot history (now creates multiple files)
    print(f"\nGenerating plots...")
    plot_selection_history(result['history'], args.algorithm, out_dir)
    
    print(f"\n{'='*60}")
    return 0


if __name__ == '__main__':
    exit(main())
