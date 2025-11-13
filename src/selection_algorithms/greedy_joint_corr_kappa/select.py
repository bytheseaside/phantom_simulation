"""
Algorithm N: Greedy Joint Correlation + Kappa

Greedily select columns by joint criterion: weighted sum of correlation and condition number.
criterion = alpha * max_corr + (1-alpha) * normalized_kappa

Time: O(n²km) where n=36, k≈20 selected, m=21 probes
Space: O(mn) for correlation and condition computation
"""

import numpy as np
from typing import List, Set, Tuple, Dict, Any
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from selection_algorithms.common import (
    load_forbidden_triads,
    check_triad_violation,
    compute_condition_number,
    save_selection_results
)
from model.utils import build_dipoles
from model.base_matrices.generate_base_matrices import build_s_matrix


def select_dipoles_greedy_joint_corr_kappa(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    alpha: float = 0.5,
    n_dipoles_max: int = 20,
    start_col: int = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Select dipoles by joint criterion: alpha*corr + (1-alpha)*kappa.
    
    Greedily selects columns by minimizing a weighted combination of:
    - Maximum correlation with selected columns (scaled 0-1)
    - Condition number of selected set (normalized by initial κ)
    
    Parameters
    ----------
    F : np.ndarray (21, 36)
        Dipole-to-scalp transfer matrix
    B : np.ndarray (36, 9)
        Antenna-to-dipole matrix
    W : np.ndarray (21, 21), optional
        Probe weighting matrix
    forbidden_triads : List[Set], optional
        Forbidden triads
    all_dipoles : List[Tuple], optional
        All 36 dipoles
    alpha : float, default=0.5
        Weight for correlation term (0=pure kappa, 1=pure correlation)
    n_dipoles_max : int, default=20
        Maximum number of dipoles to select
    start_col : int, optional
        Starting column (forced as first selection)
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    max_correlations, kappa_history, joint_scores,
                    condition_number, algorithm, parameters
    """
    # Build default dipole list if not provided
    if all_dipoles is None:
        all_dipoles = build_dipoles()
    
    # Load forbidden triads if not provided
    if forbidden_triads is None:
        triads_path = Path(__file__).parent.parent.parent / 'model' / 'forbidden_triads.npy'
        forbidden_triads = load_forbidden_triads(triads_path)
    
    n_dipoles = len(all_dipoles)
    
    if verbose:
        print("="*60)
        print("Algorithm N: Greedy Joint Correlation + Kappa")
        print("="*60)
        print(f"Alpha (corr weight): {alpha:.2f}")
        print(f"Max dipoles: {n_dipoles_max}")
        if start_col is not None:
            print(f"Starting column: {start_col}")
        print()
    
    # Compute F_eff = W @ F
    F_eff = W @ F if W is not None else F
    
    # Initialize with start_col if provided, else highest energy
    if start_col is not None:
        first_idx = start_col
    else:
        norms = np.linalg.norm(F_eff, axis=0)
        first_idx = np.argmax(norms)
    
    selected_dipoles = [all_dipoles[first_idx]]
    selected_indices = [first_idx]
    max_correlations = [0.0]
    kappa_history = []
    joint_scores = [0.0]
    
    # Initial condition number (for normalization)
    kappa_init = compute_condition_number(F, B, W, [first_idx])
    kappa_history.append(kappa_init)
    
    if verbose:
        print(f"[1] ✓ {all_dipoles[first_idx]} (κ={kappa_init:.2e}) → START")
    
    iteration = 1
    while len(selected_dipoles) < n_dipoles_max:
        iteration += 1
        
        # Find candidate with minimum joint score
        best_idx = None
        best_joint_score = np.inf
        best_max_corr = 0.0
        best_kappa = 0.0
        
        selected_cols = F_eff[:, selected_indices]
        
        for idx in range(n_dipoles):
            if idx in selected_indices:
                continue
            
            candidate_dipole = all_dipoles[idx]
            
            # Check triad
            if check_triad_violation(selected_dipoles + [candidate_dipole], forbidden_triads):
                continue
            
            # Compute max correlation
            candidate_col = F_eff[:, idx]
            corr_matrix = np.corrcoef(np.column_stack([candidate_col, selected_cols.T]).T)
            correlations = np.abs(corr_matrix[0, 1:])
            max_corr = np.max(correlations)
            
            # Compute condition number with candidate added
            test_indices = selected_indices + [idx]
            kappa_test = compute_condition_number(F, B, W, test_indices)
            
            # Normalize kappa by initial value
            kappa_norm = kappa_test / kappa_init if kappa_init > 0 else kappa_test
            
            # Joint score: alpha * max_corr + (1-alpha) * kappa_norm
            # Both terms scaled roughly 0-1
            joint_score = alpha * max_corr + (1 - alpha) * (kappa_norm / 100.0)  # Scale kappa term
            
            if joint_score < best_joint_score:
                best_joint_score = joint_score
                best_max_corr = max_corr
                best_kappa = kappa_test
                best_idx = idx
        
        # If no valid candidate, stop
        if best_idx is None:
            if verbose:
                print("\nNo more valid candidates. Stopping.")
            break
        
        # Accept candidate
        selected_dipoles.append(all_dipoles[best_idx])
        selected_indices.append(best_idx)
        max_correlations.append(best_max_corr)
        kappa_history.append(best_kappa)
        joint_scores.append(best_joint_score)
        
        if verbose:
            print(f"[{iteration:2d}] ✓ {all_dipoles[best_idx]} "
                  f"(score={best_joint_score:.4f}, corr={best_max_corr:.3f}, κ={best_kappa:.2e}) → ACCEPT")
    
    # Build S matrix
    S = build_s_matrix(selected_indices, n_dipoles)
    
    # Final condition number
    kappa_final = compute_condition_number(F, B, W, selected_indices)
    
    if verbose:
        print(f"\nSelected {len(selected_dipoles)} dipoles, κ={kappa_final:.2e}")
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': selected_indices,
        'n_selected': len(selected_dipoles),
        'max_correlations': max_correlations,
        'kappa_history': kappa_history,
        'joint_scores': joint_scores,
        'condition_number': kappa_final,
        'algorithm': 'greedy_joint_corr_kappa',
        'parameters': {
            'alpha': alpha,
            'n_dipoles_max': n_dipoles_max,
            'start_col': start_col
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm N: Greedy Joint Correlation + Kappa selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Balanced (alpha=0.5): equal weight to corr and kappa
  python -m src.selection_algorithms.greedy_joint_corr_kappa.select \\
      --run-dir run --alpha 0.5 --n-dipoles-max 20

  # Correlation-focused (alpha=0.8)
  python -m src.selection_algorithms.greedy_joint_corr_kappa.select \\
      --run-dir run --alpha 0.8 --n-dipoles-max 18

  # Conditioning-focused (alpha=0.2)
  python -m src.selection_algorithms.greedy_joint_corr_kappa.select \\
      --run-dir run --alpha 0.2 --n-dipoles-max 22 --start-col 5
        """
    )
    parser.add_argument('--run-dir', type=str, required=True,
                        help='Run directory (mesh-specific)')
    parser.add_argument('--f-matrix-path', type=str, required=True,
                        help='Path to F_matrix.npy')
    parser.add_argument('--b-matrix-path', type=str, required=True,
                        help='Path to B_matrix.npy')
    parser.add_argument('--w-matrix-path', type=str, default=None,
                        help='Path to W_matrix.npy (optional)')
    parser.add_argument('--forbidden-triads', type=str, default='src/model/forbidden_triads.npy',
                        help='Path to forbidden triads file')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for correlation term (0=pure kappa, 1=pure corr, default: 0.5)')
    parser.add_argument('--n-dipoles-max', type=int, default=20,
                        help='Maximum number of dipoles (default: 20)')
    parser.add_argument('--start-col', type=int, default=None,
                        help='Starting column index (optional)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Build paths from run-dir
    run_dir = Path(args.run_dir)
    output_dir = run_dir / 'results' / 'greedy_joint_corr_kappa'
    
    # Load matrices
    F = np.load(Path(args.f_matrix_path))
    B = np.load(Path(args.b_matrix_path))
    W = np.load(Path(args.w_matrix_path)) if args.w_matrix_path else None
    
    forbidden_triads = load_forbidden_triads(Path(args.forbidden_triads))
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_greedy_joint_corr_kappa(
        F=F,
        B=B,
        W=W,
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        alpha=args.alpha,
        n_dipoles_max=args.n_dipoles_max,
        start_col=args.start_col,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        # Build filename with parameters
        filename_parts = ['S_greedy_joint_corr_kappa']
        filename_parts.append(f'alpha{args.alpha:.2f}')
        if args.n_dipoles_max != 20:
            filename_parts.append(f'n{args.n_dipoles_max}')
        if args.start_col is not None:
            filename_parts.append(f'start{args.start_col}')
        
        filename = '_'.join(filename_parts)
        
        save_selection_results(
            S=result['S'],
            metadata={k: v for k, v in result.items() if k != 'S'},
            output_dir=output_dir,
            filename=filename
        )
    
    return result


if __name__ == '__main__':
    main()
