"""
Algorithm O: Backward Eliminate Leverage

Start with all 36 dipoles, iteratively remove the column with LOWEST leverage score.
Backward elimination based on importance (leverage).

Time: O(nk·min(m,k)) where n=36, k starts at 36
Space: O(mk) for SVD computation
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


def select_dipoles_backward_eliminate_leverage(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    n_dipoles_target: int = 18,
    r_keep: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Backward elimination by leverage score: start with all, remove lowest leverage.
    
    Starts with all 36 dipoles, computes leverage scores from SVD, iteratively
    removes the column with lowest leverage (least important).
    
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
    n_dipoles_target : int, default=18
        Target number of dipoles (stop when reached)
    r_keep : int, default=10
        Number of top singular modes for leverage computation
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    removed_dipoles, removed_indices, leverage_history,
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
        print("Algorithm O: Backward Eliminate Leverage")
        print("="*60)
        print(f"Starting with all {n_dipoles} dipoles")
        print(f"Target: {n_dipoles_target} dipoles")
        print(f"r_keep: {r_keep} modes")
        print()
    
    # Start with all dipoles
    remaining_indices = list(range(n_dipoles))
    removed_indices = []
    removed_dipoles = []
    leverage_history = []
    
    if verbose:
        print(f"Initial: {len(remaining_indices)} dipoles")
    
    # Compute F_eff = W @ F
    F_eff = W @ F if W is not None else F
    
    iteration = 0
    while len(remaining_indices) > n_dipoles_target:
        iteration += 1
        
        # Compute SVD of current selection
        F_current = F_eff[:, remaining_indices]
        U, sigma, Vt = np.linalg.svd(F_current, full_matrices=False)
        V = Vt.T
        
        # Compute leverage scores: Σ(r=1..r_keep) σ_r² × V[j,r]²
        r_actual = min(r_keep, len(sigma))
        leverage_scores = np.zeros(len(remaining_indices))
        for r in range(r_actual):
            leverage_scores += (sigma[r]**2) * (V[:, r]**2)
        
        # Find column with lowest leverage (least important)
        # Must not create triad violation when removed
        best_local_idx = None
        best_leverage = np.inf
        
        for local_idx, global_idx in enumerate(remaining_indices):
            # Test removal
            test_indices = [i for i in remaining_indices if i != global_idx]
            test_dipoles = [all_dipoles[i] for i in test_indices]
            
            # Check if removal creates triad violation
            if check_triad_violation(test_dipoles, forbidden_triads):
                continue
            
            # Check leverage
            if leverage_scores[local_idx] < best_leverage:
                best_leverage = leverage_scores[local_idx]
                best_local_idx = local_idx
        
        # If no valid removal found, stop
        if best_local_idx is None:
            if verbose:
                print(f"\n[{iteration}] No valid removal (all create triads). Stopping.")
            break
        
        # Remove the column with lowest leverage
        global_idx_to_remove = remaining_indices[best_local_idx]
        remaining_indices.remove(global_idx_to_remove)
        removed_indices.append(global_idx_to_remove)
        removed_dipoles.append(all_dipoles[global_idx_to_remove])
        leverage_history.append(best_leverage)
        
        if verbose:
            print(f"[{iteration:2d}] Remove {all_dipoles[global_idx_to_remove]} "
                  f"(leverage={best_leverage:.4e}): {len(remaining_indices)} dipoles remain")
    
    # Build S matrix from remaining indices
    S = build_s_matrix(remaining_indices, n_dipoles)
    
    # Final condition number
    kappa_final = compute_condition_number(F, B, W, remaining_indices)
    
    if verbose:
        print(f"\nFinal: {len(remaining_indices)} dipoles, κ={kappa_final:.2e}")
    
    return {
        'S': S,
        'selected_dipoles': [all_dipoles[i] for i in remaining_indices],
        'selected_indices': remaining_indices,
        'n_selected': len(remaining_indices),
        'removed_dipoles': removed_dipoles,
        'removed_indices': removed_indices,
        'leverage_history': leverage_history,
        'condition_number': kappa_final,
        'algorithm': 'backward_eliminate_leverage',
        'parameters': {
            'n_dipoles_target': n_dipoles_target,
            'r_keep': r_keep
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm O: Backward Eliminate Leverage selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: eliminate down to 18 dipoles using top 10 modes
  python -m src.selection_algorithms.backward_eliminate_leverage.select \\
      --run-dir run --n-dipoles-target 18 --r-keep 10

  # More modes for leverage computation
  python -m src.selection_algorithms.backward_eliminate_leverage.select \\
      --run-dir run --n-dipoles-target 20 --r-keep 15

  # Fewer modes (faster)
  python -m src.selection_algorithms.backward_eliminate_leverage.select \\
      --run-dir run --n-dipoles-target 16 --r-keep 8
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
    parser.add_argument('--n-dipoles-target', type=int, default=18,
                        help='Target number of dipoles (default: 18)')
    parser.add_argument('--r-keep', type=int, default=10,
                        help='Number of top singular modes for leverage (default: 10)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Build paths from run-dir
    run_dir = Path(args.run_dir)
    output_dir = run_dir / 'results' / 'backward_eliminate_leverage'
    
    # Load matrices
    F = np.load(Path(args.f_matrix_path))
    B = np.load(Path(args.b_matrix_path))
    W = np.load(Path(args.w_matrix_path)) if args.w_matrix_path else None
    
    forbidden_triads = load_forbidden_triads(Path(args.forbidden_triads))
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_backward_eliminate_leverage(
        F=F,
        B=B,
        W=W,
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        n_dipoles_target=args.n_dipoles_target,
        r_keep=args.r_keep,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        # Build filename with parameters
        filename_parts = ['S_backward_eliminate_leverage']
        if args.n_dipoles_target != 18:
            filename_parts.append(f'n{args.n_dipoles_target}')
        if args.r_keep != 10:
            filename_parts.append(f'r{args.r_keep}')
        
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
