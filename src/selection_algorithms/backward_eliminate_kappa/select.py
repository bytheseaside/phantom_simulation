"""
Algorithm L: Backward Eliminate Kappa

Start with all 36 dipoles, iteratively remove the column that minimizes κ(remaining).
Backward elimination for optimal conditioning.

Triad handling: Option 1 - If removal creates triad, try next candidate.

Time: O(n²k·min(m,k)) where n=36, k starts at 36, expensive
Space: O(mn) for condition number computation
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


def select_dipoles_backward_eliminate_kappa(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    n_dipoles_target: int = 18,
    target_kappa: float = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Backward elimination: start with all dipoles, remove to minimize κ.
    
    Starts with all 36 dipoles, iteratively removes the column whose removal
    gives the smallest condition number for the remaining set.
    
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
    target_kappa : float, optional
        Stop if κ <= target_kappa achieved
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    removed_dipoles, removed_indices, kappa_history, 
                    condition_number, algorithm, parameters, stop_reason
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
        print("Algorithm L: Backward Eliminate Kappa")
        print("="*60)
        print(f"Starting with all {n_dipoles} dipoles")
        print(f"Target: {n_dipoles_target} dipoles")
        if target_kappa is not None:
            print(f"Target κ: {target_kappa:.2e}")
        print()
    
    # Start with all dipoles
    remaining_indices = list(range(n_dipoles))
    remaining_dipoles = all_dipoles.copy()
    removed_indices = []
    removed_dipoles = []
    kappa_history = []
    stop_reason = "target_reached"
    
    # Compute initial condition number
    kappa_current = compute_condition_number(F, B, W, remaining_indices)
    kappa_history.append(kappa_current)
    
    if verbose:
        print(f"Initial: {len(remaining_indices)} dipoles, κ={kappa_current:.2e}")
    
    iteration = 0
    while len(remaining_indices) > n_dipoles_target:
        iteration += 1
        
        # Try removing each remaining dipole
        best_idx_to_remove = None
        best_kappa = np.inf
        
        for idx in remaining_indices:
            # Test removal
            test_indices = [i for i in remaining_indices if i != idx]
            test_dipoles = [all_dipoles[i] for i in test_indices]
            
            # Check if removal creates triad violation
            # (Triad handling option 1: skip if creates triad)
            if check_triad_violation(test_dipoles, forbidden_triads):
                continue
            
            # Compute κ without this column
            kappa_test = compute_condition_number(F, B, W, test_indices)
            
            if kappa_test < best_kappa:
                best_kappa = kappa_test
                best_idx_to_remove = idx
        
        # If no valid removal found (all create triads), stop
        if best_idx_to_remove is None:
            stop_reason = "no_valid_removal"
            if verbose:
                print(f"\n[{iteration}] No valid removal (all create triads). Stopping.")
            break
        
        # Remove the best candidate
        remaining_indices.remove(best_idx_to_remove)
        removed_indices.append(best_idx_to_remove)
        removed_dipoles.append(all_dipoles[best_idx_to_remove])
        kappa_history.append(best_kappa)
        kappa_current = best_kappa
        
        if verbose:
            print(f"[{iteration:2d}] Remove {all_dipoles[best_idx_to_remove]}: "
                  f"{len(remaining_indices)} dipoles, κ={best_kappa:.2e}")
        
        # Check target_kappa stop condition
        if target_kappa is not None and best_kappa <= target_kappa:
            stop_reason = "target_kappa_reached"
            if verbose:
                print(f"\nReached target κ={best_kappa:.2e} <= {target_kappa:.2e}. Stopping.")
            break
    
    # Build S matrix from remaining indices
    S = build_s_matrix(remaining_indices, n_dipoles)
    
    # Final condition number
    kappa_final = compute_condition_number(F, B, W, remaining_indices)
    
    if verbose:
        print(f"\nFinal: {len(remaining_indices)} dipoles, κ={kappa_final:.2e}")
        print(f"Stop reason: {stop_reason}")
    
    return {
        'S': S,
        'selected_dipoles': [all_dipoles[i] for i in remaining_indices],
        'selected_indices': remaining_indices,
        'n_selected': len(remaining_indices),
        'removed_dipoles': removed_dipoles,
        'removed_indices': removed_indices,
        'kappa_history': kappa_history,
        'condition_number': kappa_final,
        'algorithm': 'backward_eliminate_kappa',
        'parameters': {
            'n_dipoles_target': n_dipoles_target,
            'target_kappa': target_kappa
        },
        'stop_reason': stop_reason
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm L: Backward Eliminate Kappa selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: eliminate down to 18 dipoles
  python -m src.selection_algorithms.backward_eliminate_kappa.select \\
      --run-dir run --n-dipoles-target 18

  # Eliminate until target condition number
  python -m src.selection_algorithms.backward_eliminate_kappa.select \\
      --run-dir run --target-kappa 1e3

  # Both conditions
  python -m src.selection_algorithms.backward_eliminate_kappa.select \\
      --run-dir run --n-dipoles-target 20 --target-kappa 5e3

WARNING: This algorithm is very expensive (starts with 36 dipoles, ~630 SVD calls).
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
    parser.add_argument('--target-kappa', type=float, default=None,
                        help='Stop if condition number <= target_kappa')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Build paths from run-dir
    run_dir = Path(args.run_dir)
    output_dir = run_dir / 'results' / 'backward_eliminate_kappa'
    
    # Load matrices
    F = np.load(Path(args.f_matrix_path))
    B = np.load(Path(args.b_matrix_path))
    W = np.load(Path(args.w_matrix_path)) if args.w_matrix_path else None
    
    forbidden_triads = load_forbidden_triads(Path(args.forbidden_triads))
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_backward_eliminate_kappa(
        F=F,
        B=B,
        W=W,
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        n_dipoles_target=args.n_dipoles_target,
        target_kappa=args.target_kappa,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        # Build filename with parameters
        filename_parts = ['S_backward_eliminate_kappa']
        if args.n_dipoles_target != 18:
            filename_parts.append(f'n{args.n_dipoles_target}')
        if args.target_kappa is not None:
            filename_parts.append(f'kappa{args.target_kappa:.0e}')
        
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
