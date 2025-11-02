"""
Algorithm F: Min-Condition Greedy

At each step, add the dipole that minimizes κ(M).
Global optimization approach (expensive but optimal conditioning).

Time: O(nk² · min(m,k)) where k ≈ 20 final dipoles, ~700 SVD calls
Space: O(mk) for temporary matrices
"""

import numpy as np
from typing import List, Set, Tuple, Dict, Any
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from selection_algorithms.common import (
    load_base_matrices,
    load_forbidden_triads,
    check_triad_violation,
    compute_condition_number,
    save_selection_results
)
from model.utils import build_dipoles
from model.base_matrices.generate_base_matrices import build_s_matrix


def select_dipoles_algorithm_F(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Select dipoles by iteratively minimizing condition number.
    
    At each iteration, tests all remaining candidates and picks the one
    that gives minimum κ(M). Stops when no valid candidates remain.
    
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
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    condition_history, condition_number, algorithm, parameters
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
        print("Algorithm F: Min-Condition Greedy")
        print("="*60)
        print("WARNING: Computationally expensive (~700 SVD calls)")
        print()
    
    # Compute F_eff = W @ F (inline)
    F_eff = W @ F if W is not None else F
    
    selected_dipoles = []
    selected_indices = []
    condition_history = []
    
    iteration = 0
    while True:
        iteration += 1
        best_idx = None
        best_cond = np.inf
        
        # Test all remaining dipoles
        for idx in range(n_dipoles):
            if idx in selected_indices:
                continue
            
            candidate_dipole = all_dipoles[idx]
            
            # Check triad
            if check_triad_violation(selected_dipoles + [candidate_dipole], forbidden_triads):
                continue
            
            # Test condition number with candidate added
            test_indices = selected_indices + [idx]
            kappa_test = compute_condition_number(F, B, W, test_indices)
            
            if kappa_test < best_cond:
                best_cond = kappa_test
                best_idx = idx
        
        # If no valid candidate found, stop
        if best_idx is None:
            if verbose:
                print("\nNo more valid candidates. Stopping.")
            break
        
        # Add best candidate
        selected_dipoles.append(all_dipoles[best_idx])
        selected_indices.append(best_idx)
        condition_history.append(best_cond)
        
        if verbose:
            print(f"[{iteration:2d}] ✓ {all_dipoles[best_idx]} (κ={best_cond:.2e})")
    
    # Final S and condition number
    S = build_s_matrix(selected_indices, n_dipoles)
    kappa_final = compute_condition_number(F, B, W, selected_indices)
    
    if verbose:
        print(f"\nSelected {len(selected_dipoles)} dipoles, final κ={kappa_final:.2e}")
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': selected_indices,
        'n_selected': len(selected_dipoles),
        'condition_history': condition_history,
        'condition_number': kappa_final,
        'algorithm': 'F',
        'parameters': {}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm F: Min-Condition Greedy selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.selection_algorithms.algorithm_F_min_condition.select

WARNING: This algorithm is computationally expensive (~700 SVD evaluations).
It may take several seconds to complete.
        """
    )
    parser.add_argument('--base-matrices', type=str, default='src/model/base_matrices',
                        help='Path to directory with F.npy, B.npy, W.npy')
    parser.add_argument('--forbidden-triads', type=str, default='src/model/forbidden_triads.npy',
                        help='Path to forbidden triads file')
    parser.add_argument('--output-dir', type=str, default='results/algorithm_F',
                        help='Output directory for results')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Load matrices
    matrices = load_base_matrices(Path(args.base_matrices))
    forbidden_triads = load_forbidden_triads(Path(args.forbidden_triads))
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_algorithm_F(
        F=matrices['F'],
        B=matrices['B'],
        W=matrices['W'],
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        output_dir = Path(args.output_dir)
        filename = 'S_algorithm_F'
        save_selection_results(
            S=result['S'],
            metadata={k: v for k, v in result.items() if k != 'S'},
            output_dir=output_dir,
            filename=filename
        )
    
    return result


if __name__ == '__main__':
    main()
