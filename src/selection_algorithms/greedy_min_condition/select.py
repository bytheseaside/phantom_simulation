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
    max_dipoles: int = None,
    delta_threshold: float = None,
    target_kappa: float = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Select dipoles by iteratively minimizing condition number.
    
    At each iteration, tests all remaining candidates and picks the one
    that gives minimum κ(M). Stops when no valid candidates remain OR any stop condition met.
    
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
    max_dipoles : int, optional
        Stop after selecting this many dipoles
    delta_threshold : float, optional
        Stop if improvement (prev_kappa - new_kappa) < delta_threshold
    target_kappa : float, optional
        Stop if κ <= target_kappa achieved
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    condition_history, condition_number, algorithm, parameters, stop_reason
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
        if max_dipoles is not None:
            print(f"Stop: max_dipoles={max_dipoles}")
        if delta_threshold is not None:
            print(f"Stop: delta_threshold={delta_threshold:.2e}")
        if target_kappa is not None:
            print(f"Stop: target_kappa={target_kappa:.2e}")
        print()
    
    # Compute F_eff = W @ F  
    F_eff = W @ F if W is not None else F
    
    selected_dipoles = []
    selected_indices = []
    condition_history = []
    stop_reason = "no_more_candidates"
    
    iteration = 0
    while True:
        iteration += 1
        
        # Check max_dipoles stop condition
        if max_dipoles is not None and len(selected_dipoles) >= max_dipoles:
            stop_reason = "max_dipoles_reached"
            if verbose:
                print(f"\nReached max_dipoles={max_dipoles}. Stopping.")
            break
        
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
            stop_reason = "no_more_candidates"
            if verbose:
                print("\nNo more valid candidates. Stopping.")
            break
        
        # Add best candidate
        selected_dipoles.append(all_dipoles[best_idx])
        selected_indices.append(best_idx)
        condition_history.append(best_cond)
        
        if verbose:
            print(f"[{iteration:2d}] ✓ {all_dipoles[best_idx]} (κ={best_cond:.2e})")
        
        # Check target_kappa stop condition
        if target_kappa is not None and best_cond <= target_kappa:
            stop_reason = "target_kappa_reached"
            if verbose:
                print(f"\nReached target κ={best_cond:.2e} <= {target_kappa:.2e}. Stopping.")
            break
        
        # Check delta_threshold stop condition (improvement too small)
        if delta_threshold is not None and len(condition_history) >= 2:
            prev_kappa = condition_history[-2]
            improvement = prev_kappa - best_cond
            if improvement < delta_threshold:
                stop_reason = "delta_threshold_reached"
                if verbose:
                    print(f"\nImprovement Δκ={improvement:.2e} < {delta_threshold:.2e}. Stopping.")
                break
    
    # Final S and condition number
    S = build_s_matrix(selected_indices, n_dipoles)
    kappa_final = compute_condition_number(F, B, W, selected_indices)
    
    if verbose:
        print(f"\nSelected {len(selected_dipoles)} dipoles, final κ={kappa_final:.2e}")
        print(f"Stop reason: {stop_reason}")
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': selected_indices,
        'n_selected': len(selected_dipoles),
        'condition_history': condition_history,
        'condition_number': kappa_final,
        'algorithm': 'greedy_min_condition',
        'parameters': {
            'max_dipoles': max_dipoles,
            'delta_threshold': delta_threshold,
            'target_kappa': target_kappa
        },
        'stop_reason': stop_reason
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
    parser.add_argument('--max-dipoles', type=int, default=None,
                        help='Stop after selecting this many dipoles')
    parser.add_argument('--delta-threshold', type=float, default=None,
                        help='Stop if improvement (prev_kappa - new_kappa) < delta_threshold')
    parser.add_argument('--target-kappa', type=float, default=None,
                        help='Stop if condition number <= target_kappa achieved')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Build paths from run-dir
    run_dir = Path(args.run_dir)
    output_dir = run_dir / 'results' / 'greedy_min_condition'
    
    # Load matrices
    F = np.load(Path(args.f_matrix_path))
    B = np.load(Path(args.b_matrix_path))
    W = np.load(Path(args.w_matrix_path)) if args.w_matrix_path else None
    
    forbidden_triads = load_forbidden_triads(Path(args.forbidden_triads))
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_algorithm_F(
        F=F,
        B=B,
        W=W,
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        max_dipoles=args.max_dipoles,
        delta_threshold=args.delta_threshold,
        target_kappa=args.target_kappa,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        # Build filename with non-default params
        filename_parts = ['S_greedy_min_condition']
        if args.max_dipoles is not None:
            filename_parts.append(f'max{args.max_dipoles}')
        if args.delta_threshold is not None:
            filename_parts.append(f'delta{args.delta_threshold:.0e}')
        if args.target_kappa is not None:
            filename_parts.append(f'target{args.target_kappa:.0e}')
        
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
