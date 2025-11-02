"""
Algorithm G: D-Optimal Design

Maximize log det(Φ) where Φ = (FS)ᵀ(FS) is the information matrix.
At each step, pick dipole that maximizes Δlog-det. Stops when no improvement.

Time: O(nk³) where k ≈ 20 final dipoles (k matrix multiplies per iteration)
Space: O(k²) for information matrix
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


def select_dipoles_algorithm_G(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    regularization: float = 1e-8,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Select dipoles using D-optimal design (maximize log det of information matrix).
    
    At each iteration, picks dipole that maximizes Δlog-det(Φ).
    Stops when no candidate improves log-det.
    
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
    regularization : float, default=1e-8
        Regularization λ added to diagonal: det(Φ + λI)
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    logdet_history, final_logdet, condition_number, algorithm, parameters
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
        print("Algorithm G: D-Optimal Design")
        print("="*60)
        print(f"Regularization: {regularization:.2e}")
        print("WARNING: Computationally expensive (~700 log-det calls)")
        print()
    
    # Compute F_eff = W @ F (inline)
    F_eff = W @ F if W is not None else F
    
    selected_dipoles = []
    selected_indices = []
    logdet_history = []
    current_logdet = -np.inf
    
    iteration = 0
    while True:
        iteration += 1
        best_idx = None
        best_delta = -np.inf
        best_logdet = current_logdet
        
        # Test all remaining dipoles
        for idx in range(n_dipoles):
            if idx in selected_indices:
                continue
            
            candidate_dipole = all_dipoles[idx]
            
            # Check triad
            if check_triad_violation(selected_dipoles + [candidate_dipole], forbidden_triads):
                continue
            
            # Compute information matrix with candidate: Φ = (FS)ᵀ(FS)
            test_indices = selected_indices + [idx]
            S_test = build_s_matrix(test_indices, n_dipoles)
            F_selected = F_eff @ S_test
            Phi_test = F_selected.T @ F_selected
            
            # Add regularization and compute log-det
            Phi_reg = Phi_test + regularization * np.eye(Phi_test.shape[0])
            sign, logdet_test = np.linalg.slogdet(Phi_reg)
            
            if sign <= 0:
                logdet_test = -np.inf
            
            delta = logdet_test - current_logdet
            
            if delta > best_delta:
                best_delta = delta
                best_idx = idx
                best_logdet = logdet_test
        
        # If no valid candidate or no improvement, stop
        if best_idx is None or best_delta <= 0:
            if verbose:
                print("\nNo candidates improve log-det. Stopping.")
            break
        
        # Add best candidate
        selected_dipoles.append(all_dipoles[best_idx])
        selected_indices.append(best_idx)
        logdet_history.append(best_logdet)
        current_logdet = best_logdet
        
        if verbose:
            print(f"[{iteration:2d}] ✓ {all_dipoles[best_idx]} "
                  f"(log-det={best_logdet:.4f}, Δ={best_delta:.4f})")
    
    # Final S and condition number
    S = build_s_matrix(selected_indices, n_dipoles)
    kappa_final = compute_condition_number(F, B, W, selected_indices)
    
    if verbose:
        print(f"\nSelected {len(selected_dipoles)} dipoles")
        print(f"Final log-det: {current_logdet:.4f}, κ={kappa_final:.2e}")
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': selected_indices,
        'n_selected': len(selected_dipoles),
        'logdet_history': logdet_history,
        'final_logdet': current_logdet,
        'condition_number': kappa_final,
        'algorithm': 'G',
        'parameters': {'regularization': regularization}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm G: D-Optimal Design selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: regularization = 1e-8
  python -m src.selection_algorithms.algorithm_G_d_optimal.select

  # Custom regularization
  python -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-6

WARNING: This algorithm is computationally expensive (~700 log-det evaluations).
It may take several seconds to complete.
        """
    )
    parser.add_argument('--base-matrices', type=str, default='src/model/base_matrices',
                        help='Path to directory with F.npy, B.npy, W.npy')
    parser.add_argument('--forbidden-triads', type=str, default='src/model/forbidden_triads.npy',
                        help='Path to forbidden triads file')
    parser.add_argument('--regularization', type=float, default=1e-8,
                        help='Regularization for log-det (default: 1e-8)')
    parser.add_argument('--output-dir', type=str, default='results/algorithm_G',
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
    result = select_dipoles_algorithm_G(
        F=matrices['F'],
        B=matrices['B'],
        W=matrices['W'],
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        regularization=args.regularization,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        output_dir = Path(args.output_dir)
        # Include regularization in filename if non-default
        if abs(args.regularization - 1e-8) > 1e-12:
            filename = f'S_algorithm_G_reg{args.regularization:.0e}'
        else:
            filename = 'S_algorithm_G'
        save_selection_results(
            S=result['S'],
            metadata={k: v for k, v in result.items() if k != 'S'},
            output_dir=output_dir,
            filename=filename
        )
    
    return result


if __name__ == '__main__':
    main()
