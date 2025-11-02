"""
Algorithm E: Greedy-by-Norm with Condition Cap

Select by descending energy, accept only if κ(M) ≤ kappa_max.
Trades selection size for guaranteed conditioning.

Time: O(n log n + nk²·min(m,k)) for sort + k SVD checks, k ≈ final count
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
    save_selection_results
)
from model.utils import build_dipoles
from model.base_matrices.generate_base_matrices import build_s_matrix


def select_dipoles_algorithm_E(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    kappa_max: float = 1e4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Select dipoles by norm with condition number cap.
    
    Sorts by energy, greedily accepts if (a) no triad and (b) κ(M) ≤ kappa_max.
    
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
    kappa_max : float, default=1e4
        Maximum allowed condition number
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    condition_numbers, condition_number, algorithm, parameters
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
        print("Algorithm E: Greedy-by-Norm with Condition Cap")
        print("="*60)
        print(f"κ_max = {kappa_max:.2e}")
        print()
    
    # Compute F_eff = W @ F (inline)
    F_eff = W @ F if W is not None else F
    
    # Sort by norm
    norms = np.linalg.norm(F_eff, axis=0)
    sorted_indices = np.argsort(-norms)
    
    selected_dipoles = []
    selected_indices = []
    condition_numbers = []
    
    for idx in sorted_indices:
        candidate_dipole = all_dipoles[idx]
        
        # Check triad
        if check_triad_violation(selected_dipoles + [candidate_dipole], forbidden_triads):
            if verbose:
                print(f"  {candidate_dipole} → SKIP (triad)")
            continue
        
        # Test condition number with candidate added
        test_indices = selected_indices + [idx]
        S_test = build_s_matrix(test_indices, n_dipoles)
        M_test = F_eff @ S_test @ B
        sigma = np.linalg.svd(M_test, compute_uv=False)
        kappa_test = sigma[0] / sigma[-1] if sigma[-1] > 1e-14 else np.inf
        
        if kappa_test <= kappa_max:
            selected_dipoles.append(candidate_dipole)
            selected_indices.append(idx)
            condition_numbers.append(kappa_test)
            
            if verbose:
                print(f"✓ {candidate_dipole} (κ={kappa_test:.2e}) → ACCEPT")
        else:
            if verbose:
                print(f"  {candidate_dipole} (κ={kappa_test:.2e}) → REJECT")
    
    # Final S and condition number
    S = build_s_matrix(selected_indices, n_dipoles)
    M = F_eff @ S @ B
    sigma_final = np.linalg.svd(M, compute_uv=False)
    kappa_final = sigma_final[0] / sigma_final[-1] if sigma_final[-1] > 1e-14 else np.inf
    
    if verbose:
        print(f"\nSelected {len(selected_dipoles)} dipoles, κ={kappa_final:.2e}")
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': selected_indices,
        'n_selected': len(selected_dipoles),
        'condition_numbers': condition_numbers,
        'condition_number': kappa_final,
        'algorithm': 'E',
        'parameters': {'kappa_max': kappa_max}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm E: Condition Cap selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: κ_max = 1e4
  python -m src.selection_algorithms.algorithm_E_condition_cap.select

  # Stricter conditioning
  python -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e3

  # More relaxed
  python -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e5
        """
    )
    parser.add_argument('--base-matrices', type=str, default='src/model/base_matrices',
                        help='Path to directory with F.npy, B.npy, W.npy')
    parser.add_argument('--forbidden-triads', type=str, default='src/model/forbidden_triads.npy',
                        help='Path to forbidden triads file')
    parser.add_argument('--kappa-max', type=float, default=1e4,
                        help='Maximum allowed condition number (default: 1e4)')
    parser.add_argument('--output-dir', type=str, default='results/algorithm_E',
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
    result = select_dipoles_algorithm_E(
        F=matrices['F'],
        B=matrices['B'],
        W=matrices['W'],
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        kappa_max=args.kappa_max,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        output_dir = Path(args.output_dir)
        # Include kappa_max in filename if non-default
        if abs(args.kappa_max - 1e4) > 1e-6:
            filename = f'S_algorithm_E_kappa{args.kappa_max:.0e}'
        else:
            filename = 'S_algorithm_E'
        save_selection_results(
            S=result['S'],
            metadata={k: v for k, v in result.items() if k != 'S'},
            output_dir=output_dir,
            filename=filename
        )
    
    return result


if __name__ == '__main__':
    main()
