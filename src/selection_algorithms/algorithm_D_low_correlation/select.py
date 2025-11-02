"""
Algorithm D: Greedy by Low Correlation

Select dipoles with max pairwise correlation < rho_max.
Ensures diverse, uncorrelated selection for better conditioning.

Time: O(nk²) where k ≈ 20 selected dipoles (correlation checks)
Space: O(mn) for normalized matrix
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


def select_dipoles_algorithm_D(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    rho_max: float = 0.85,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Select dipoles using correlation-based diversity criterion.
    
    Greedily accepts dipoles if max correlation with already-selected < rho_max.
    
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
    rho_max : float, default=0.85
        Maximum allowed correlation between selected dipoles
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    max_correlations, condition_number, algorithm, parameters
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
        print("Algorithm D: Greedy by Low Correlation")
        print("="*60)
        print(f"Max correlation threshold: {rho_max}")
        print()
    
    # Compute F_eff = W @ F (inline)
    F_eff = W @ F if W is not None else F
    
    # Normalize columns to unit norm for correlation computation
    norms = np.linalg.norm(F_eff, axis=0)
    F_norm = F_eff / norms[np.newaxis, :]
    
    # Start with highest-norm dipole
    start_idx = np.argmax(norms)
    selected_dipoles = [all_dipoles[start_idx]]
    selected_indices = [start_idx]
    max_correlations = [0.0]  # First dipole has no prior correlation
    
    if verbose:
        print(f"✓ Starting: {all_dipoles[start_idx]} (norm={norms[start_idx]:.4e})")
        print()
    
    # Sort remaining by norm
    sorted_indices = np.argsort(-norms)
    
    for idx in sorted_indices:
        if idx == start_idx:
            continue
        
        candidate_dipole = all_dipoles[idx]
        
        # Check triad
        if check_triad_violation(selected_dipoles + [candidate_dipole], forbidden_triads):
            if verbose:
                print(f"  {candidate_dipole} → SKIP (triad)")
            continue
        
        # Compute max correlation with selected dipoles
        candidate_col = F_norm[:, idx]
        correlations = [np.abs(np.dot(candidate_col, F_norm[:, sel_idx])) 
                       for sel_idx in selected_indices]
        max_corr = max(correlations)
        
        # Check correlation criterion
        if max_corr < rho_max:
            selected_dipoles.append(candidate_dipole)
            selected_indices.append(idx)
            max_correlations.append(max_corr)
            
            if verbose:
                print(f"✓ {candidate_dipole} (max_corr={max_corr:.4f}) → ACCEPT")
        else:
            if verbose:
                print(f"  {candidate_dipole} (max_corr={max_corr:.4f}) → REJECT")
    
    # Build S matrix
    S = build_s_matrix(selected_indices, n_dipoles)
    
    # Compute condition number
    M = F_eff @ S @ B
    sigma = np.linalg.svd(M, compute_uv=False)
    kappa = sigma[0] / sigma[-1] if sigma[-1] > 1e-14 else np.inf
    
    if verbose:
        print(f"\nSelected {len(selected_dipoles)} dipoles, κ={kappa:.2e}")
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': selected_indices,
        'n_selected': len(selected_dipoles),
        'max_correlations': max_correlations,
        'condition_number': kappa,
        'algorithm': 'D',
        'parameters': {'rho_max': rho_max}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm D: Low Correlation selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: max correlation 0.85
  python -m src.selection_algorithms.algorithm_D_low_correlation.select

  # Stricter correlation constraint
  python -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.7

  # More relaxed
  python -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.95
        """
    )
    parser.add_argument('--base-matrices', type=str, default='src/model/base_matrices',
                        help='Path to directory with F.npy, B.npy, W.npy')
    parser.add_argument('--forbidden-triads', type=str, default='src/model/forbidden_triads.npy',
                        help='Path to forbidden triads file')
    parser.add_argument('--rho-max', type=float, default=0.85,
                        help='Max correlation threshold (default: 0.85)')
    parser.add_argument('--output-dir', type=str, default='results/algorithm_D',
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
    result = select_dipoles_algorithm_D(
        F=matrices['F'],
        B=matrices['B'],
        W=matrices['W'],
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        rho_max=args.rho_max,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        output_dir = Path(args.output_dir)
        # Include rho_max in filename if non-default
        if abs(args.rho_max - 0.85) > 1e-6:
            filename = f'S_algorithm_D_rho{args.rho_max:.2f}'
        else:
            filename = 'S_algorithm_D'
        save_selection_results(
            S=result['S'],
            metadata={k: v for k, v in result.items() if k != 'S'},
            output_dir=output_dir,
            filename=filename
        )
    
    return result


if __name__ == '__main__':
    main()
