"""
Algorithm B: Greedy by Norm

Selects dipoles with highest energy (L2 norm) that don't form forbidden triads.

Time complexity: O(n log n) for sorting + O(n·t) for triad checking
Space complexity: O(n) where n=36 dipoles, t=84 triads
"""

import numpy as np
import argparse
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from selection_algorithms.common import (
    load_forbidden_triads,
    check_triad_violation,
    compute_condition_number,
    save_selection_results
)
from model.utils import build_dipoles
from model.base_matrices.generate_base_matrices import build_s_matrix


def select_dipoles_algorithm_B(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    forbidden_triads: List[Set[Tuple[int, int]]],
    all_dipoles: List[Tuple[int, int]],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Algorithm B: Select dipoles by descending energy (L2 norm).
    
    Parameters
    ----------
    F : (21, 36) array - Dipole to scalp transfer matrix
    B : (36, 9) array - Antenna to dipole matrix  
    W : (21, 21) array - Probe weighting (diagonal)
    forbidden_triads : List of forbidden triad sets
    all_dipoles : List of all 36 dipole pairs
    verbose : Print progress
    
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected, 
                    norms, condition_number, algorithm, parameters
    """
    n_dipoles = len(all_dipoles)
    
    if verbose:
        print("="*60)
        print("Algorithm B: Greedy by Norm")
        print("="*60)
    
    # Compute F_eff = W @ F
    F_eff = W @ F if W is not None else F
    
    # Compute norms and sort
    norms = np.linalg.norm(F_eff, axis=0)
    sorted_indices = np.argsort(-norms)
    
    if verbose:
        print(f"Norm range: [{norms.min():.4e}, {norms.max():.4e}]")
    
    # Select greedily
    selected_dipoles = []
    selected_indices = []
    selected_norms = []
    
    for idx in sorted_indices:
        candidate = all_dipoles[idx]
        
        # Check if would form triad
        if check_triad_violation(selected_dipoles + [candidate], forbidden_triads):
            if verbose:
                print(f"  {candidate} (norm={norms[idx]:.4e}) → SKIP")
            continue
        
        selected_dipoles.append(candidate)
        selected_indices.append(idx)
        selected_norms.append(norms[idx])
        
        if verbose:
            print(f"✓ {candidate} (norm={norms[idx]:.4e}) → ACCEPT")
    
    # Build S matrix (no save, just return)
    S = build_s_matrix(selected_indices, n_dipoles)
    
    # Compute condition number using only selected columns
    cond = compute_condition_number(F, B, W, selected_indices)
    
    if verbose:
        print(f"\nSelected {len(selected_dipoles)} dipoles, κ={cond:.2e}")
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': selected_indices,
        'n_selected': len(selected_dipoles),
        'norms': selected_norms,
        'condition_number': kappa,
        'algorithm': 'greedy_energy_simple',
        'parameters': {}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm B: Greedy dipole selection by norm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Run directory (mesh-specific)')
    parser.add_argument('--f-matrix-path', type=str, required=True,
                       help='Path to F_matrix.npy')
    parser.add_argument('--b-matrix-path', type=str, required=True,
                       help='Path to B_matrix.npy')
    parser.add_argument('--w-matrix-path', type=str, default=None,
                       help='Path to W_matrix.npy (optional)')
    parser.add_argument('--forbidden-triads', type=str,
                       default='src/model/forbidden_triads.npy',
                       help='Path to forbidden triads file')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results (console only)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Build paths from run-dir
    run_dir = Path(args.run_dir)
    output_dir = run_dir / 'results' / 'greedy_energy_simple'
    
    # Load matrices
    F = np.load(Path(args.f_matrix_path))
    B = np.load(Path(args.b_matrix_path))
    W = np.load(Path(args.w_matrix_path)) if args.w_matrix_path else None
    
    triads_path = Path(args.forbidden_triads)
    forbidden_triads = load_forbidden_triads(triads_path)
    
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_algorithm_B(
        F=F,
        B=B,
        W=W,
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        filename = 'S_greedy_energy_simple'
        save_selection_results(
            S=result['S'],
            metadata={k: v for k, v in result.items() if k != 'S'},
            output_dir=output_dir,
            filename=filename
        )
    
    return result


if __name__ == '__main__':
    main()
