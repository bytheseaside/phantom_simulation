"""
Algorithm H: Rank-Revealing QR (RRQR)

QR decomposition with column pivoting naturally orders columns by importance.
Inserts triad check within the pivot loop for a deterministic, linearly independent selection.

Time: O(nm² + nt) where n=21 probes, m=36 dipoles, t=84 triads
Space: O(nm)
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


def select_dipoles_algorithm_H(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    forbidden_triads: List[Set[Tuple[int, int]]],
    all_dipoles: List[Tuple[int, int]],
    n_dipoles_max: int = 20,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Algorithm H: Rank-Revealing QR with triad constraints.
    
    Uses scipy's QR decomposition with column pivoting to rank dipoles by importance.
    Greedily selects from highest-ranked dipoles that don't form triads.
    
    Parameters
    ----------
    F : (21, 36) array - Dipole to scalp transfer matrix
    B : (36, 9) array - Antenna to dipole matrix  
    W : (21, 21) array - Probe weighting (diagonal)
    forbidden_triads : List of forbidden triad sets
    all_dipoles : List of all 36 dipole pairs
    n_dipoles_max : Maximum dipoles to select
    verbose : Print progress
    
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected, 
                    r_values, condition_number, algorithm, parameters
    """
    from scipy.linalg import qr
    
    n_dipoles = len(all_dipoles)
    
    if verbose:
        print("="*60)
        print("Algorithm H: Rank-Revealing QR (RRQR)")
        print("="*60)
        print(f"Max dipoles: {n_dipoles_max}")
    
    # Compute F_eff = W @ F
    F_eff = W @ F if W is not None else F
    
    # QR decomposition with column pivoting
    # Q, R, P = qr(F_eff, pivoting=True, mode='economic')
    # P is a permutation array where F_eff[:, P] = Q @ R
    _, R, P = qr(F_eff, pivoting=True, mode='economic')
    
    if verbose:
        print(f"\nQR pivoting complete. Diagonal R values:")
        r_diag = np.abs(np.diag(R))
        print(f"  Range: [{r_diag.min():.4e}, {r_diag.max():.4e}]")
    
    # P gives us column importance ranking
    # Select greedily from this ordering, checking triads
    selected_dipoles = []
    selected_indices = []
    r_values = []
    
    for i, col_idx in enumerate(P):
        if len(selected_dipoles) >= n_dipoles_max:
            break
        
        candidate = all_dipoles[col_idx]
        
        # Check if would form triad
        if check_triad_violation(selected_dipoles + [candidate], forbidden_triads):
            if verbose:
                # R diagonal only has min(m, n) elements
                r_val = np.abs(R[min(i, R.shape[0]-1), min(i, R.shape[1]-1)]) if i < R.shape[0] else 0.0
                print(f"  [{i:2d}] {candidate} (R~{r_val:.4e}) → SKIP (triad)")
            continue
        
        selected_dipoles.append(candidate)
        selected_indices.append(col_idx)
        
        # Store R diagonal value if within bounds
        if i < R.shape[0]:
            r_values.append(np.abs(R[i, min(i, R.shape[1]-1)]))
        
        if verbose:
            r_val = r_values[-1] if r_values else 0.0
            print(f"✓ [{i:2d}] {candidate} (R[{i},{i}]={r_val:.4e}) → ACCEPT")
    
    # Build S matrix
    S = build_s_matrix(selected_indices, n_dipoles)
    
    # Compute condition number
    kappa = compute_condition_number(F, B, W, selected_indices)
    
    if verbose:
        print(f"\nSelected {len(selected_dipoles)} dipoles, κ={kappa:.2e}")
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': selected_indices,
        'n_selected': len(selected_dipoles),
        'r_values': r_values,
        'condition_number': kappa,
        'algorithm': 'rrqr_importance_ranking',
        'parameters': {
            'n_dipoles_max': n_dipoles_max
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm H: RRQR dipole selection',
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
    parser.add_argument('--n-dipoles-max', type=int, default=20,
                       help='Maximum number of dipoles to select')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results (console only)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Build paths from run-dir
    run_dir = Path(args.run_dir)
    output_dir = run_dir / 'results' / 'rrqr_importance_ranking'
    
    # Load matrices
    F = np.load(Path(args.f_matrix_path))
    B = np.load(Path(args.b_matrix_path))
    W = np.load(Path(args.w_matrix_path)) if args.w_matrix_path else None
    
    triads_path = Path(args.forbidden_triads)
    forbidden_triads = load_forbidden_triads(triads_path)
    
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_algorithm_H(
        F=F,
        B=B,
        W=W,
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        n_dipoles_max=args.n_dipoles_max,
        verbose=not args.quiet
    )
    
    # Save if requested and condition number is finite
    if not args.no_save:
        
        # Build filename with parameter
        if args.n_dipoles_max != 20:
            filename = f'S_rrqr_importance_ranking_nmax{args.n_dipoles_max}'
        else:
            filename = 'S_rrqr_importance_ranking'
        
        save_selection_results(
            S=result['S'],
            metadata={k: v for k, v in result.items() if k != 'S'},
            output_dir=output_dir,
            filename=filename
        )
    
    return result


if __name__ == '__main__':
    main()
