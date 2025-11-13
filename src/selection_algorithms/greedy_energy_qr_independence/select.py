"""
Algorithm A: Linear Independence + Energy

Greedy selection by energy (L2 norm) with linear independence check.
Accepts dipole if residual norm after projection > eps_rel * original_norm.

Time: O(n log n + nkm) where k ≈ 20 selected, m = 21 probes
Space: O(mk) for storing selected columns
"""

import numpy as np
from typing import List, Set, Tuple, Dict, Any
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from selection_algorithms.common import (
    load_forbidden_triads,
    check_triad_violation,
    compute_condition_number,
    save_selection_results
)
from model.utils import build_dipoles
from model.base_matrices.generate_base_matrices import build_s_matrix


def select_dipoles_algorithm_A(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    eps_rel: float = 1e-6,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Select dipoles using Linear Independence + Energy criterion.
    
    Sorts by energy, greedily accepts if (a) no triad violation and
    (b) residual norm after projection > eps_rel * original norm.
    
    Parameters
    ----------
    F : np.ndarray (21, 36)
        Dipole-to-scalp transfer matrix
    B : np.ndarray (36, 9)
        Antenna-to-dipole matrix
    W : np.ndarray (21, 21), optional
        Probe weighting matrix. If None, uses identity.
    forbidden_triads : List[Set], optional
        List of forbidden dipole triads
    all_dipoles : List[Tuple], optional
        Ordered list of all 36 dipoles
    eps_rel : float, default=1e-6
        Relative tolerance for independence: accept if ||residual|| > eps_rel * ||original||
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    norms, residual_norms, condition_number, algorithm, parameters
    """
    # Build default dipole list if not provided
    if all_dipoles is None:
        all_dipoles = build_dipoles()
    
    # Load forbidden triads if not provided
    if forbidden_triads is None:
        triads_path = Path(__file__).parent.parent.parent / 'model' / 'forbidden_triads.npy'
        forbidden_triads = load_forbidden_triads(triads_path)
    
    n_dipoles = len(all_dipoles)
    n_probes = F.shape[0]
    
    if verbose:
        print("="*60)
        print("Algorithm A: Linear Independence + Energy")
        print("="*60)
    
    # Compute F_eff = W @ F
    F_eff = W @ F if W is not None else F
    
    # Compute norms
    norms = np.linalg.norm(F_eff, axis=0)
    sorted_indices = np.argsort(-norms)  # descending
    
    if verbose:
        print(f"Norm range: [{norms.min():.4e}, {norms.max():.4e}]")
        print(f"eps_rel: {eps_rel}")
        print()
    
    # Initialize
    selected_dipoles = []
    selected_indices = []
    selected_norms = []
    residual_norms = []
    F_selected = np.zeros((n_probes, 0))
    
    # Greedy selection
    for idx in sorted_indices:
        candidate_dipole = all_dipoles[idx]
        candidate_col = F_eff[:, idx]
        candidate_norm = norms[idx]
        
        # Check triad violation
        if check_triad_violation(selected_dipoles + [candidate_dipole], forbidden_triads):
            if verbose:
                print(f"  {candidate_dipole} (norm={candidate_norm:.4e}) → SKIP (triad)")
            continue
        
        # First dipole: accept
        if len(selected_dipoles) == 0:
            selected_dipoles.append(candidate_dipole)
            selected_indices.append(idx)
            selected_norms.append(candidate_norm)
            residual_norms.append(candidate_norm)
            F_selected = candidate_col.reshape(-1, 1)
            if verbose:
                print(f"✓ {candidate_dipole} (norm={candidate_norm:.4e}) → ACCEPT (first)")
            continue
        
        # Compute residual using QR projection
        Q, R = np.linalg.qr(F_selected, mode='reduced')
        projection = Q @ (Q.T @ candidate_col)
        residual = candidate_col - projection
        residual_norm = np.linalg.norm(residual)
        
        # Independence check
        threshold = eps_rel * candidate_norm
        if residual_norm > threshold:
            selected_dipoles.append(candidate_dipole)
            selected_indices.append(idx)
            selected_norms.append(candidate_norm)
            residual_norms.append(residual_norm)
            F_selected = np.hstack([F_selected, candidate_col.reshape(-1, 1)])
            if verbose:
                print(f"✓ {candidate_dipole} (norm={candidate_norm:.4e}, res={residual_norm:.4e}) → ACCEPT")
        else:
            if verbose:
                print(f"  {candidate_dipole} (norm={candidate_norm:.4e}, res={residual_norm:.4e}) → SKIP (dependent)")
    
    # Build S matrix (no save, just return)
    S = build_s_matrix(selected_indices, n_dipoles)
    
    # Compute condition number using only selected columns
    kappa = compute_condition_number(F, B, W, selected_indices)
    
    if verbose:
        print(f"\nSelected {len(selected_dipoles)} dipoles, κ={kappa:.2e}")
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': selected_indices,
        'n_selected': len(selected_dipoles),
        'norms': selected_norms,
        'residual_norms': residual_norms,
        'condition_number': kappa,
        'algorithm': 'greedy_energy_qr_independence',
        'parameters': {'eps_rel': eps_rel}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm A: Linear Independence + Energy selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.selection_algorithms.algorithm_A_linear_independence.select \\
      --run-dir run \\
      --eps-rel 1e-6

  # With custom F matrix location
  python -m src.selection_algorithms.algorithm_A_linear_independence.select \\
      --run-dir run \\
      --f-matrix-dir run2/f_matrix
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
    parser.add_argument('--eps-rel', type=float, default=1e-6,
                        help='Relative tolerance for independence check (default: 1e-6)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Build paths from run-dir
    run_dir = Path(args.run_dir)
    output_dir = run_dir / 'results' / 'greedy_energy_qr_independence'
    
    # Load matrices
    F = np.load(Path(args.f_matrix_path))
    B = np.load(Path(args.b_matrix_path))
    W = np.load(Path(args.w_matrix_path)) if args.w_matrix_path else None
    
    forbidden_triads = load_forbidden_triads(Path(args.forbidden_triads))
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_algorithm_A(
        F=F,
        B=B,
        W=W,
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        eps_rel=args.eps_rel,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        # Include eps_rel in filename if non-default
        if args.eps_rel != 1e-6:
            filename = f'S_greedy_energy_qr_independence_eps{args.eps_rel:.0e}'
        else:
            filename = 'S_greedy_energy_qr_independence'
        save_selection_results(
            S=result['S'],
            metadata={k: v for k, v in result.items() if k != 'S'},
            output_dir=output_dir,
            filename=filename
        )
    
    return result


if __name__ == '__main__':
    main()
