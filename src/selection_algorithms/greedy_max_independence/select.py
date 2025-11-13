"""
Algorithm J: Greedy Max Independence

Starting from start_col, greedily select the column with LARGEST residual norm
(most independent from currently selected set). Pure independence-driven selection.

Stop conditions: (1) n_dipoles_max reached, (2) no more valid candidates,
(3) eps_abs threshold (residual < eps_abs), (4) min_residual_ratio (residual/original < ratio)

Time: O(nkm) where k ≈ 20 selected, m = 21 probes
Space: O(mk) for QR decomposition
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


def select_dipoles_greedy_max_independence(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    start_col: int = 0,
    n_dipoles_max: int = None,
    eps_abs: float = None,
    min_residual_ratio: float = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Greedily select dipoles by maximum residual norm (maximum independence).
    
    Starts with start_col, then iteratively selects the column with the largest
    residual norm after projecting out currently selected columns.
    
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
    start_col : int, default=0
        Starting column (forced as first selection)
    n_dipoles_max : int, optional
        Stop after selecting this many dipoles
    eps_abs : float, optional
        Stop if residual norm < eps_abs (absolute threshold)
    min_residual_ratio : float, optional
        Stop if residual_norm / original_norm < min_residual_ratio
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    residual_norms, original_norms, condition_number, algorithm, parameters, stop_reason
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
        print("Algorithm J: Greedy Max Independence")
        print("="*60)
        print(f"Starting column: {start_col}")
        if n_dipoles_max is not None:
            print(f"Max dipoles: {n_dipoles_max}")
        if eps_abs is not None:
            print(f"Absolute threshold: {eps_abs:.2e}")
        if min_residual_ratio is not None:
            print(f"Min residual ratio: {min_residual_ratio:.2e}")
        print()
    
    # Compute F_eff = W @ F
    F_eff = W @ F if W is not None else F
    
    # Store original norms for ratio calculation
    original_norms = np.linalg.norm(F_eff, axis=0)
    
    # Initialize with start_col
    selected_dipoles = [all_dipoles[start_col]]
    selected_indices = [start_col]
    residual_norms = [original_norms[start_col]]
    F_selected = F_eff[:, start_col].reshape(-1, 1)
    stop_reason = "max_dipoles_reached"
    
    if verbose:
        print(f"[1] ✓ {all_dipoles[start_col]} (norm={original_norms[start_col]:.4e}) → START")
    
    iteration = 1
    while True:
        iteration += 1
        
        # Check max_dipoles stop condition
        if n_dipoles_max is not None and len(selected_dipoles) >= n_dipoles_max:
            stop_reason = "max_dipoles_reached"
            break
        
        # Compute QR for current selection
        Q, R = np.linalg.qr(F_selected, mode='reduced')
        
        # Find best candidate: maximum residual norm
        best_idx = None
        best_residual_norm = -np.inf
        
        for idx in range(n_dipoles):
            if idx in selected_indices:
                continue
            
            candidate_dipole = all_dipoles[idx]
            candidate_col = F_eff[:, idx]
            
            # Check triad
            if check_triad_violation(selected_dipoles + [candidate_dipole], forbidden_triads):
                continue
            
            # Compute residual
            projection = Q @ (Q.T @ candidate_col)
            residual = candidate_col - projection
            residual_norm = np.linalg.norm(residual)
            
            if residual_norm > best_residual_norm:
                best_residual_norm = residual_norm
                best_idx = idx
        
        # If no valid candidate found, stop
        if best_idx is None:
            stop_reason = "no_more_candidates"
            if verbose:
                print("\nNo more valid candidates. Stopping.")
            break
        
        # Check stopping conditions before accepting
        # Stop condition 3: eps_abs
        if eps_abs is not None and best_residual_norm < eps_abs:
            stop_reason = "eps_abs_reached"
            if verbose:
                print(f"\nResidual norm {best_residual_norm:.4e} < {eps_abs:.4e}. Stopping.")
            break
        
        # Stop condition 4: min_residual_ratio
        if min_residual_ratio is not None:
            original_norm = original_norms[best_idx]
            ratio = best_residual_norm / original_norm if original_norm > 0 else 0
            if ratio < min_residual_ratio:
                stop_reason = "min_residual_ratio_reached"
                if verbose:
                    print(f"\nResidual ratio {ratio:.4e} < {min_residual_ratio:.4e}. Stopping.")
                break
        
        # Accept candidate
        selected_dipoles.append(all_dipoles[best_idx])
        selected_indices.append(best_idx)
        residual_norms.append(best_residual_norm)
        F_selected = np.hstack([F_selected, F_eff[:, best_idx].reshape(-1, 1)])
        
        if verbose:
            print(f"[{iteration:2d}] ✓ {all_dipoles[best_idx]} (res={best_residual_norm:.4e}) → ACCEPT")
    
    # Build S matrix
    S = build_s_matrix(selected_indices, n_dipoles)
    
    # Compute condition number
    kappa = compute_condition_number(F, B, W, selected_indices)
    
    if verbose:
        print(f"\nSelected {len(selected_dipoles)} dipoles, κ={kappa:.2e}")
        print(f"Stop reason: {stop_reason}")
    
    return {
        'S': S,
        'selected_dipoles': selected_dipoles,
        'selected_indices': selected_indices,
        'n_selected': len(selected_dipoles),
        'residual_norms': residual_norms,
        'original_norms': original_norms.tolist(),
        'condition_number': kappa,
        'algorithm': 'greedy_max_independence',
        'parameters': {
            'start_col': start_col,
            'n_dipoles_max': n_dipoles_max,
            'eps_abs': eps_abs,
            'min_residual_ratio': min_residual_ratio
        },
        'stop_reason': stop_reason
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm J: Greedy Max Independence selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: start from column 0, select until 20 dipoles
  python -m src.selection_algorithms.greedy_max_independence.select \\
      --run-dir run --start-col 0 --n-dipoles-max 20

  # With absolute threshold
  python -m src.selection_algorithms.greedy_max_independence.select \\
      --run-dir run --start-col 5 --eps-abs 1e-6

  # With residual ratio threshold
  python -m src.selection_algorithms.greedy_max_independence.select \\
      --run-dir run --start-col 10 --min-residual-ratio 0.1
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
    parser.add_argument('--start-col', type=int, default=0,
                        help='Starting column index (0-35, default: 0)')
    parser.add_argument('--n-dipoles-max', type=int, default=None,
                        help='Maximum number of dipoles to select')
    parser.add_argument('--eps-abs', type=float, default=None,
                        help='Stop if residual norm < eps_abs')
    parser.add_argument('--min-residual-ratio', type=float, default=None,
                        help='Stop if residual/original < min_residual_ratio')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Build paths from run-dir
    run_dir = Path(args.run_dir)
    output_dir = run_dir / 'results' / 'greedy_max_independence'
    
    # Load matrices
    F = np.load(Path(args.f_matrix_path))
    B = np.load(Path(args.b_matrix_path))
    W = np.load(Path(args.w_matrix_path)) if args.w_matrix_path else None
    
    forbidden_triads = load_forbidden_triads(Path(args.forbidden_triads))
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_greedy_max_independence(
        F=F,
        B=B,
        W=W,
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        start_col=args.start_col,
        n_dipoles_max=args.n_dipoles_max,
        eps_abs=args.eps_abs,
        min_residual_ratio=args.min_residual_ratio,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        # Build filename with parameters
        filename_parts = ['S_greedy_max_independence']
        filename_parts.append(f'start{args.start_col}')
        if args.n_dipoles_max is not None:
            filename_parts.append(f'max{args.n_dipoles_max}')
        if args.eps_abs is not None:
            filename_parts.append(f'eps{args.eps_abs:.0e}')
        if args.min_residual_ratio is not None:
            filename_parts.append(f'ratio{args.min_residual_ratio:.2f}')
        
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
