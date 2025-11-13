"""
Algorithm K: Greedy Low Correlation

Select dipoles by maintaining low maximum correlation with already-selected set.
Replaces deleted Algorithm D with proper np.corrcoef implementation.

Two selection orders:
- 'energy_sorted': Sort by energy first, then greedily accept if max_corr < rho_max
- 'greedy_min_corr': Greedily select column with minimum max_correlation to selected set

Time: O(nkm) for energy_sorted, O(n²km) for greedy_min_corr where k ≈ 20 selected
Space: O(nm) for correlation computation
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


def select_dipoles_greedy_low_correlation(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    rho_max: float = 0.7,
    selection_order: str = 'energy_sorted',
    start_col: int = None,
    n_dipoles_max: int = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Select dipoles by maintaining low correlation.
    
    Two modes:
    1. 'energy_sorted': Sort by energy, greedily accept if max_corr < rho_max
    2. 'greedy_min_corr': Greedily select column with minimum max_correlation
    
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
    rho_max : float, default=0.7
        Maximum allowed correlation with selected set
    selection_order : str, required
        'energy_sorted' or 'greedy_min_corr'
    start_col : int, optional
        Starting column (only valid with 'greedy_min_corr' mode)
    n_dipoles_max : int, optional
        Maximum number of dipoles to select
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    max_correlations, condition_number, algorithm, parameters
    """
    # Validate selection_order
    if selection_order not in ['energy_sorted', 'greedy_min_corr']:
        raise ValueError(f"selection_order must be 'energy_sorted' or 'greedy_min_corr', got '{selection_order}'")
    
    # Validate start_col compatibility
    if start_col is not None and selection_order == 'energy_sorted':
        raise ValueError("start_col is incompatible with selection_order='energy_sorted'. Use 'greedy_min_corr' instead.")
    
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
        print("Algorithm K: Greedy Low Correlation")
        print("="*60)
        print(f"Selection order: {selection_order}")
        print(f"Max correlation: {rho_max}")
        if start_col is not None:
            print(f"Starting column: {start_col}")
        if n_dipoles_max is not None:
            print(f"Max dipoles: {n_dipoles_max}")
        print()
    
    # Compute F_eff = W @ F
    F_eff = W @ F if W is not None else F
    
    selected_dipoles = []
    selected_indices = []
    max_correlations = []
    
    if selection_order == 'energy_sorted':
        # Mode 1: Sort by energy, greedily accept if correlation ok
        norms = np.linalg.norm(F_eff, axis=0)
        sorted_indices = np.argsort(-norms)  # descending
        
        for idx in sorted_indices:
            # Check max dipoles
            if n_dipoles_max is not None and len(selected_dipoles) >= n_dipoles_max:
                break
            
            candidate_dipole = all_dipoles[idx]
            
            # Check triad
            if check_triad_violation(selected_dipoles + [candidate_dipole], forbidden_triads):
                if verbose:
                    print(f"  {candidate_dipole} → SKIP (triad)")
                continue
            
            # First dipole: accept
            if len(selected_dipoles) == 0:
                selected_dipoles.append(candidate_dipole)
                selected_indices.append(idx)
                max_correlations.append(0.0)
                if verbose:
                    print(f"✓ {candidate_dipole} (norm={norms[idx]:.4e}) → ACCEPT (first)")
                continue
            
            # Compute correlations with all selected columns
            candidate_col = F_eff[:, idx]
            selected_cols = F_eff[:, selected_indices]
            
            # Correlation matrix between candidate and selected
            corr_matrix = np.corrcoef(np.column_stack([candidate_col, selected_cols.T]).T)
            # First row contains correlations with candidate
            correlations = np.abs(corr_matrix[0, 1:])
            max_corr = np.max(correlations)
            
            # Accept if max_corr < rho_max
            if max_corr < rho_max:
                selected_dipoles.append(candidate_dipole)
                selected_indices.append(idx)
                max_correlations.append(max_corr)
                if verbose:
                    print(f"✓ {candidate_dipole} (max_corr={max_corr:.3f}) → ACCEPT")
            else:
                if verbose:
                    print(f"  {candidate_dipole} (max_corr={max_corr:.3f}) → SKIP (corr too high)")
    
    else:  # greedy_min_corr
        # Mode 2: Greedily select column with minimum max_correlation
        
        # Initialize with start_col if provided, else highest energy
        if start_col is not None:
            first_idx = start_col
        else:
            norms = np.linalg.norm(F_eff, axis=0)
            first_idx = np.argmax(norms)
        
        selected_dipoles.append(all_dipoles[first_idx])
        selected_indices.append(first_idx)
        max_correlations.append(0.0)
        
        if verbose:
            print(f"[1] ✓ {all_dipoles[first_idx]} → START")
        
        iteration = 1
        while True:
            iteration += 1
            
            # Check max dipoles
            if n_dipoles_max is not None and len(selected_dipoles) >= n_dipoles_max:
                break
            
            # Find candidate with minimum max_correlation
            best_idx = None
            best_max_corr = np.inf
            
            selected_cols = F_eff[:, selected_indices]
            
            for idx in range(n_dipoles):
                if idx in selected_indices:
                    continue
                
                candidate_dipole = all_dipoles[idx]
                
                # Check triad
                if check_triad_violation(selected_dipoles + [candidate_dipole], forbidden_triads):
                    continue
                
                # Compute max correlation with selected
                candidate_col = F_eff[:, idx]
                corr_matrix = np.corrcoef(np.column_stack([candidate_col, selected_cols.T]).T)
                correlations = np.abs(corr_matrix[0, 1:])
                max_corr = np.max(correlations)
                
                if max_corr < best_max_corr:
                    best_max_corr = max_corr
                    best_idx = idx
            
            # If no valid candidate, stop
            if best_idx is None:
                if verbose:
                    print("\nNo more valid candidates. Stopping.")
                break
            
            # Check if exceeds rho_max
            if best_max_corr >= rho_max:
                if verbose:
                    print(f"\nAll remaining candidates have max_corr >= {rho_max}. Stopping.")
                break
            
            # Accept
            selected_dipoles.append(all_dipoles[best_idx])
            selected_indices.append(best_idx)
            max_correlations.append(best_max_corr)
            
            if verbose:
                print(f"[{iteration:2d}] ✓ {all_dipoles[best_idx]} (max_corr={best_max_corr:.3f}) → ACCEPT")
    
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
        'max_correlations': max_correlations,
        'condition_number': kappa,
        'algorithm': 'greedy_low_correlation',
        'parameters': {
            'rho_max': rho_max,
            'selection_order': selection_order,
            'start_col': start_col,
            'n_dipoles_max': n_dipoles_max
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm K: Greedy Low Correlation selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Energy-sorted mode: sort by energy, accept if correlation ok
  python -m src.selection_algorithms.greedy_low_correlation.select \\
      --run-dir run --selection-order energy_sorted --rho-max 0.7

  # Greedy min-corr mode: select lowest correlation
  python -m src.selection_algorithms.greedy_low_correlation.select \\
      --run-dir run --selection-order greedy_min_corr --rho-max 0.5

  # With start column (only valid with greedy_min_corr)
  python -m src.selection_algorithms.greedy_low_correlation.select \\
      --run-dir run --selection-order greedy_min_corr --start-col 10 --rho-max 0.6
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
    parser.add_argument('--rho-max', type=float, default=0.7,
                        help='Maximum allowed correlation (default: 0.7)')
    parser.add_argument('--selection-order', type=str, required=True,
                        choices=['energy_sorted', 'greedy_min_corr'],
                        help='Selection order: energy_sorted or greedy_min_corr (REQUIRED)')
    parser.add_argument('--start-col', type=int, default=None,
                        help='Starting column (only valid with greedy_min_corr mode)')
    parser.add_argument('--n-dipoles-max', type=int, default=None,
                        help='Maximum number of dipoles to select')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to disk')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Build paths from run-dir
    run_dir = Path(args.run_dir)
    output_dir = run_dir / 'results' / 'greedy_low_correlation'
    
    # Load matrices
    F = np.load(Path(args.f_matrix_path))
    B = np.load(Path(args.b_matrix_path))
    W = np.load(Path(args.w_matrix_path)) if args.w_matrix_path else None
    
    forbidden_triads = load_forbidden_triads(Path(args.forbidden_triads))
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_greedy_low_correlation(
        F=F,
        B=B,
        W=W,
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        rho_max=args.rho_max,
        selection_order=args.selection_order,
        start_col=args.start_col,
        n_dipoles_max=args.n_dipoles_max,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        # Build filename with parameters
        filename_parts = ['S_greedy_low_correlation']
        filename_parts.append(args.selection_order)
        filename_parts.append(f'rho{args.rho_max:.2f}')
        if args.start_col is not None:
            filename_parts.append(f'start{args.start_col}')
        if args.n_dipoles_max is not None:
            filename_parts.append(f'max{args.n_dipoles_max}')
        
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
