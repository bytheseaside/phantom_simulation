"""
Algorithm C: SVD-Based Leverage Score

Select dipoles by contribution to top-r singular modes of F_eff.
Score[j] = Σ(r=1..r_keep) σ_r² × V[j,r]²

Time: O(mn² + nr) for SVD + scoring, r = min(r_keep, rank)
Space: O(mn) for matrices
"""

import numpy as np
from typing import List, Set, Tuple, Dict, Any
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
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


def select_dipoles_algorithm_C(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray = None,
    forbidden_triads: List[Set[Tuple[int, int]]] = None,
    all_dipoles: List[Tuple[int, int]] = None,
    r_keep: int = 10,
    selection_mode: str = 'all',
    n_dipoles_max: int = None,
    score_threshold: float = 0.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Select dipoles using SVD-based leverage scores.
    
    Computes leverage scores based on contribution to top r_keep singular modes,
    then selects by mode: 'all' (greedy all), 'top_n' (limit count), 'threshold' (min score).
    
    Parameters
    ----------
    F : np.ndarray (21, 36)
        Dipole-to-scalp transfer matrix
    B : np.ndarray (36, 9)
        Antenna-to-dipole matrix
    W : np.ndarray (21, 21), optional
        Probe weighting matrix
    forbidden_triads : List[Set], optional
        Forbidden dipole triads
    all_dipoles : List[Tuple], optional
        All 36 dipoles
    r_keep : int, default=10
        Number of top singular modes to consider
    selection_mode : str, default='all'
        'all': select all non-triad, 'top_n': limit to n_dipoles_max, 'threshold': score >= score_threshold
    n_dipoles_max : int, optional
        Max dipoles for 'top_n' mode
    score_threshold : float, default=0.0
        Minimum score for 'threshold' mode
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected,
                    scores, singular_values, variance_explained, condition_number, algorithm, parameters
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
        print("Algorithm C: SVD-Based Leverage Score")
        print("="*60)
        print(f"Modes: {r_keep}, Selection: {selection_mode}")
    
    # Compute F_eff = W @ F (inline)
    F_eff = W @ F if W is not None else F
    
    # Compute SVD
    U, sigma, Vt = np.linalg.svd(F_eff, full_matrices=False)
    V = Vt.T
    
    # Leverage scores: Σ(r=1..r_keep) σ_r² × V[j,r]²
    r_keep = min(r_keep, len(sigma))
    scores = np.zeros(n_dipoles)
    for r in range(r_keep):
        scores += (sigma[r]**2) * (V[:, r]**2)
    
    # Variance explained
    total_var = np.sum(sigma**2)
    kept_var = np.sum(sigma[:r_keep]**2)
    var_explained = kept_var / total_var if total_var > 0 else 0.0
    
    if verbose:
        print(f"Variance explained: {var_explained*100:.2f}%")
        print(f"Score range: [{scores.min():.4e}, {scores.max():.4e}]")
        print()
    
    # Sort by descending score
    sorted_indices = np.argsort(-scores)
    
    # Greedy selection with stopping criteria
    selected_dipoles = []
    selected_indices = []
    selected_scores = []
    
    for idx in sorted_indices:
        candidate_dipole = all_dipoles[idx]
        candidate_score = scores[idx]
        
        # Check stopping criteria
        if selection_mode == 'top_n' and n_dipoles_max is not None:
            if len(selected_dipoles) >= n_dipoles_max:
                break
        elif selection_mode == 'threshold':
            if candidate_score < score_threshold:
                break
        
        # Check triad violation
        if check_triad_violation(selected_dipoles + [candidate_dipole], forbidden_triads):
            if verbose:
                print(f"  {candidate_dipole} (score={candidate_score:.4e}) → SKIP (triad)")
            continue
        
        # Accept
        selected_dipoles.append(candidate_dipole)
        selected_indices.append(idx)
        selected_scores.append(candidate_score)
        
        if verbose:
            print(f"✓ {candidate_dipole} (score={candidate_score:.4e}) → ACCEPT")
    
    # Build S matrix
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
        'scores': selected_scores,
        'singular_values': sigma.tolist(),
        'variance_explained': var_explained,
        'condition_number': kappa,
        'algorithm': 'C',
        'parameters': {
            'r_keep': r_keep,
            'selection_mode': selection_mode,
            'n_dipoles_max': n_dipoles_max,
            'score_threshold': score_threshold
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm C: SVD-Based Leverage Score selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: all non-triad dipoles, top 10 modes
  python -m src.selection_algorithms.algorithm_C_svd_leverage.select

  # Top 15 dipoles from top 12 modes
  python -m src.selection_algorithms.algorithm_C_svd_leverage.select \\
      --r-keep 12 --selection-mode top_n --n-dipoles-max 15

  # Score threshold mode
  python -m src.selection_algorithms.algorithm_C_svd_leverage.select \\
      --selection-mode threshold --score-threshold 0.01
        """
    )
    parser.add_argument('--base-matrices', type=str, default='src/model/base_matrices',
                        help='Path to directory with F.npy, B.npy, W.npy')
    parser.add_argument('--forbidden-triads', type=str, default='src/model/forbidden_triads.npy',
                        help='Path to forbidden triads file')
    parser.add_argument('--r-keep', type=int, default=10,
                        help='Number of top singular modes to consider (default: 10)')
    parser.add_argument('--selection-mode', type=str, default='all',
                        choices=['all', 'top_n', 'threshold'],
                        help='Selection mode: all (greedy all), top_n (limit count), threshold (min score)')
    parser.add_argument('--n-dipoles-max', type=int, default=None,
                        help='Max dipoles for top_n mode')
    parser.add_argument('--score-threshold', type=float, default=0.0,
                        help='Minimum score for threshold mode (default: 0.0)')
    parser.add_argument('--output-dir', type=str, default='results/algorithm_C',
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
    result = select_dipoles_algorithm_C(
        F=matrices['F'],
        B=matrices['B'],
        W=matrices['W'],
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        r_keep=args.r_keep,
        selection_mode=args.selection_mode,
        n_dipoles_max=args.n_dipoles_max,
        score_threshold=args.score_threshold,
        verbose=not args.quiet
    )
    
    # Save if requested
    if not args.no_save:
        output_dir = Path(args.output_dir)
        
        # Build filename with non-default params
        filename_parts = ['S_algorithm_C']
        if args.r_keep != 10:
            filename_parts.append(f'r{args.r_keep}')
        if args.selection_mode == 'top_n' and args.n_dipoles_max is not None:
            filename_parts.append(f'top{args.n_dipoles_max}')
        elif args.selection_mode == 'threshold' and args.score_threshold != 0.0:
            filename_parts.append(f'thr{args.score_threshold:.0e}')
        
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
