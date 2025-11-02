"""
Algorithm I: Hybrid Greedy→SVD

Hybrid scheme: Greedy-by-Norm to preselect ~20 triad-free dipoles,
then re-rank using SVD leverage scores to pick the final best subset.

Time: O(n log n + nt + nm²) where n=36, t=84 triads, m=probes
Space: O(nm)
"""

import numpy as np
import argparse
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any
import sys

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


def select_dipoles_algorithm_I(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    forbidden_triads: List[Set[Tuple[int, int]]],
    all_dipoles: List[Tuple[int, int]],
    n_preselect: int = 20,
    n_final: int = 16,
    r_keep: int = 10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Algorithm I: Hybrid Greedy→SVD.
    
    Phase 1: Greedy-by-Norm to preselect n_preselect triad-free dipoles
    Phase 2: SVD on preselected set, re-rank by leverage scores, pick top n_final
    
    Parameters
    ----------
    F : (21, 36) array - Dipole to scalp transfer matrix
    B : (36, 9) array - Antenna to dipole matrix  
    W : (21, 21) array - Probe weighting (diagonal)
    forbidden_triads : List of forbidden triad sets
    all_dipoles : List of all 36 dipole pairs
    n_preselect : Number of dipoles to preselect with greedy
    n_final : Final number of dipoles after SVD re-ranking
    r_keep : SVD components to keep for leverage scoring
    verbose : Print progress
    
    Returns
    -------
    dict with keys: S, selected_dipoles, selected_indices, n_selected, 
                    preselect_dipoles, leverage_scores, condition_number, 
                    algorithm, parameters
    """
    n_dipoles = len(all_dipoles)
    
    if verbose:
        print("="*60)
        print("Algorithm I: Hybrid Greedy→SVD")
        print("="*60)
        print(f"Phase 1: Greedy preselect {n_preselect} dipoles")
        print(f"Phase 2: SVD re-rank to select top {n_final}")
    
    # Compute F_eff = W @ F
    F_eff = W @ F if W is not None else F
    
    # ========== PHASE 1: Greedy by Norm ==========
    if verbose:
        print("\n--- Phase 1: Greedy-by-Norm ---")
    
    norms = np.linalg.norm(F_eff, axis=0)
    sorted_indices = np.argsort(-norms)
    
    preselect_dipoles = []
    preselect_indices = []
    
    for idx in sorted_indices:
        if len(preselect_dipoles) >= n_preselect:
            break
        
        candidate = all_dipoles[idx]
        
        if check_triad_violation(preselect_dipoles + [candidate], forbidden_triads):
            continue
        
        preselect_dipoles.append(candidate)
        preselect_indices.append(idx)
        
        if verbose and len(preselect_dipoles) <= 10:
            print(f"  Preselect {len(preselect_dipoles):2d}: {candidate} (norm={norms[idx]:.4e})")
    
    if verbose:
        if len(preselect_dipoles) > 10:
            print(f"  ... ({len(preselect_dipoles) - 10} more)")
        print(f"\nPreselected {len(preselect_dipoles)} dipoles")
    
    # ========== PHASE 2: SVD Re-ranking ==========
    if verbose:
        print(f"\n--- Phase 2: SVD Leverage Scoring (r_keep={r_keep}) ---")
    
    # Extract preselected columns
    F_preselect = F_eff[:, preselect_indices]
    
    # SVD on preselected subset
    U, sigma, Vt = np.linalg.svd(F_preselect, full_matrices=False)
    
    # Keep top r_keep components
    r = min(r_keep, len(sigma))
    U_r = U[:, :r]
    Vt_r = Vt[:r, :]
    
    if verbose:
        var_explained = np.sum(sigma[:r]**2) / np.sum(sigma**2) * 100
        print(f"Kept {r} components, {var_explained:.2f}% variance")
    
    # Compute leverage scores for preselected dipoles (columns)
    # Leverage = row sums of (V_r @ V_r^T) = ||V_r^T[j, :]||^2 for each column j
    leverage_scores = np.sum(Vt_r**2, axis=0)
    
    # Rank preselected dipoles by leverage score
    score_ranking = np.argsort(-leverage_scores)
    
    # Select top n_final by leverage score (still checking triads)
    final_dipoles = []
    final_indices = []
    final_scores = []
    
    for rank_idx in score_ranking:
        if len(final_dipoles) >= n_final:
            break
        
        dipole_idx = preselect_indices[rank_idx]
        candidate = all_dipoles[dipole_idx]
        score = leverage_scores[rank_idx]
        
        # Check triads again (shouldn't form any since preselect was triad-free)
        if check_triad_violation(final_dipoles + [candidate], forbidden_triads):
            if verbose:
                print(f"  {candidate} (score={score:.4e}) → SKIP (triad)")
            continue
        
        final_dipoles.append(candidate)
        final_indices.append(dipole_idx)
        final_scores.append(score)
        
        if verbose:
            print(f"✓ {candidate} (leverage={score:.4e}) → ACCEPT")
    
    # Build S matrix
    S = build_s_matrix(final_indices, n_dipoles)
    
    # Compute condition number
    kappa = compute_condition_number(F, B, W, final_indices)
    
    if verbose:
        print(f"\nSelected {len(final_dipoles)} dipoles, κ={kappa:.2e}")
    
    return {
        'S': S,
        'selected_dipoles': final_dipoles,
        'selected_indices': final_indices,
        'n_selected': len(final_dipoles),
        'preselect_dipoles': preselect_dipoles,
        'leverage_scores': final_scores,
        'condition_number': kappa,
        'algorithm': 'I',
        'parameters': {
            'n_preselect': n_preselect,
            'n_final': n_final,
            'r_keep': r_keep
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Algorithm I: Hybrid Greedy→SVD selection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--base-matrices', type=str,
                       default='src/model/base_matrices',
                       help='Path to F, B, W matrices')
    parser.add_argument('--forbidden-triads', type=str,
                       default='src/model/forbidden_triads.npy',
                       help='Path to forbidden triads file')
    parser.add_argument('--n-preselect', type=int, default=20,
                       help='Number of dipoles to preselect with greedy')
    parser.add_argument('--n-final', type=int, default=16,
                       help='Final number after SVD re-ranking')
    parser.add_argument('--r-keep', type=int, default=10,
                       help='SVD components to keep for scoring')
    parser.add_argument('--output-dir', type=str,
                       default='results/algorithm_I',
                       help='Output directory for results')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results (console only)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Load data
    base_path = Path(args.base_matrices)
    matrices = load_base_matrices(base_path)
    
    triads_path = Path(args.forbidden_triads)
    forbidden_triads = load_forbidden_triads(triads_path)
    
    all_dipoles = build_dipoles()
    
    # Run algorithm
    result = select_dipoles_algorithm_I(
        F=matrices['F'],
        B=matrices['B'],
        W=matrices['W'],
        forbidden_triads=forbidden_triads,
        all_dipoles=all_dipoles,
        n_preselect=args.n_preselect,
        n_final=args.n_final,
        r_keep=args.r_keep,
        verbose=not args.quiet
    )
    
    # Save if requested and condition number is finite
    if not args.no_save:
        output_dir = Path(args.output_dir)
        
        # Build filename with non-default parameters
        filename = 'S_algorithm_I'
        if args.n_preselect != 20:
            filename += f'_pre{args.n_preselect}'
        if args.n_final != 16:
            filename += f'_n{args.n_final}'
        if args.r_keep != 10:
            filename += f'_r{args.r_keep}'
        
        save_selection_results(
            S=result['S'],
            metadata={k: v for k, v in result.items() if k != 'S'},
            output_dir=output_dir,
            filename=filename
        )
    
    return result


if __name__ == '__main__':
    main()
