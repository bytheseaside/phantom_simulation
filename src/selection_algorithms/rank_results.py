"""
Rank and order algorithm results based on custom scoring functions.

Usage:
    python3 -m src.selection_algorithms.rank_results --results-dir results/ --output ranked_results.txt

Custom scoring:
    from src.selection_algorithms.rank_results import rank_results
    
    def my_scorer(metadata, S_matrix):
        # Lower is better
        return metadata['condition_number']
    
    ranked = rank_results('results/', scorer=my_scorer)
"""

import numpy as np
import json
from pathlib import Path
from typing import Callable, List, Dict, Any, Optional, Tuple
import argparse


def default_scorer(metadata: Dict[str, Any], S: np.ndarray) -> float:
    """
    Default scoring function: prioritize low condition number.
    
    Lower score = better result.
    
    Parameters
    ----------
    metadata : dict
        Metadata from _metadata.json file
    S : np.ndarray
        Selection matrix from .npy file
        
    Returns
    -------
    float
        Score (lower is better). Returns inf for invalid/infinite κ.
    """
    kappa = metadata.get('condition_number', np.inf)
    
    # Handle string representations
    if isinstance(kappa, str):
        if 'inf' in kappa.lower():
            return np.inf
        try:
            kappa = float(kappa)
        except:
            return np.inf
    
    if not np.isfinite(kappa):
        return np.inf
    
    return float(kappa)


def condition_and_size_scorer(metadata: Dict[str, Any], S: np.ndarray) -> float:
    """
    Score balancing condition number and number of dipoles.
    
    Rewards: low κ, more dipoles
    Formula: κ / sqrt(n_dipoles)
    """
    kappa = metadata.get('condition_number', np.inf)
    n = metadata.get('n_selected', 1)
    
    if isinstance(kappa, str):
        if 'inf' in kappa.lower():
            return np.inf
        kappa = float(kappa)
    
    if not np.isfinite(kappa) or n == 0:
        return np.inf
    
    return float(kappa) / np.sqrt(n)


def rank_results(
    results_dir: str,
    scorer: Optional[Callable[[Dict, np.ndarray], float]] = None,
    verbose: bool = True
) -> List[Tuple[str, Dict[str, Any], float]]:
    """
    Rank all results in results_dir using the provided scoring function.
    
    Parameters
    ----------
    results_dir : str
        Path to results directory containing algorithm_* subdirectories
    scorer : callable, optional
        Function(metadata, S_matrix) -> score (lower is better)
        Default: prioritize low condition number
    verbose : bool
        Print progress messages
        
    Returns
    -------
    list of tuples
        [(result_name, metadata, score), ...] sorted by score (best first)
    """
    if scorer is None:
        scorer = default_scorer
    
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Collect all results
    all_results = []
    
    for algo_dir in sorted(results_path.glob('algorithm_*')):
        if not algo_dir.is_dir():
            continue
        
        if verbose:
            print(f"Scanning {algo_dir.name}...")
        
        # Find all .npy files
        for npy_file in sorted(algo_dir.glob('S_*.npy')):
            # Find corresponding metadata
            metadata_file = npy_file.with_name(npy_file.stem + '_metadata.json')
            
            if not metadata_file.exists():
                if verbose:
                    print(f"  WARNING: No metadata for {npy_file.name}, skipping")
                continue
            
            # Load data
            try:
                S = np.load(npy_file)
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Compute score
                score = scorer(metadata, S)
                
                # Store result
                result_name = f"{algo_dir.name}/{npy_file.stem}"
                all_results.append((result_name, metadata, score))
                
                if verbose:
                    n = metadata.get('n_selected', '?')
                    kappa = metadata.get('condition_number', '?')
                    print(f"  ✓ {npy_file.stem}: n={n}, κ={kappa}, score={score:.4e}")
            
            except Exception as e:
                if verbose:
                    print(f"  ERROR loading {npy_file.name}: {e}")
                continue
    
    # Sort by score (lower is better)
    all_results.sort(key=lambda x: x[2])
    
    if verbose:
        print(f"\nTotal results ranked: {len(all_results)}")
    
    return all_results


def save_ranked_results(
    ranked: List[Tuple[str, Dict[str, Any], float]],
    output_file: str,
    scorer_name: str = "default"
):
    """
    Save ranked results to a text file.
    
    Parameters
    ----------
    ranked : list of tuples
        Output from rank_results()
    output_file : str
        Path to output file
    scorer_name : str
        Name/description of scoring function used
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RANKED ALGORITHM RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Scoring function: {scorer_name}\n")
        f.write(f"Total results: {len(ranked)}\n")
        f.write(f"Lower score = better\n")
        f.write("="*80 + "\n\n")
        
        for rank, (name, metadata, score) in enumerate(ranked, 1):
            # Skip infinite scores
            if not np.isfinite(score):
                continue
            
            f.write(f"Rank {rank:3d}: {name}\n")
            f.write(f"  Score: {score:.6e}\n")
            
            # Key metadata
            n = metadata.get('n_selected', '?')
            kappa = metadata.get('condition_number', '?')
            algo = metadata.get('algorithm', '?')
            params = metadata.get('parameters', {})
            
            f.write(f"  Algorithm: {algo}\n")
            f.write(f"  Dipoles selected: {n}\n")
            f.write(f"  Condition number: {kappa}\n")
            
            if params:
                param_str = ', '.join(f"{k}={v}" for k, v in params.items())
                f.write(f"  Parameters: {param_str}\n")
            
            f.write("\n")
        
        # Summary statistics
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        
        finite_scores = [s for _, _, s in ranked if np.isfinite(s)]
        if finite_scores:
            f.write(f"Best score: {min(finite_scores):.6e}\n")
            f.write(f"Worst score: {max(finite_scores):.6e}\n")
            f.write(f"Mean score: {np.mean(finite_scores):.6e}\n")
            f.write(f"Median score: {np.median(finite_scores):.6e}\n")
        
        infinite_count = sum(1 for _, _, s in ranked if not np.isfinite(s))
        if infinite_count > 0:
            f.write(f"\nResults with infinite/invalid scores: {infinite_count}\n")
    
    print(f"Ranked results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Rank algorithm results by custom scoring function',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scoring functions:
  default              - Minimize condition number (κ)
  condition_and_size   - Balance κ and dipole count: κ/sqrt(n)
  
Custom scorer example:
  Create a Python file with:
    def my_scorer(metadata, S_matrix):
        return metadata['condition_number'] * metadata['n_selected']
  
  Then use: --scorer-module my_file --scorer-function my_scorer
"""
    )
    
    parser.add_argument('--results-dir', type=str, default='results/',
                       help='Directory containing algorithm results')
    parser.add_argument('--output', type=str, default='ranked_results.txt',
                       help='Output file for ranked results')
    parser.add_argument('--scorer', type=str, default='default',
                       choices=['default', 'condition_and_size'],
                       help='Built-in scoring function')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress messages')
    
    args = parser.parse_args()
    
    # Select scorer
    if args.scorer == 'default':
        scorer = default_scorer
        scorer_name = "Minimize condition number (κ)"
    elif args.scorer == 'condition_and_size':
        scorer = condition_and_size_scorer
        scorer_name = "Balance condition and size: κ/sqrt(n_dipoles)"
    else:
        scorer = default_scorer
        scorer_name = "default"
    
    # Rank results
    ranked = rank_results(
        results_dir=args.results_dir,
        scorer=scorer,
        verbose=not args.quiet
    )
    
    # Save ranked results
    save_ranked_results(ranked, args.output, scorer_name)
    
    # Print top 5
    if not args.quiet:
        print("\n" + "="*80)
        print("TOP 5 RESULTS")
        print("="*80)
        
        for rank, (name, metadata, score) in enumerate(ranked[:5], 1):
            if not np.isfinite(score):
                continue
            n = metadata.get('n_selected', '?')
            kappa = metadata.get('condition_number', '?')
            print(f"{rank}. {name}")
            print(f"   Score={score:.4e}, n={n}, κ={kappa}")


if __name__ == '__main__':
    main()
