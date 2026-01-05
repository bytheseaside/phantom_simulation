#!/usr/bin/env python3
"""
Compare column selection results across all algorithms.

Generates:
  1. Summary comparison table (CSV + console)
  2. Intersection analysis (which columns are selected by multiple algorithms)
  3. Consensus ranking (columns ranked by selection frequency)

Usage:
  python compare_selections.py --dir run_phantom/9dof/selection
  python compare_selections.py --dir run_phantom/18dof/selection --out comparison.csv
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import csv


def load_selection_results(selection_dir: Path) -> dict:
    """Load all selection JSON files from subdirectories."""
    results = {}
    
    for algo_dir in sorted(selection_dir.iterdir()):
        if not algo_dir.is_dir():
            continue
        
        # Find the selection JSON file
        json_files = list(algo_dir.glob('selection_*.json'))
        if not json_files:
            continue
        
        json_file = json_files[0]
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        algo_name = algo_dir.name
        results[algo_name] = data
    
    return results


def build_comparison_table(results: dict) -> list[dict]:
    """Build comparison table with metrics for each algorithm."""
    table = []
    
    for algo_name, data in sorted(results.items()):
        final = data.get('final_metrics', {})
        selected = data.get('selected', [])
        removed = data.get('removed', [])
        
        # Extract selected column names
        selected_names = [s['name'] for s in selected]
        removed_names = [r['name'] for r in removed]
        
        row = {
            'algorithm': algo_name,
            'n_selected': len(selected),
            'n_removed': len(removed),
            'condition_number': final.get('condition_number', 0),
            'max_correlation': final.get('max_correlation', 0),
            'mean_correlation': final.get('mean_correlation', 0),
            'max_coherence': final.get('max_coherence', 0),
            'mean_coherence': final.get('mean_coherence', 0),
            'max_sv': final.get('max_singular_value', 0),
            'min_sv': final.get('min_singular_value', 0),
            'selected_columns': ', '.join(selected_names),
            'removed_columns': ', '.join(removed_names),
        }
        table.append(row)
    
    return table


def compute_intersection_analysis(results: dict) -> dict:
    """
    Analyze which columns are selected by multiple algorithms.
    
    Returns:
        - selection_counts: how many algorithms selected each column
        - removal_counts: how many algorithms removed each column
        - consensus_selected: columns selected by ALL algorithms
        - consensus_removed: columns removed by ALL algorithms
        - frequency_ranking: columns ranked by selection frequency
    """
    n_algos = len(results)
    
    # Count selections and removals
    selection_counter = Counter()
    removal_counter = Counter()
    
    for algo_name, data in results.items():
        selected = data.get('selected', [])
        removed = data.get('removed', [])
        
        for s in selected:
            selection_counter[s['name']] += 1
        for r in removed:
            removal_counter[r['name']] += 1
    
    # Consensus (selected/removed by ALL)
    all_columns = set(selection_counter.keys()) | set(removal_counter.keys())
    consensus_selected = [col for col in all_columns if selection_counter[col] == n_algos]
    consensus_removed = [col for col in all_columns if removal_counter[col] == n_algos]
    
    # Frequency ranking (most frequently selected first)
    frequency_ranking = selection_counter.most_common()
    
    return {
        'n_algorithms': n_algos,
        'selection_counts': dict(selection_counter),
        'removal_counts': dict(removal_counter),
        'consensus_selected': sorted(consensus_selected),
        'consensus_removed': sorted(consensus_removed),
        'frequency_ranking': frequency_ranking,
    }


def print_comparison_table(table: list[dict]):
    print("\n" + "=" * 100)
    print("ALGORITHM COMPARISON TABLE")
    print("=" * 100)
    
    # Header
    print(f"{'Algorithm':<12} {'κ':>8} {'max ρ':>8} {'mean ρ':>8} {'max coh':>8} {'Selected Columns':<40}")
    print("-" * 100)
    
    # Sort by condition number
    sorted_table = sorted(table, key=lambda x: x['condition_number'])
    
    for row in sorted_table:
        print(f"{row['algorithm']:<12} "
              f"{row['condition_number']:>8.2f} "
              f"{row['max_correlation']:>8.4f} "
              f"{row['mean_correlation']:>8.4f} "
              f"{row['max_coherence']:>8.4f} "
              f"{row['selected_columns']:<40}")
    
    print("=" * 100)
    print("κ = condition number, ρ = correlation, coh = coherence")


def print_intersection_analysis(analysis: dict):
    """Print intersection analysis to console."""
    print("\n" + "=" * 100)
    print("INTERSECTION ANALYSIS")
    print("=" * 100)
    
    n = analysis['n_algorithms']
    
    # Consensus
    print(f"\nConsensus Selected (by ALL {n} algorithms):")
    if analysis['consensus_selected']:
        print(f"  {', '.join(analysis['consensus_selected'])}")
    else:
        print("  (none)")
    
    print(f"\nConsensus Removed (by ALL {n} algorithms):")
    if analysis['consensus_removed']:
        print(f"  {', '.join(analysis['consensus_removed'])}")
    else:
        print("  (none)")
    
    # Frequency ranking
    print(f"\nColumn Selection Frequency (out of {n} algorithms):")
    print("-" * 50)
    for col, count in analysis['frequency_ranking']:
        bar = '█' * count + '░' * (n - count)
        pct = 100 * count / n
        print(f"  {col:<10} [{bar}] {count}/{n} ({pct:.0f}%)")
    
    print("=" * 100)


def save_comparison_csv(table: list[dict], output_path: Path):
    """Save comparison table to CSV."""
    fieldnames = [
        'algorithm', 'n_selected', 'n_removed', 'condition_number',
        'max_correlation', 'mean_correlation', 'max_coherence', 'mean_coherence',
        'max_sv', 'min_sv', 'selected_columns', 'removed_columns'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(table, key=lambda x: x['condition_number']))
    
    print(f"\nSaved comparison table: {output_path}")


def save_intersection_json(analysis: dict, output_path: Path):
    """Save intersection analysis to JSON."""
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Saved intersection analysis: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare column selection results across algorithms.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dir run_phantom/9dof/selection
  %(prog)s --dir run_phantom/18dof/selection --out comparison.csv
        """
    )
    parser.add_argument('--dir', type=Path, required=True,
                        help='Directory containing algorithm subdirectories')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output CSV file (default: {dir}/comparison.csv)')
    
    args = parser.parse_args()
    
    if not args.dir.exists():
        print(f"ERROR: Directory not found: {args.dir}")
        return 1
    
    # Load results
    print(f"Loading results from: {args.dir}")
    results = load_selection_results(args.dir)
    
    if not results:
        print("ERROR: No selection results found!")
        return 1
    
    print(f"Found {len(results)} algorithms: {', '.join(sorted(results.keys()))}")
    
    # Build comparison table
    table = build_comparison_table(results)
    print_comparison_table(table)
    
    # Intersection analysis
    analysis = compute_intersection_analysis(results)
    print_intersection_analysis(analysis)
    
    # Save outputs
    out_csv = args.out or (args.dir / 'comparison.csv')
    out_json = args.dir / 'intersection_analysis.json'
    
    save_comparison_csv(table, out_csv)
    save_intersection_json(analysis, out_json)
    
    return 0


if __name__ == '__main__':
    exit(main())
