#!/usr/bin/env python3
"""
generate_test_cases.py

Generate realistic antenna voltage test cases for validation.

Usage:
    python3 generate_test_cases.py --n-cases 15 --output test_cases.csv

Output CSV format:
    case,v_1,v_2,v_3,v_4,v_5,v_6,v_7,v_8,v_9
    test_case_001,0.010000,0.000000,...
    test_case_002,0.020000,-0.020000,...
"""

import argparse
import csv
import numpy as np
from pathlib import Path


def generate_test_cases(n_cases: int, v_min: float, v_max: float, seed: int = 42, include_baselines: bool = True) -> list:
    """
    Generate diverse antenna voltage test cases.
    
    EEG-realistic range: 1-50 mV antenna voltages → 10-100 μV probe measurements
    
    Test case variety:
    - Baseline dipoles (e1e2, e1e3, e1e4, ..., e1e9 at 1V) - pre-computed
    - Random realistic cases (all antennas active with random voltages)
    
    Parameters
    ----------
    n_cases : int
        Total number of random test cases (baseline dipoles added separately)
    v_min : float
        Minimum antenna voltage (V) for random cases
    v_max : float
        Maximum antenna voltage (V) for random cases
    seed : int
        Random seed for reproducibility
    include_baselines : bool
        Include baseline 1V dipole cases (e1e2, e1e3, ..., e1e9)
        
    Returns
    -------
    list of tuples
        [(case_name, [v_1, v_2, ..., v_9]), ...]
    """
    cases = []
    
    # Add baseline dipole cases (1V) - these are pre-computed
    # All combinations: e1-e2, e1-e3, ..., e1-e9, e2-e3, e2-e4, ..., e8-e9
    if include_baselines:
        baseline_dipoles = []
        for i in range(1, 10):  # e1 to e9
            for j in range(i+1, 10):  # Only pairs where j > i (avoid duplicates)
                v = [0.0] * 9
                v[i-1] = 1.0   # First electrode positive
                v[j-1] = -1.0  # Second electrode negative
                case_name = f"dipole_e{i}e{j}_1V"
                baseline_dipoles.append((case_name, v))
        cases.extend(baseline_dipoles)
    
    # Add random realistic cases
    np.random.seed(seed)
    for i in range(n_cases):
        v = np.random.uniform(v_min, v_max, 9)
        case_name = f"test_case_{i+1:03d}"
        cases.append((case_name, v.tolist()))
    
    return cases


def save_test_cases(cases: list, output_path: Path):
    """Save test cases to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['case', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9'])
        
        # Cases
        for case_name, voltages in cases:
            row = [case_name] + [f"{v:.6f}" for v in voltages]
            writer.writerow(row)
    
    print(f"✓ Generated {len(cases)} test cases: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate antenna voltage test cases for validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 15 random cases + 8 baseline dipoles with 1-50 mV range
  python3 generate_test_cases.py --n-cases 15 --output test_cases.csv
  
  # Generate 20 cases with custom range, no baselines
  python3 generate_test_cases.py --n-cases 20 --v-range 0.001,0.100 --no-baselines --output my_tests.csv

Test case types:
  - Baseline dipoles: 1V dipoles e1e2, e1e3, ..., e1e9 (pre-computed, fast)
  - Random realistic: All antennas active with random voltages
"""
    )
    
    parser.add_argument('--n-cases', type=int, default=15,
                       help='Number of random test cases to generate (default: 15)')
    parser.add_argument('--output', type=Path, default=Path('test_cases.csv'),
                       help='Output CSV file path (default: test_cases.csv)')
    parser.add_argument('--v-range', type=str, default='0.001,0.050',
                       help='Voltage range as min,max in Volts (default: 0.001,0.050 = 1-50 mV). Can use negative values for dipole patterns.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-baselines', action='store_true',
                       help='Exclude baseline 1V dipole cases (default: include them)')
    
    args = parser.parse_args()
    
    # Parse voltage range
    try:
        v_min, v_max = map(float, args.v_range.split(','))
        if v_min >= v_max:
            raise ValueError("v_min must be less than v_max")
    except Exception as e:
        print("✗ Invalid --v-range format. Use: min,max (e.g., 0.001,0.050 or -0.5,0.5)")
        print(f"  Error: {e}")
        return 1
    
    print(f"Generating test cases...")
    if not args.no_baselines:
        n_dipole_pairs = 9 * 8 // 2  # C(9,2) = 36 pairs
        print(f"  Baseline dipoles: {n_dipole_pairs} (1V, pre-computed)")
    print(f"  Random cases: {args.n_cases}")
    print(f"  Voltage range: {v_min*1000:.2f} - {v_max*1000:.2f} mV")
    print(f"  Random seed: {args.seed}")
    
    cases = generate_test_cases(args.n_cases, v_min, v_max, args.seed, include_baselines=not args.no_baselines)
    save_test_cases(cases, args.output)
    
    # Print summary
    print("\nTest case summary:")
    print(f"  Total cases: {len(cases)}")
    n_baseline = (9 * 8 // 2) if not args.no_baselines else 0
    n_random = len(cases) - n_baseline
    if n_baseline > 0:
        print(f"  Baseline 1V dipoles: {n_baseline} (pre-computed, solver will skip)")
    print(f"  Random realistic cases: {n_random}")
    
    print("\nReady to use with validate.py:")
    print(f"  --test-cases {args.output}")


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
