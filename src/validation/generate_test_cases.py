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


def generate_test_cases(n_cases: int, v_min: float, v_max: float, seed: int = 42) -> list:
    """
    Generate diverse antenna voltage test cases.
    
    EEG-realistic range: 1-50 mV antenna voltages → 10-100 μV probe measurements
    
    Test case variety:
    - Single dominant antenna (focal source)
    - Dipole pairs (adjacent antennas, opposite polarity)
    - Uniform random (all antennas active)
    - Sparse random (few active antennas)
    
    Parameters
    ----------
    n_cases : int
        Total number of test cases
    v_min : float
        Minimum antenna voltage (V)
    v_max : float
        Maximum antenna voltage (V)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    list of tuples
        [(case_name, [v_1, v_2, ..., v_9]), ...]
    """
    np.random.seed(seed)
    cases = []
    
    # Divide cases into 4 types
    n_single = max(1, n_cases // 4)
    n_dipole = max(1, n_cases // 4)
    n_uniform = max(1, n_cases // 4)
    n_sparse = n_cases - n_single - n_dipole - n_uniform
    
    case_num = 1
    
    # Type 1: Single dominant antenna (focal source)
    for _ in range(n_single):
        v = np.zeros(9)
        idx = np.random.randint(0, 9)
        v[idx] = np.random.uniform(v_min * 5, v_max)  # 5x min for visibility
        case_name = f"test_case_{case_num:03d}"
        cases.append((case_name, v.tolist()))
        case_num += 1
    
    # Type 2: Dipole pairs (adjacent antennas, opposite polarity)
    for _ in range(n_dipole):
        v = np.zeros(9)
        idx1 = np.random.randint(0, 8)
        idx2 = idx1 + 1
        amp = np.random.uniform(v_min * 5, v_max)
        v[idx1] = amp
        v[idx2] = -amp
        case_name = f"test_case_{case_num:03d}"
        cases.append((case_name, v.tolist()))
        case_num += 1
    
    # Type 3: Uniform random (all antennas active)
    for _ in range(n_uniform):
        v = np.random.uniform(-v_max, v_max, 9)
        case_name = f"test_case_{case_num:03d}"
        cases.append((case_name, v.tolist()))
        case_num += 1
    
    # Type 4: Sparse random (only 2-4 antennas active)
    for _ in range(n_sparse):
        v = np.zeros(9)
        n_active = np.random.randint(2, 5)
        active_idx = np.random.choice(9, n_active, replace=False)
        v[active_idx] = np.random.uniform(-v_max, v_max, n_active)
        case_name = f"test_case_{case_num:03d}"
        cases.append((case_name, v.tolist()))
        case_num += 1
    
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
  # Generate 15 cases with 1-50 mV range
  python3 generate_test_cases.py --n-cases 15 --output test_cases.csv
  
  # Generate 20 cases with custom range
  python3 generate_test_cases.py --n-cases 20 --v-range 0.001,0.100 --output my_tests.csv

Test case types:
  - Single antenna: One dominant source (focal activation)
  - Dipole pairs: Two adjacent antennas with opposite polarity
  - Uniform random: All antennas active with random voltages
  - Sparse random: Only 2-4 antennas active
"""
    )
    
    parser.add_argument('--n-cases', type=int, default=15,
                       help='Number of test cases to generate (default: 15)')
    parser.add_argument('--output', type=Path, default=Path('test_cases.csv'),
                       help='Output CSV file path (default: test_cases.csv)')
    parser.add_argument('--v-range', type=str, default='0.001,0.050',
                       help='Voltage range as min,max in Volts (default: 0.001,0.050 = 1-50 mV). Can use negative values for dipole patterns.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
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
    
    print(f"Generating {args.n_cases} test cases...")
    print(f"  Voltage range: {v_min*1000:.2f} - {v_max*1000:.2f} mV")
    print(f"  Random seed: {args.seed}")
    
    cases = generate_test_cases(args.n_cases, v_min, v_max, args.seed)
    save_test_cases(cases, args.output)
    
    # Print summary
    print("\nTest case summary:")
    print(f"  Total cases: {len(cases)}")
    
    # Count types
    n_single = sum(1 for _, v in cases if sum(abs(x) > 1e-9 for x in v) == 1)
    n_dipole = sum(1 for _, v in cases if sum(abs(x) > 1e-9 for x in v) == 2 and sum(v) < 1e-9)
    n_sparse = sum(1 for _, v in cases if 2 < sum(abs(x) > 1e-9 for x in v) < 5)
    n_uniform = len(cases) - n_single - n_dipole - n_sparse
    
    print(f"  Single antenna: {n_single}")
    print(f"  Dipole pairs: {n_dipole}")
    print(f"  Sparse (2-4 active): {n_sparse}")
    print(f"  Uniform (all active): {n_uniform}")
    
    print("\nReady to use with validate_comprehensive.py:")
    print(f"  --test-cases {args.output}")


if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
