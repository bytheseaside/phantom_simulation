#!/usr/bin/env python3
"""
Generate random test cases for manifest validation.

Creates random boundary condition cases that can be used to:
1. Run the FEM solver and get numerical solutions
2. Compare with theoretical reconstructions using the F matrix

Test cases are generated with:
9dof mode: T = -R constraint (dipolar), S = 0
- Values in [-1e-3, 1e-3] range

Usage:
  python generate_test_cases.py --manifest manifest.json --n 5 --seed 42

The script modifies the manifest file directly, appending new test cases.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import sys


def generate_case(case_name: str, values: np.ndarray) -> dict:
    """
    Generate a 9dof test case where T = -R (dipolar constraint).
    
    Args:
        case_name: Name for the case
        values: Array of 9 values for T electrodes (R = -T, S = 0)
    
    Returns:
        Case dict matching manifest schema
    """
    assert len(values) == 9, "Need exactly 9 values for 9dof mode"
    
    dirichlet = []
    for i in range(1, 10):
        t_val = float(values[i - 1])
        r_val = -t_val  # T = -R constraint
        
        dirichlet.append({"name": f"e{i}_T", "value": t_val})
        dirichlet.append({"name": f"e{i}_R", "value": r_val})
        dirichlet.append({"name": f"e{i}_S", "value": 0.0})
    
    return {
        "name": case_name,
        "dirichlet": dirichlet
    }


def generate_test_cases(n_cases: int, seed: int, value_range: tuple = (-0.001, 0.001)) -> list:
    """
    Generate random test cases.
    
    Args:
        n_cases: Number of cases to generate
        seed: Random seed for reproducibility
        value_range: (min, max) range for random values
    
    Returns:
        List of case dicts
    """
    np.random.seed(seed)
    cases = []
    
    for i in range(n_cases):
        case_name = f"random_case_{i+1:03d}"
        
        values = np.random.uniform(value_range[0], value_range[1], 9)
        # Round to 6 decimal places for uV precision
        values = np.round(values, 6)
        case = generate_case(case_name, values)
        
        cases.append(case)
    
    return cases


def main():
    parser = argparse.ArgumentParser(description='Generate random test cases for manifest.')
    parser.add_argument('--manifest', type=Path, required=True, help='Path to manifest.json')
    parser.add_argument('--n', type=int, default=5, help='Number of test cases to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true', help='Print cases without modifying manifest')
    parser.add_argument('--value-min', type=float, default=-0.001, help='Minimum value for BCs')
    parser.add_argument('--value-max', type=float, default=0.001, help='Maximum value for BCs')
    
    args = parser.parse_args()
    
    if not args.manifest.exists():
        print(f"ERROR: Manifest file not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)
    
    # Load existing manifest
    with open(args.manifest, 'r') as f:
        manifest = json.load(f)
    
    # Check for existing case names
    existing_names = {case['name'] for case in manifest['cases']}
    
    # Generate new test cases
    print(f"Generating {args.n} test cases (seed={args.seed})...")
    new_cases = generate_test_cases(args.n, args.seed, (args.value_min, args.value_max))
    
    # Check for name conflicts
    for case in new_cases:
        if case['name'] in existing_names:
            print(f"WARNING: Case name '{case['name']}' already exists. Skipping.", file=sys.stderr)
            new_cases.remove(case)
    
    if not new_cases:
        print("No new cases to add.")
        return
    
    # Print summary
    print(f"\nGenerated {len(new_cases)} test cases:")
    for case in new_cases:
        # Extract T values for summary
        t_vals = [bc['value'] for bc in case['dirichlet'] if '_T' in bc['name']]
        r_vals = [bc['value'] for bc in case['dirichlet'] if '_R' in bc['name']]
        print(f"  {case['name']}:")
        print(f"    T: [{', '.join(f'{v:.2f}' for v in t_vals)}]")
        print(f"    R: [{', '.join(f'{v:.2f}' for v in r_vals)}]")
    
    if args.dry_run:
        print("\n[DRY RUN] Not modifying manifest.")
        print("\nJSON output:")
        print(json.dumps(new_cases, indent=2))
        return
    
    # Append to manifest
    manifest['cases'].extend(new_cases)
    
    # Save manifest
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nAppended {len(new_cases)} cases to {args.manifest}")
    print(f"Total cases in manifest: {len(manifest['cases'])}")


if __name__ == '__main__':
    main()
