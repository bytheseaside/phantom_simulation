#!/usr/bin/env python3
"""
validate_comprehensive.py

Comprehensive validation of dipole selections against FEM solver simulations.

Usage:
  python validate_comprehensive.py --test-cases path/to/test_cases.csv [--validate-top-n N] [--n-parallel P]

Workflow:
  1. Load test cases from CSV
  2. Create validation_solver_runs/ directory, copy mesh/probes
  3. Run FEM solver (run_step_03.sh) with test antenna configurations
  4. Extract probe measurements (run_probing.sh) from solution VTU files
  5. Load F matrix, B matrix, all S matrices
  6. For each S matrix and each test case:
     - Compute F_full = F @ B @ V_antenna
     - Compute F_subset = F @ S @ V_antenna
     - Load solver probe measurements (ground truth)
     - Compare all three: SOLVER, F_FULL, F_SUBSET
  7. Generate graphs (PNG) and reports (MD, CSV)

Output:
  results/validation_results/algorithm_X/S_matrix_name/
    ├── graphs/
    │   ├── overview_rmse_bars.png
    │   ├── regional_heatmap.png
    │   ├── case_001_probe_comparison.png
    │   ├── case_001_errors.png
    │   ├── case_001_scatter.png
    │   └── ...
    ├── per_case/
    │   ├── case_001.csv
    │   ├── case_002.csv
    │   └── ...
    ├── report.md
    └── test_cases_data.csv
  
  results/validation_results/
    ├── summary_all.csv (ranked list of all S matrices)
    ├── algorithm_comparison_boxplot.png
    └── best_per_algorithm_bars.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import subprocess
from typing import List, Dict, Any

# Import our validation modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from validation import comparison, plotting, report_generator


# EEG 10-20 regions (from comparison.py)
EEG_REGIONS = {
    'Fp1': 'frontal', 'Fp2': 'frontal', 'F3': 'frontal', 'F4': 'frontal',
    'F7': 'frontal', 'F8': 'frontal', 'Fz': 'frontal',
    'C3': 'central', 'C4': 'central', 'Cz': 'central',
    'P3': 'parietal', 'P4': 'parietal', 'Pz': 'parietal',
    'O1': 'occipital', 'O2': 'occipital', 'Oz': 'occipital',
    'T3': 'temporal', 'T4': 'temporal', 'T5': 'temporal', 'T6': 'temporal'
}


def setup_solver_directory(workspace_root: Path, test_cases_csv: Path) -> Path:
    """
    Create validation_solver_runs/ directory with required files.
    Returns path to validation directory.
    """
    val_dir = workspace_root / 'results' / 'validation_solver_runs'
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create cases subdirectory
    cases_dir = val_dir / 'cases'
    cases_dir.mkdir(exist_ok=True)
    
    # Copy mesh and probes
    run_dir = workspace_root / 'run'
    
    mesh_src = run_dir / 'mesh.msh'
    probes_src = run_dir / 'probes.csv'
    
    if not mesh_src.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_src}")
    if not probes_src.exists():
        raise FileNotFoundError(f"Probes file not found: {probes_src}")
    
    shutil.copy2(mesh_src, val_dir / 'mesh.msh')
    shutil.copy2(probes_src, val_dir / 'probes.csv')
    
    # Copy test cases as simulation_cases.csv
    shutil.copy2(test_cases_csv, val_dir / 'simulation_cases.csv')
    
    print(f"✓ Setup validation directory: {val_dir}")
    return val_dir


def run_solver(val_dir: Path, n_parallel: int = 4):
    """
    Run FEM solver on test cases.
    Calls run_step_03.sh from workspace root.
    """
    workspace_root = val_dir.parent.parent
    script_path = workspace_root / 'src' / 'steps' / 'run_step_03.sh'
    
    if not script_path.exists():
        raise FileNotFoundError(f"Solver script not found: {script_path}")
    
    print(f"\n🔧 Running FEM solver with {n_parallel} parallel processes...")
    print(f"   Working directory: {val_dir}")
    print(f"   This may take several minutes...\n")
    
    # Run solver
    result = subprocess.run(
        ['bash', str(script_path), str(val_dir), '--n-parallel', str(n_parallel)],
        cwd=workspace_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Solver failed!")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        sys.exit(1)
    
    print("✓ Solver completed successfully")


def run_probing(val_dir: Path):
    """
    Extract probe measurements from VTU files.
    Calls run_probing.sh from workspace root.
    """
    workspace_root = val_dir.parent.parent
    script_path = workspace_root / 'src' / 'steps' / 'run_probing.sh'
    
    if not script_path.exists():
        raise FileNotFoundError(f"Probing script not found: {script_path}")
    
    print(f"\n📊 Extracting probe measurements...")
    
    result = subprocess.run(
        ['bash', str(script_path), str(val_dir)],
        cwd=workspace_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Probing failed!")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        sys.exit(1)
    
    print("✓ Probe extraction completed")


def load_matrices(workspace_root: Path) -> tuple:
    """
    Load F matrix, B matrix.
    Returns (F, B, probe_names)
    """
    results_dir = workspace_root / 'results'
    
    # Load F matrix
    F_path = results_dir / 'F_matrix.npy'
    if not F_path.exists():
        raise FileNotFoundError(f"F matrix not found: {F_path}")
    F = np.load(F_path)
    
    # Load B matrix
    B_path = results_dir / 'B_matrix.npy'
    if not B_path.exists():
        raise FileNotFoundError(f"B matrix not found: {B_path}")
    B = np.load(B_path)
    
    # Load probe names
    probes_csv = workspace_root / 'run' / 'probes.csv'
    probes_df = pd.read_csv(probes_csv)
    probe_names = probes_df['name'].tolist()
    
    print(f"✓ Loaded F matrix: {F.shape}")
    print(f"✓ Loaded B matrix: {B.shape}")
    print(f"✓ Loaded {len(probe_names)} probe names")
    
    return F, B, probe_names


def find_s_matrices(results_dir: Path, validate_top_n: int = None) -> List[Dict[str, Path]]:
    """
    Find all S matrices in results/ directory.
    Returns list of {algorithm, path, name}
    """
    s_matrices = []
    
    for algo_dir in results_dir.iterdir():
        if not algo_dir.is_dir():
            continue
        if algo_dir.name.startswith('validation'):
            continue
        
        # Look for S matrices
        for s_file in algo_dir.glob('S_*.npy'):
            s_matrices.append({
                'algorithm': algo_dir.name,
                'path': s_file,
                'name': s_file.stem
            })
    
    if validate_top_n is not None:
        # Load rankings and select top N per algorithm
        # For now, just take first N per algorithm
        algo_groups = {}
        for s_mat in s_matrices:
            algo = s_mat['algorithm']
            if algo not in algo_groups:
                algo_groups[algo] = []
            algo_groups[algo].append(s_mat)
        
        filtered = []
        for algo, mats in algo_groups.items():
            filtered.extend(mats[:validate_top_n])
        
        s_matrices = filtered
    
    print(f"✓ Found {len(s_matrices)} S matrices")
    return s_matrices


def validate_s_matrix(
    s_matrix_info: Dict,
    F: np.ndarray,
    B: np.ndarray,
    probe_names: List[str],
    test_cases: pd.DataFrame,
    val_dir: Path
) -> Dict[str, Any]:
    """
    Validate a single S matrix against all test cases.
    Returns comprehensive results dictionary.
    """
    S = np.load(s_matrix_info['path'])
    
    # Result accumulator
    result = {
        'algorithm': s_matrix_info['algorithm'],
        's_matrix_name': s_matrix_info['name'],
        'n_selected_dipoles': S.shape[1],
        'n_test_cases': len(test_cases),
        'per_case_results': []
    }
    
    # Regional accumulator
    all_regional = []
    
    # Metrics accumulators
    solver_vs_full_rmses = []
    solver_vs_subset_rmses = []
    full_vs_subset_rmses = []
    
    solver_vs_full_max_errors = []
    solver_vs_subset_max_errors = []
    
    solver_vs_full_mean_errors = []
    solver_vs_subset_mean_errors = []
    
    solver_vs_full_rel_rmses = []
    solver_vs_subset_rel_rmses = []
    
    # Process each test case
    for _, row in test_cases.iterrows():
        case_name = row['case']
        V_antenna = row[['v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9']].values
        
        # Compute forward models
        V_full = comparison.compute_full_forward(F, B, V_antenna)
        V_subset = comparison.compute_subset_forward(F, S, V_antenna)
        
        # Load solver probes
        solver_csv = val_dir / 'cases' / f'probes_{case_name}.csv'
        V_solver = comparison.load_solver_probes(solver_csv, probe_names)
        
        # Compute regions
        regions = [EEG_REGIONS.get(p, 'unknown') for p in probe_names]
        
        # Comprehensive comparison
        comp_result = comparison.compare_all_methods(
            V_solver, V_full, V_subset, probe_names, regions
        )
        
        # Store case result
        result['per_case_results'].append({
            'case_name': case_name,
            'V_solver': V_solver,
            'V_full': V_full,
            'V_subset': V_subset,
            'solver_vs_full': comp_result['solver_vs_full'],
            'solver_vs_subset': comp_result['solver_vs_subset'],
            'full_vs_subset': comp_result['full_vs_subset'],
            'regional': comp_result['regional']
        })
        
        # Accumulate metrics
        solver_vs_full_rmses.append(comp_result['solver_vs_full']['rmse'])
        solver_vs_subset_rmses.append(comp_result['solver_vs_subset']['rmse'])
        full_vs_subset_rmses.append(comp_result['full_vs_subset']['rmse'])
        
        solver_vs_full_max_errors.append(comp_result['solver_vs_full']['max_error'])
        solver_vs_subset_max_errors.append(comp_result['solver_vs_subset']['max_error'])
        
        solver_vs_full_mean_errors.append(comp_result['solver_vs_full']['mean_error'])
        solver_vs_subset_mean_errors.append(comp_result['solver_vs_subset']['mean_error'])
        
        solver_vs_full_rel_rmses.append(comp_result['solver_vs_full']['relative_rmse'])
        solver_vs_subset_rel_rmses.append(comp_result['solver_vs_subset']['relative_rmse'])
        
        all_regional.append(comp_result['regional'])
    
    # Aggregate results
    result['solver_vs_full_rmse'] = np.mean(solver_vs_full_rmses)
    result['solver_vs_subset_rmse'] = np.mean(solver_vs_subset_rmses)
    result['full_vs_subset_rmse'] = np.mean(full_vs_subset_rmses)
    
    result['solver_vs_full_max_error'] = np.mean(solver_vs_full_max_errors)
    result['solver_vs_subset_max_error'] = np.mean(solver_vs_subset_max_errors)
    
    result['solver_vs_full_mean_error'] = np.mean(solver_vs_full_mean_errors)
    result['solver_vs_subset_mean_error'] = np.mean(solver_vs_subset_mean_errors)
    
    result['solver_vs_full_relative_rmse'] = np.mean(solver_vs_full_rel_rmses)
    result['solver_vs_subset_relative_rmse'] = np.mean(solver_vs_subset_rel_rmses)
    
    result['regional_breakdown'] = all_regional
    
    return result


def generate_outputs(
    result: Dict[str, Any],
    probe_names: List[str],
    output_dir: Path
):
    """
    Generate all graphs and reports for a single S matrix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    graphs_dir = output_dir / 'graphs'
    graphs_dir.mkdir(exist_ok=True)
    
    per_case_dir = output_dir / 'per_case'
    per_case_dir.mkdir(exist_ok=True)
    
    # Regions
    regions = [EEG_REGIONS.get(p, 'unknown') for p in probe_names]
    
    # Overview RMSE bars
    plotting.plot_rmse_comparison(
        result['solver_vs_full_rmse'],
        result['solver_vs_subset_rmse'],
        result['full_vs_subset_rmse'],
        graphs_dir / 'overview_rmse_bars.png'
    )
    
    # Regional heatmap
    test_case_names = [c['case_name'] for c in result['per_case_results']]
    regional_data = [c['regional'] for c in result['per_case_results']]
    plotting.plot_regional_heatmap(
        test_case_names,
        regional_data,
        graphs_dir / 'regional_heatmap.png'
    )
    
    # Per-case graphs and CSVs
    for case_result in result['per_case_results']:
        case_name = case_result['case_name']
        V_solver = case_result['V_solver']
        V_full = case_result['V_full']
        V_subset = case_result['V_subset']
        
        # Probe comparison bars
        plotting.plot_probe_comparison_bars(
            probe_names, V_solver, V_full, V_subset, case_name,
            graphs_dir / f'{case_name}_probe_comparison.png'
        )
        
        # Error bars
        errors_solver_subset = V_solver - V_subset
        plotting.plot_error_bars(
            probe_names, errors_solver_subset, regions, case_name,
            graphs_dir / f'{case_name}_errors.png'
        )
        
        # Scatter plot
        plotting.plot_scatter_comparison(
            probe_names, V_solver, V_subset, regions, case_name,
            graphs_dir / f'{case_name}_scatter.png'
        )
        
        # CSV
        report_generator.generate_case_detail_csv(
            case_name, probe_names, regions, V_solver, V_full, V_subset,
            per_case_dir / f'{case_name}.csv'
        )
    
    # Combined CSV
    all_case_data = []
    for case_result in result['per_case_results']:
        all_case_data.append({
            's_matrix_name': result['s_matrix_name'],
            'case_name': case_result['case_name'],
            'probe_names': probe_names,
            'regions': regions,
            'V_solver': case_result['V_solver'],
            'V_full': case_result['V_full'],
            'V_subset': case_result['V_subset']
        })
    
    report_generator.generate_combined_test_cases_csv(
        all_case_data,
        output_dir / 'test_cases_data.csv'
    )
    
    # Markdown report
    report_generator.generate_s_matrix_report(
        result['s_matrix_name'],
        result['algorithm'],
        result,
        output_dir / 'report.md'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive validation of dipole selections against FEM solver'
    )
    parser.add_argument('--test-cases', required=True, type=Path,
                       help='Path to test_cases.csv')
    parser.add_argument('--validate-top-n', type=int, default=None,
                       help='Only validate top N S matrices per algorithm')
    parser.add_argument('--n-parallel', type=int, default=4,
                       help='Number of parallel solver processes')
    parser.add_argument('--skip-solver', action='store_true',
                       help='Skip solver execution (use existing results)')
    
    args = parser.parse_args()
    
    # Paths
    workspace_root = Path(__file__).parent.parent.parent
    test_cases_csv = args.test_cases.resolve()
    
    if not test_cases_csv.exists():
        print(f"❌ Test cases file not found: {test_cases_csv}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("COMPREHENSIVE DIPOLE SELECTION VALIDATION")
    print("="*70)
    
    # Load test cases
    test_cases = pd.read_csv(test_cases_csv)
    print(f"\n✓ Loaded {len(test_cases)} test cases from {test_cases_csv.name}")
    
    # Setup validation directory
    val_dir = setup_solver_directory(workspace_root, test_cases_csv)
    
    # Run solver and probing
    if not args.skip_solver:
        run_solver(val_dir, args.n_parallel)
        run_probing(val_dir)
    else:
        print("\n⚠️  Skipping solver execution (--skip-solver)")
    
    # Load matrices
    F, B, probe_names = load_matrices(workspace_root)
    
    # Find S matrices
    results_dir = workspace_root / 'results'
    s_matrices = find_s_matrices(results_dir, args.validate_top_n)
    
    if not s_matrices:
        print("❌ No S matrices found in results/ directory")
        sys.exit(1)
    
    # Validate each S matrix
    print(f"\n{'='*70}")
    print(f"VALIDATING {len(s_matrices)} S MATRICES")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for i, s_mat_info in enumerate(s_matrices, 1):
        print(f"[{i}/{len(s_matrices)}] {s_mat_info['algorithm']} / {s_mat_info['name']}")
        
        # Validate
        result = validate_s_matrix(s_mat_info, F, B, probe_names, test_cases, val_dir)
        all_results.append(result)
        
        # Generate outputs
        output_dir = results_dir / 'validation_results' / s_mat_info['algorithm'] / s_mat_info['name']
        generate_outputs(result, probe_names, output_dir)
        
        print(f"  RMSE (Solver vs F_subset): {result['solver_vs_subset_rmse']*1e6:.2f} μV")
        print()
    
    # Generate summary reports
    print(f"\n{'='*70}")
    print("GENERATING SUMMARY REPORTS")
    print(f"{'='*70}\n")
    
    summary_dir = results_dir / 'validation_results'
    
    # Summary CSV
    report_generator.generate_summary_csv(all_results, summary_dir / 'summary_all.csv')
    
    # Algorithm comparison graphs
    algo_rmses = {}
    for result in all_results:
        algo = result['algorithm']
        if algo not in algo_rmses:
            algo_rmses[algo] = []
        algo_rmses[algo].append(result['solver_vs_subset_rmse'] * 1e6)
    
    plotting.plot_algorithm_comparison_boxplot(
        algo_rmses,
        summary_dir / 'algorithm_comparison_boxplot.png'
    )
    
    # Best per algorithm
    algo_best = {}
    for result in all_results:
        algo = result['algorithm']
        rmse = result['solver_vs_subset_rmse'] * 1e6
        if algo not in algo_best or rmse < algo_best[algo]:
            algo_best[algo] = rmse
    
    plotting.plot_best_per_algorithm_bars(
        list(algo_best.keys()),
        list(algo_best.values()),
        summary_dir / 'best_per_algorithm_bars.png'
    )
    
    print(f"\n{'='*70}")
    print("✅ VALIDATION COMPLETE")
    print(f"{'='*70}\n")
    print(f"Results saved to: {summary_dir}")
    print()


if __name__ == '__main__':
    main()
