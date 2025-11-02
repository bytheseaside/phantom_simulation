#!/usr/bin/env python3
"""
validate_with_solver.py

THE REAL VALIDATION: Compare dipole selection reconstruction against actual FEM solver results.

This is the mandatory validation that tests whether selected dipoles can accurately
reconstruct probe measurements from real physics-based simulations.

Process:
1. Generate test antenna voltage configurations
2. Run FEM solver (step 3) to compute ground truth probe measurements
3. Use selected dipoles to reconstruct probe measurements  
4. Compare solver ground truth vs dipole reconstruction
5. Analyze errors and spatial patterns

This validates the complete pipeline: antenna → solver → probes vs antenna → dipoles → probes
"""

import sys
from pathlib import Path
import numpy as np
import json
import csv
import subprocess
import tempfile
import shutil
from typing import Dict, List, Tuple, Any
import pyvista as pv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from selection_algorithms.common import load_base_matrices


# ============================================================================
# EEG Regions (same as before)
# ============================================================================

EEG_REGIONS = {
    'frontal_polar': ['Fp1', 'Fp2', 'FPz'],
    'frontal_left': ['F7', 'F3'],
    'frontal_midline': ['Fz'],
    'frontal_right': ['F4', 'F8'],
    'temporal_left': ['T3 (T7)', 'T5 (P7)'],
    'temporal_right': ['T4 (T8)', 'T6 (P8)'],
    'central_left': ['C3'],
    'central_midline': ['Cz'],
    'central_right': ['C4'],
    'parietal_left': ['P3'],
    'parietal_midline': ['Pz'],
    'parietal_right': ['P4'],
    'occipital': ['O1', 'Oz', 'O2']
}

PROBE_TO_REGION = {}
for region, probes in EEG_REGIONS.items():
    for probe in probes:
        PROBE_TO_REGION[probe] = region


def load_probe_coords(probes_csv: Path) -> Tuple[List[str], np.ndarray]:
    """Load probe names and coordinates."""
    names = []
    coords = []
    with open(probes_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not row[0].startswith('#'):
                names.append(row[0])
                coords.append([float(row[1]), float(row[2]), float(row[3])])
    return names, np.array(coords)


# ============================================================================
# Test Case Generation
# ============================================================================

def generate_antenna_test_cases(n_cases: int = 10, seed: int = 42) -> np.ndarray:
    """
    Generate realistic antenna voltage test cases (EEG scale: 1-50 mV).
    
    Reduced to 10 cases since we need to run the FEM solver for each.
    """
    np.random.seed(seed)
    
    V_MIN = 0.001  # 1 mV
    V_MAX = 0.050  # 50 mV
    
    cases = []
    
    # Type 1: Single antenna (2 cases)
    for _ in range(2):
        v = np.zeros(9)
        idx = np.random.randint(0, 9)
        v[idx] = np.random.uniform(V_MIN * 5, V_MAX)
        cases.append(v)
    
    # Type 2: Dipole pairs (3 cases)
    for _ in range(3):
        v = np.zeros(9)
        idx1 = np.random.randint(0, 8)
        idx2 = idx1 + 1
        amp = np.random.uniform(V_MIN * 5, V_MAX)
        v[idx1] = amp
        v[idx2] = -amp
        cases.append(v)
    
    # Type 3: Random uniform (2 cases)
    for _ in range(2):
        v = np.random.uniform(-V_MAX, V_MAX, 9)
        cases.append(v)
    
    # Type 4: Random sparse (remaining)
    remaining = n_cases - len(cases)
    for _ in range(remaining):
        v = np.zeros(9)
        n_active = np.random.randint(2, 5)
        active_idx = np.random.choice(9, n_active, replace=False)
        v[active_idx] = np.random.uniform(-V_MAX, V_MAX, n_active)
        cases.append(v)
    
    return np.array(cases)


# ============================================================================
# Solver Integration
# ============================================================================

def create_simulation_cases_csv(V_antenna_cases: np.ndarray, output_path: Path):
    """Create simulation_cases.csv for the solver."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: case,v_1,v_2,...,v_9
        writer.writerow(['case'] + [f'v_{i}' for i in range(1, 10)])
        
        for case_idx, v_antenna in enumerate(V_antenna_cases):
            case_name = f'test_case_{case_idx:02d}'
            writer.writerow([case_name] + [f'{v:.6e}' for v in v_antenna])


def run_solver(work_dir: Path, mesh_path: Path, cases_csv: Path, probes_csv: Path) -> bool:
    """
    Run the FEM solver (step 3).
    
    Returns True if successful, False otherwise.
    """
    # Copy required files to work directory
    solver_script = Path(__file__).parent.parent / 'steps' / 'step_3' / 'solver.py'
    
    if not solver_script.exists():
        print(f"ERROR: Solver script not found at {solver_script}")
        return False
    
    # Create subdirectory structure expected by solver
    cases_dir = work_dir / 'cases'
    cases_dir.mkdir(exist_ok=True)
    
    # Copy files
    shutil.copy(mesh_path, work_dir / 'mesh.msh')
    shutil.copy(cases_csv, work_dir / 'simulation_cases.csv')
    shutil.copy(probes_csv, work_dir / 'probes.csv')
    
    # Run solver
    print(f"  Running FEM solver...")
    try:
        result = subprocess.run(
            ['python3', str(solver_script)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"  ERROR: Solver failed with return code {result.returncode}")
            print(f"  STDERR: {result.stderr}")
            return False
        
        return True
    
    except subprocess.TimeoutExpired:
        print(f"  ERROR: Solver timeout (>5 minutes)")
        return False
    except Exception as e:
        print(f"  ERROR: Solver execution failed: {e}")
        return False


def sample_probes_from_vtu(vtu_path: Path, probe_coords: np.ndarray) -> np.ndarray:
    """Sample probe voltages from solver VTU output."""
    try:
        pv.OFF_SCREEN = True
    except:
        pass
    
    grid = pv.read(str(vtu_path))
    
    # Extract head surface (region_id=2)
    try:
        head_vol = grid.threshold((1.5, 2.5), scalars='region_id')
        head_surface = head_vol.extract_surface()
    except:
        head_surface = grid.extract_surface()
    
    if 'u' not in head_surface.point_data:
        raise ValueError(f"No 'u' field in {vtu_path.name}")
    
    # Sample at each probe
    values = np.zeros(len(probe_coords))
    for i, coord in enumerate(probe_coords):
        closest_id = head_surface.find_closest_point(coord)
        values[i] = head_surface.point_data['u'][closest_id]
    
    return values


def extract_solver_measurements(work_dir: Path, probe_coords: np.ndarray, n_cases: int) -> np.ndarray:
    """
    Extract probe measurements from solver VTU files.
    
    Returns (n_cases, n_probes) array of voltages.
    """
    cases_dir = work_dir / 'cases'
    n_probes = len(probe_coords)
    measurements = np.zeros((n_cases, n_probes))
    
    for case_idx in range(n_cases):
        case_name = f'test_case_{case_idx:02d}'
        vtu_file = cases_dir / f'solution_{case_name}.vtu'
        
        if not vtu_file.exists():
            raise FileNotFoundError(f"Solver output not found: {vtu_file}")
        
        measurements[case_idx, :] = sample_probes_from_vtu(vtu_file, probe_coords)
    
    return measurements


# ============================================================================
# Dipole Reconstruction (CORRECTED)
# ============================================================================

def reconstruct_from_dipoles(
    F: np.ndarray,
    S: np.ndarray,
    V_antenna: np.ndarray
) -> np.ndarray:
    """
    Reconstruct probe measurements using selection matrix S.
    
    CORRECT FORMULA: V_probes = F @ S @ V_antenna
    
    Parameters
    ----------
    F : (n_probes, 36) - Forward transfer matrix
    S : (36, 9) - Selection matrix (maps antennas to selected dipoles)
    V_antenna : (9,) - Antenna voltages
    
    Returns
    -------
    V_probes : (n_probes,) - Reconstructed probe voltages
    """
    # S @ V_antenna gives dipole activations (36,) with zeros for unselected
    dipole_activations = S @ V_antenna  # (36,)
    
    # F @ dipole_activations gives probe measurements
    V_probes = F @ dipole_activations  # (n_probes,)
    
    return V_probes


# ============================================================================
# Error Analysis
# ============================================================================

def compute_errors(V_solver: np.ndarray, V_reconstructed: np.ndarray) -> Dict[str, float]:
    """Compute error metrics comparing solver vs reconstruction."""
    diff = V_solver - V_reconstructed
    
    rmse = np.sqrt(np.mean(diff**2))
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    
    rms_solver = np.sqrt(np.mean(V_solver**2))
    relative_rmse = rmse / rms_solver if rms_solver > 1e-15 else np.inf
    
    max_solver = np.max(np.abs(V_solver))
    relative_max = max_abs / max_solver if max_solver > 1e-15 else np.inf
    
    return {
        'rmse': rmse,
        'max_abs_error': max_abs,
        'mean_abs_error': mean_abs,
        'relative_rmse': relative_rmse,
        'relative_max': relative_max
    }


def analyze_spatial_errors(probe_names: List[str], errors: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Analyze errors by EEG brain region."""
    region_errors = {}
    
    for region, region_probes in EEG_REGIONS.items():
        region_err_values = []
        for probe in region_probes:
            if probe in probe_names:
                idx = probe_names.index(probe)
                region_err_values.append(errors[idx])
        
        if region_err_values:
            region_errors[region] = {
                'mean_error': np.mean(np.abs(region_err_values)),
                'max_error': np.max(np.abs(region_err_values)),
                'n_probes': len(region_err_values)
            }
    
    return region_errors


# ============================================================================
# Validation Pipeline
# ============================================================================

def validate_selection_with_solver(
    result_name: str,
    metadata: Dict,
    S_matrix: np.ndarray,
    F: np.ndarray,
    probe_names: List[str],
    probe_coords: np.ndarray,
    V_antenna_cases: np.ndarray,
    mesh_path: Path,
    work_dir: Path
) -> Dict[str, Any]:
    """
    Validate one selection against solver ground truth.
    
    Returns validation results dict.
    """
    n_selected = np.sum(np.any(S_matrix != 0, axis=0))
    
    if n_selected == 0:
        return {
            'result_name': result_name,
            'error': 'No dipoles selected',
            'status': 'FAILED'
        }
    
    print(f"    Preparing solver run...")
    
    # Create simulation cases CSV
    cases_csv = work_dir / 'simulation_cases.csv'
    probes_csv = work_dir / 'probes.csv'
    
    create_simulation_cases_csv(V_antenna_cases, cases_csv)
    
    # Copy original probes.csv
    orig_probes = Path('run/probes.csv')
    if orig_probes.exists():
        shutil.copy(orig_probes, probes_csv)
    else:
        # Create from probe_names and coords
        with open(probes_csv, 'w') as f:
            f.write('# name,x,y,z\n')
            for name, coord in zip(probe_names, probe_coords):
                f.write(f'{name},{coord[0]:.6f},{coord[1]:.6f},{coord[2]:.6f}\n')
    
    # Run solver
    solver_success = run_solver(work_dir, mesh_path, cases_csv, probes_csv)
    
    if not solver_success:
        return {
            'result_name': result_name,
            'error': 'Solver failed',
            'status': 'FAILED'
        }
    
    print(f"    Extracting solver measurements...")
    
    # Extract solver measurements
    try:
        solver_measurements = extract_solver_measurements(work_dir, probe_coords, len(V_antenna_cases))
    except Exception as e:
        return {
            'result_name': result_name,
            'error': f'Failed to extract measurements: {e}',
            'status': 'FAILED'
        }
    
    print(f"    Computing dipole reconstructions...")
    
    # Compute reconstructions for each case
    case_results = []
    
    for case_idx, V_antenna in enumerate(V_antenna_cases):
        V_solver = solver_measurements[case_idx, :]
        V_recon = reconstruct_from_dipoles(F, S_matrix, V_antenna)
        
        errors_dict = compute_errors(V_solver, V_recon)
        probe_errors = V_solver - V_recon
        spatial_errors = analyze_spatial_errors(probe_names, probe_errors)
        
        case_results.append({
            'case_idx': case_idx,
            'V_antenna': V_antenna.tolist(),
            'V_solver': V_solver.tolist(),
            'V_reconstructed': V_recon.tolist(),
            'probe_errors': probe_errors.tolist(),
            'errors': errors_dict,
            'spatial_errors': spatial_errors
        })
    
    # Aggregate statistics
    all_rmse = [c['errors']['rmse'] for c in case_results]
    all_max_err = [c['errors']['max_abs_error'] for c in case_results]
    all_rel_rmse = [c['errors']['relative_rmse'] for c in case_results]
    
    return {
        'result_name': result_name,
        'algorithm': metadata.get('algorithm', 'unknown'),
        'n_selected': int(n_selected),
        'condition_number': metadata.get('condition_number', np.nan),
        'parameters': metadata.get('parameters', {}),
        'status': 'SUCCESS',
        'aggregate_metrics': {
            'rmse_mean': float(np.mean(all_rmse)),
            'rmse_std': float(np.std(all_rmse)),
            'rmse_min': float(np.min(all_rmse)),
            'rmse_max': float(np.max(all_rmse)),
            'max_error_mean': float(np.mean(all_max_err)),
            'max_error_max': float(np.max(all_max_err)),
            'relative_rmse_mean': float(np.mean(all_rel_rmse)),
            'relative_rmse_std': float(np.std(all_rel_rmse))
        },
        'case_results': case_results
    }


def load_top_results(results_dir: Path, n_per_algo: int = 2) -> List[Tuple[str, Dict, np.ndarray]]:
    """Load top N results per algorithm (reduced to 2 since solver is slow)."""
    all_results = []
    
    for algo_dir in sorted(results_dir.glob('algorithm_*')):
        if not algo_dir.is_dir():
            continue
        
        algo_results = []
        
        for npy_file in sorted(algo_dir.glob('S_*.npy')):
            metadata_file = npy_file.with_name(npy_file.stem + '_metadata.json')
            
            if not metadata_file.exists():
                continue
            
            try:
                S = np.load(npy_file)
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                kappa = metadata.get('condition_number', np.inf)
                if isinstance(kappa, str):
                    if 'inf' in kappa.lower():
                        kappa = np.inf
                    else:
                        kappa = float(kappa)
                
                if not np.isfinite(kappa):
                    continue
                
                result_name = f"{algo_dir.name}/{npy_file.stem}"
                algo_results.append((kappa, result_name, metadata, S))
            
            except Exception as e:
                print(f"WARNING: Failed to load {npy_file.name}: {e}")
                continue
        
        algo_results.sort(key=lambda x: x[0])
        top_n = algo_results[:n_per_algo]
        
        for _, result_name, metadata, S in top_n:
            all_results.append((result_name, metadata, S))
    
    return all_results


# ============================================================================
# Report Generation (similar to before but with "Solver" instead of "Measured")
# ============================================================================

def generate_report(validation_results: List[Dict], output_dir: Path, probe_names: List[str]):
    """Generate validation report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary CSV
    summary_path = output_dir / 'solver_validation_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Rank', 'Algorithm', 'Result Name', 'N Dipoles', 'Condition Number',
            'RMSE Mean (μV)', 'RMSE Std (μV)', 'Max Error Mean (μV)',
            'Relative RMSE (%)', 'Status'
        ])
        
        for rank, result in enumerate(validation_results, 1):
            if result.get('status') != 'SUCCESS':
                writer.writerow([
                    rank, result.get('algorithm', 'unknown'),
                    result['result_name'], 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A',
                    result.get('status', 'UNKNOWN')
                ])
                continue
            
            agg = result['aggregate_metrics']
            writer.writerow([
                rank,
                result['algorithm'],
                result['result_name'],
                result['n_selected'],
                f"{result['condition_number']:.2f}",
                f"{agg['rmse_mean']*1e6:.2f}",
                f"{agg['rmse_std']*1e6:.2f}",
                f"{agg['max_error_mean']*1e6:.2f}",
                f"{agg['relative_rmse_mean']*100:.2f}",
                result['status']
            ])
    
    print(f"✓ Saved: {summary_path}")
    
    # Detailed reports (similar structure)
    for result in validation_results:
        if result.get('status') != 'SUCCESS':
            continue
        
        safe_name = result['result_name'].replace('/', '_')
        detail_path = output_dir / f'solver_detail_{safe_name}.csv'
        
        with open(detail_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"=== SOLVER VALIDATION: {result['result_name']} ==="])
            writer.writerow([f"Algorithm: {result['algorithm']}"])
            writer.writerow([f"Dipoles: {result['n_selected']}, κ: {result['condition_number']:.4f}"])
            writer.writerow([])
            
            for case_res in result['case_results']:
                case_idx = case_res['case_idx']
                writer.writerow([f"--- Test Case {case_idx} ---"])
                
                v_ant = np.array(case_res['V_antenna'])
                writer.writerow(['Antenna (mV):', ', '.join(f"{v*1e3:.2f}" for v in v_ant)])
                writer.writerow([])
                
                writer.writerow(['Probe', 'Region', 'Solver (μV)', 'Dipoles (μV)', 'Error (μV)', '|Error| (μV)'])
                
                V_solver = np.array(case_res['V_solver'])
                V_recon = np.array(case_res['V_reconstructed'])
                errs = np.array(case_res['probe_errors'])
                
                for i, probe in enumerate(probe_names):
                    region = PROBE_TO_REGION.get(probe, 'unknown')
                    writer.writerow([
                        probe, region,
                        f"{V_solver[i]*1e6:.4f}",
                        f"{V_recon[i]*1e6:.4f}",
                        f"{errs[i]*1e6:.4f}",
                        f"{abs(errs[i])*1e6:.4f}"
                    ])
                
                writer.writerow([])
        
        print(f"✓ Saved: {detail_path}")
    
    # JSON export
    json_path = output_dir / 'solver_validation_results.json'
    json_results = []
    for result in validation_results:
        if result.get('status') == 'SUCCESS':
            json_results.append({
                'result_name': result['result_name'],
                'algorithm': result['algorithm'],
                'n_selected': result['n_selected'],
                'condition_number': result['condition_number'],
                'aggregate_metrics': result['aggregate_metrics']
            })
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✓ Saved: {json_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate dipole selections against FEM solver (MANDATORY VALIDATION)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-dir', type=Path, default=Path('results'),
                       help='Algorithm results directory')
    parser.add_argument('--base-matrices', type=Path, default=Path('src/model/base_matrices'),
                       help='F, B, W matrices directory')
    parser.add_argument('--mesh', type=Path, default=Path('run/mesh.msh'),
                       help='Mesh file for solver')
    parser.add_argument('--probes-csv', type=Path, default=Path('run/probes.csv'),
                       help='Probe coordinates')
    parser.add_argument('--output', type=Path, default=Path('solver_validation_reports'),
                       help='Output directory')
    parser.add_argument('--n-per-algo', type=int, default=2,
                       help='Top N results per algorithm (solver is slow!)')
    parser.add_argument('--n-test-cases', type=int, default=10,
                       help='Number of test cases (solver is slow!)')
    parser.add_argument('--work-dir', type=Path, default=None,
                       help='Work directory for solver (default: temp)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SOLVER-BASED VALIDATION (MANDATORY)")
    print("="*80)
    print(f"This validates reconstruction against REAL FEM solver results!")
    print(f"Results dir: {args.results_dir}")
    print(f"Mesh: {args.mesh}")
    print(f"Test cases: {args.n_test_cases} (×{args.n_per_algo} per algorithm)")
    print()
    
    # Load matrices
    print("Loading base matrices...")
    matrices = load_base_matrices(args.base_matrices)
    F = matrices['F']
    print(f"  F: {F.shape}")
    
    # Load probes
    print("Loading probes...")
    probe_names, probe_coords = load_probe_coords(args.probes_csv)
    print(f"  {len(probe_names)} probes")
    
    # Generate test cases
    print(f"\nGenerating {args.n_test_cases} test cases...")
    V_antenna_cases = generate_antenna_test_cases(args.n_test_cases)
    print(f"  Range: [{np.min(V_antenna_cases)*1e3:.2f}, {np.max(V_antenna_cases)*1e3:.2f}] mV")
    
    # Load results
    print(f"\nLoading top {args.n_per_algo} per algorithm...")
    results_to_test = load_top_results(args.results_dir, args.n_per_algo)
    print(f"  {len(results_to_test)} results to validate")
    
    # Validate
    print("\nRunning solver-based validation...")
    print("  WARNING: This will take time (FEM solver runs for each result)")
    
    validation_results = []
    
    for idx, (result_name, metadata, S_matrix) in enumerate(results_to_test, 1):
        print(f"\n  [{idx}/{len(results_to_test)}] {result_name}")
        
        # Create temporary work directory
        if args.work_dir:
            work_dir = args.work_dir / f'work_{idx}'
            work_dir.mkdir(parents=True, exist_ok=True)
        else:
            work_dir = Path(tempfile.mkdtemp(prefix='solver_validation_'))
        
        try:
            result = validate_selection_with_solver(
                result_name, metadata, S_matrix, F,
                probe_names, probe_coords, V_antenna_cases,
                args.mesh, work_dir
            )
            
            validation_results.append(result)
            
            if result['status'] == 'SUCCESS':
                agg = result['aggregate_metrics']
                print(f"    ✓ RMSE={agg['rmse_mean']*1e6:.2f}±{agg['rmse_std']*1e6:.2f} μV")
            else:
                print(f"    ✗ {result.get('error', 'FAILED')}")
        
        finally:
            # Cleanup temp directory
            if not args.work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
    
    # Generate reports
    print(f"\nGenerating reports...")
    generate_report(validation_results, args.output, probe_names)
    
    print("\n" + "="*80)
    print("SOLVER VALIDATION COMPLETE")
    print("="*80)
    print(f"Reports: {args.output}")


if __name__ == '__main__':
    main()
