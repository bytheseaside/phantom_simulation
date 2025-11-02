#!/usr/bin/env python3
"""
validate_selections.py

Validate dipole selection results by testing reconstruction accuracy with random antenna configurations.

This script:
1. Loads top-ranked selection results from each algorithm
2. Creates realistic test cases with random antenna voltages
3. Computes "ground truth" probe measurements using full forward model
4. Reconstructs probe measurements using only selected dipoles
5. Analyzes reconstruction errors and identifies spatial patterns

EEG Context:
- Typical scalp EEG amplitudes: 10-100 μV (microvolts)
- To achieve this, antenna voltages are set in millivolt range (1-50 mV)
- This produces realistic probe measurements similar to actual EEG recordings
"""

import sys
from pathlib import Path
import numpy as np
import json
import csv
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from selection_algorithms.common import load_base_matrices


# ============================================================================
# EEG Probe Naming and Spatial Analysis
# ============================================================================

# International 10-20 EEG system probe grouping
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

# Build reverse lookup: probe_name -> region_name
PROBE_TO_REGION = {}
for region, probes in EEG_REGIONS.items():
    for probe in probes:
        PROBE_TO_REGION[probe] = region


def load_probe_names(probes_csv: Path) -> List[str]:
    """Load probe names from CSV file."""
    names = []
    with open(probes_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not row[0].startswith('#'):
                names.append(row[0])
    return names


# ============================================================================
# Test Case Generation
# ============================================================================

def generate_antenna_test_cases(n_cases: int = 15, seed: int = 42) -> np.ndarray:
    """
    Generate realistic antenna voltage test cases.
    
    EEG scalp measurements are typically 10-100 μV.
    To achieve this range, antenna voltages are set to millivolt scale (1-50 mV).
    
    Test case variety:
    - Single antenna dominant (mimics focal source)
    - Dipole pairs (two antennas, opposite polarity)
    - Random uniform (distributed activation)
    - Random sparse (few active antennas)
    
    Parameters
    ----------
    n_cases : int
        Number of test cases to generate
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    V_antenna_cases : (n_cases, 9) array
        Each row is one antenna voltage configuration (in Volts)
    """
    np.random.seed(seed)
    
    # Target: estimated input millivolt range for EEG-scale outputs
    V_MIN = 0.001  # 1 mV
    V_MAX = 0.050  # 50 mV
    
    cases = []
    
    # Type 1: Single dominant antenna (focal source) - 4 cases
    for _ in range(4):
        v = np.zeros(9)
        idx = np.random.randint(0, 9)
        v[idx] = np.random.uniform(V_MIN * 5, V_MAX)  # 5-50 mV
        cases.append(v)
    
    # Type 2: Dipole pairs (adjacent antennas, opposite polarity) - 4 cases
    for _ in range(4):
        v = np.zeros(9)
        idx1 = np.random.randint(0, 8)
        idx2 = idx1 + 1
        amp = np.random.uniform(V_MIN * 5, V_MAX)
        v[idx1] = amp
        v[idx2] = -amp
        cases.append(v)
    
    # Type 3: Random uniform (all antennas active) - 3 cases
    for _ in range(3):
        v = np.random.uniform(-V_MAX, V_MAX, 9)
        cases.append(v)
    
    # Type 4: Random sparse (only 2-4 antennas active) - remaining cases
    remaining = n_cases - len(cases)
    for _ in range(remaining):
        v = np.zeros(9)
        n_active = np.random.randint(2, 5)
        active_idx = np.random.choice(9, n_active, replace=False)
        v[active_idx] = np.random.uniform(-V_MAX, V_MAX, n_active)
        cases.append(v)
    
    return np.array(cases)


# ============================================================================
# Forward Model and Reconstruction
# ============================================================================

def compute_ground_truth_probes(F: np.ndarray, B: np.ndarray, V_antenna: np.ndarray) -> np.ndarray:
    """
    Compute ground truth probe measurements using full forward model.
    
    V_probes = F @ B @ V_antenna
    
    Parameters
    ----------
    F : (n_probes, 36) array - Forward transfer matrix
    B : (36, 9) array - Dipole to antenna matrix
    V_antenna : (9,) array - Antenna voltages
    
    Returns
    -------
    V_probes : (n_probes,) array - Probe voltages
    """
    # Full system: all 36 dipoles contribute
    dipole_activations = B @ V_antenna  # (36,) - each dipole's amplitude
    V_probes = F @ dipole_activations   # (n_probes,) - probe measurements
    return V_probes


def reconstruct_probes_from_selection(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    selected_indices: List[int],
    V_antenna: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct probe measurements using only selected dipoles.
    
    Process:
    1. Extract F_selected and B_selected for chosen dipoles
    2. Compute dipole activations: x_selected = B_selected @ V_antenna
    3. Reconstruct probes: V_probes_reconstructed = F_selected @ x_selected
    
    Parameters
    ----------
    F : (n_probes, 36) array
    B : (36, 9) array
    W : (n_probes, n_probes) array - Probe weights (diagonal)
    selected_indices : list of int - Which dipoles are selected
    V_antenna : (9,) array
    
    Returns
    -------
    V_probes_reconstructed : (n_probes,) array
    dipole_activations : (n_selected,) array - Computed dipole amplitudes
    """
    # Extract selected columns
    F_selected = F[:, selected_indices]  # (n_probes, n_selected)
    B_selected = B[selected_indices, :]  # (n_selected, 9)
    
    # Compute dipole activations for selected dipoles
    dipole_activations = B_selected @ V_antenna  # (n_selected,)
    
    # Reconstruct probe voltages
    V_probes_reconstructed = F_selected @ dipole_activations  # (n_probes,)
    
    return V_probes_reconstructed, dipole_activations


# ============================================================================
# Error Analysis
# ============================================================================

def compute_errors(V_true: np.ndarray, V_reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Compute various error metrics.
    
    Returns
    -------
    dict with keys:
        - rmse: Root mean square error (V)
        - max_abs_error: Maximum absolute error (V)
        - mean_abs_error: Mean absolute error (V)
        - relative_rmse: RMSE / RMS(V_true) (dimensionless)
        - relative_max: Max error / Max(|V_true|) (dimensionless)
    """
    diff = V_true - V_reconstructed
    
    rmse = np.sqrt(np.mean(diff**2))
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    
    rms_true = np.sqrt(np.mean(V_true**2))
    relative_rmse = rmse / rms_true if rms_true > 1e-15 else np.inf
    
    max_true = np.max(np.abs(V_true))
    relative_max = max_abs / max_true if max_true > 1e-15 else np.inf
    
    return {
        'rmse': rmse,
        'max_abs_error': max_abs,
        'mean_abs_error': mean_abs,
        'relative_rmse': relative_rmse,
        'relative_max': relative_max
    }


def analyze_spatial_errors(
    probe_names: List[str],
    errors: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Analyze errors by EEG brain region.
    
    Returns
    -------
    dict mapping region_name to:
        - mean_error: Mean absolute error in region (V)
        - max_error: Maximum error in region (V)
        - n_probes: Number of probes in region
    """
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

def validate_single_selection(
    result_name: str,
    metadata: Dict,
    S_matrix: np.ndarray,
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    probe_names: List[str],
    V_antenna_cases: np.ndarray
) -> Dict[str, Any]:
    """
    Validate one selection result across all test cases.
    
    Returns
    -------
    dict with validation results including per-case and aggregate metrics
    """
    # Extract selected dipoles
    selected_indices = np.where(np.any(S_matrix != 0, axis=0))[0].tolist()
    n_selected = len(selected_indices)
    
    if n_selected == 0:
        return {
            'result_name': result_name,
            'error': 'No dipoles selected',
            'status': 'FAILED'
        }
    
    # Run all test cases
    case_results = []
    
    for case_idx, V_antenna in enumerate(V_antenna_cases):
        # Ground truth
        V_true = compute_ground_truth_probes(F, B, V_antenna)
        
        # Reconstruction
        V_recon, dipole_amps = reconstruct_probes_from_selection(
            F, B, W, selected_indices, V_antenna
        )
        
        # Errors
        errors_dict = compute_errors(V_true, V_recon)
        
        # Per-probe errors
        probe_errors = V_true - V_recon
        
        # Spatial analysis
        spatial_errors = analyze_spatial_errors(probe_names, probe_errors)
        
        case_results.append({
            'case_idx': case_idx,
            'V_antenna': V_antenna.tolist(),
            'V_true': V_true.tolist(),
            'V_reconstructed': V_recon.tolist(),
            'probe_errors': probe_errors.tolist(),
            'errors': errors_dict,
            'spatial_errors': spatial_errors
        })
    
    # Aggregate statistics across all cases
    all_rmse = [c['errors']['rmse'] for c in case_results]
    all_max_err = [c['errors']['max_abs_error'] for c in case_results]
    all_rel_rmse = [c['errors']['relative_rmse'] for c in case_results]
    
    return {
        'result_name': result_name,
        'algorithm': metadata.get('algorithm', 'unknown'),
        'n_selected': n_selected,
        'condition_number': metadata.get('condition_number', np.nan),
        'parameters': metadata.get('parameters', {}),
        'status': 'SUCCESS',
        'aggregate_metrics': {
            'rmse_mean': np.mean(all_rmse),
            'rmse_std': np.std(all_rmse),
            'rmse_min': np.min(all_rmse),
            'rmse_max': np.max(all_rmse),
            'max_error_mean': np.mean(all_max_err),
            'max_error_max': np.max(all_max_err),
            'relative_rmse_mean': np.mean(all_rel_rmse),
            'relative_rmse_std': np.std(all_rel_rmse)
        },
        'case_results': case_results
    }


def load_top_results_per_algorithm(
    results_dir: Path,
    n_per_algorithm: int = 3
) -> List[Tuple[str, Dict, np.ndarray]]:
    """
    Load top N results from each algorithm directory.
    
    Returns
    -------
    list of (result_name, metadata, S_matrix) tuples
    """
    all_results = []
    
    for algo_dir in sorted(results_dir.glob('algorithm_*')):
        if not algo_dir.is_dir():
            continue
        
        algo_results = []
        
        # Find all results in this algorithm
        for npy_file in sorted(algo_dir.glob('S_*.npy')):
            metadata_file = npy_file.with_name(npy_file.stem + '_metadata.json')
            
            if not metadata_file.exists():
                continue
            
            try:
                S = np.load(npy_file)
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Score by condition number (lower is better)
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
        
        # Sort by condition number and take top N
        algo_results.sort(key=lambda x: x[0])
        top_n = algo_results[:n_per_algorithm]
        
        for _, result_name, metadata, S in top_n:
            all_results.append((result_name, metadata, S))
    
    return all_results


# ============================================================================
# Report Generation
# ============================================================================

def generate_validation_report(
    validation_results: List[Dict],
    output_dir: Path,
    probe_names: List[str]
):
    """Generate comprehensive validation report with tables and analysis."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === Summary Table ===
    summary_path = output_dir / 'validation_summary.csv'
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
                f"{agg['rmse_mean']*1e6:.2f}",  # Convert to μV
                f"{agg['rmse_std']*1e6:.2f}",
                f"{agg['max_error_mean']*1e6:.2f}",
                f"{agg['relative_rmse_mean']*100:.2f}",
                result['status']
            ])
    
    print(f"✓ Saved summary: {summary_path}")
    
    # === Per-Result Detailed Reports ===
    for result in validation_results:
        if result.get('status') != 'SUCCESS':
            continue
        
        safe_name = result['result_name'].replace('/', '_')
        detail_path = output_dir / f'detail_{safe_name}.csv'
        
        with open(detail_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([f"=== Validation Report: {result['result_name']} ==="])
            writer.writerow([f"Algorithm: {result['algorithm']}"])
            writer.writerow([f"Dipoles Selected: {result['n_selected']}"])
            writer.writerow([f"Condition Number: {result['condition_number']:.4f}"])
            writer.writerow([])
            
            # Aggregate metrics
            writer.writerow(['Aggregate Metrics (across all test cases)'])
            agg = result['aggregate_metrics']
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['RMSE Mean (μV)', f"{agg['rmse_mean']*1e6:.4f}"])
            writer.writerow(['RMSE Std (μV)', f"{agg['rmse_std']*1e6:.4f}"])
            writer.writerow(['RMSE Range (μV)', f"[{agg['rmse_min']*1e6:.4f}, {agg['rmse_max']*1e6:.4f}]"])
            writer.writerow(['Max Error Mean (μV)', f"{agg['max_error_mean']*1e6:.4f}"])
            writer.writerow(['Relative RMSE (%)', f"{agg['relative_rmse_mean']*100:.4f}"])
            writer.writerow([])
            
            # Per-case tables
            for case_res in result['case_results']:
                case_idx = case_res['case_idx']
                writer.writerow([f"--- Test Case {case_idx} ---"])
                
                # Antenna voltages
                v_ant = np.array(case_res['V_antenna'])
                writer.writerow(['Antenna Voltages (mV):', ', '.join(f"{v*1e3:.2f}" for v in v_ant)])
                writer.writerow([])
                
                # Per-probe table
                writer.writerow(['Probe', 'Region', 'Measured (μV)', 'Reconstructed (μV)', 'Error (μV)', '|Error| (μV)'])
                
                V_true = np.array(case_res['V_true'])
                V_recon = np.array(case_res['V_reconstructed'])
                probe_errs = np.array(case_res['probe_errors'])
                
                for i, probe in enumerate(probe_names):
                    region = PROBE_TO_REGION.get(probe, 'unknown')
                    writer.writerow([
                        probe,
                        region,
                        f"{V_true[i]*1e6:.4f}",
                        f"{V_recon[i]*1e6:.4f}",
                        f"{probe_errs[i]*1e6:.4f}",
                        f"{abs(probe_errs[i])*1e6:.4f}"
                    ])
                
                writer.writerow([])
                
                # Spatial error summary
                writer.writerow(['Regional Error Summary'])
                writer.writerow(['Region', 'Mean |Error| (μV)', 'Max |Error| (μV)', 'N Probes'])
                
                spatial = case_res['spatial_errors']
                for region, stats in sorted(spatial.items()):
                    writer.writerow([
                        region,
                        f"{stats['mean_error']*1e6:.4f}",
                        f"{stats['max_error']*1e6:.4f}",
                        stats['n_probes']
                    ])
                
                writer.writerow([])
                writer.writerow([])
        
        print(f"✓ Saved detail: {detail_path}")
    
    # === Spatial Analysis Aggregate ===
    spatial_path = output_dir / 'spatial_analysis.csv'
    with open(spatial_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Result', 'Region', 'Mean Error (μV)', 'Max Error (μV)', 'Frequency'])
        
        for result in validation_results:
            if result.get('status') != 'SUCCESS':
                continue
            
            # Aggregate spatial errors across all cases
            region_accumulator = {}
            
            for case_res in result['case_results']:
                for region, stats in case_res['spatial_errors'].items():
                    if region not in region_accumulator:
                        region_accumulator[region] = []
                    region_accumulator[region].append(stats['mean_error'])
            
            for region, errors in region_accumulator.items():
                writer.writerow([
                    result['algorithm'],
                    result['result_name'],
                    region,
                    f"{np.mean(errors)*1e6:.4f}",
                    f"{np.max(errors)*1e6:.4f}",
                    len(errors)
                ])
    
    print(f"✓ Saved spatial analysis: {spatial_path}")
    
    # === JSON Export (for programmatic access) ===
    json_path = output_dir / 'validation_results.json'
    # Create serializable version (convert numpy arrays)
    json_results = []
    for result in validation_results:
        if result.get('status') == 'SUCCESS':
            # Remove case_results (too large) but keep aggregate
            json_result = {
                'result_name': result['result_name'],
                'algorithm': result['algorithm'],
                'n_selected': result['n_selected'],
                'condition_number': result['condition_number'],
                'parameters': result['parameters'],
                'aggregate_metrics': result['aggregate_metrics']
            }
            json_results.append(json_result)
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✓ Saved JSON: {json_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate dipole selection results with reconstruction tests',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-dir', type=Path, default=Path('results'),
                       help='Directory containing algorithm results')
    parser.add_argument('--base-matrices', type=Path, default=Path('run/f_matrix'),
                       help='Directory with F, B, W matrices')
    parser.add_argument('--probes-csv', type=Path, default=Path('run/probes.csv'),
                       help='Probe coordinates CSV')
    parser.add_argument('--output', type=Path, default=Path('validation_reports'),
                       help='Output directory for reports')
    parser.add_argument('--n-per-algo', type=int, default=3,
                       help='Number of top results to test per algorithm')
    parser.add_argument('--n-test-cases', type=int, default=15,
                       help='Number of random antenna test cases')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DIPOLE SELECTION VALIDATION")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Base matrices: {args.base_matrices}")
    print(f"Top results per algorithm: {args.n_per_algo}")
    print(f"Test cases: {args.n_test_cases}")
    print()
    
    # Load base matrices
    print("Loading base matrices...")
    matrices = load_base_matrices(args.base_matrices)
    F = matrices['F']  # (21, 36)
    B = matrices['B']  # (36, 9)
    W = matrices['W']  # (21, 21)
    print(f"  F: {F.shape}, B: {B.shape}, W: {W.shape}")
    
    # Load probe names
    print("Loading probe names...")
    probe_names = load_probe_names(args.probes_csv)
    print(f"  {len(probe_names)} probes")
    
    # Generate test cases
    print(f"\nGenerating {args.n_test_cases} test cases...")
    V_antenna_cases = generate_antenna_test_cases(args.n_test_cases, args.seed)
    print(f"  Antenna voltage range: [{np.min(V_antenna_cases)*1e3:.2f}, {np.max(V_antenna_cases)*1e3:.2f}] mV")
    
    # Load top results
    print(f"\nLoading top {args.n_per_algo} results per algorithm...")
    results_to_test = load_top_results_per_algorithm(args.results_dir, args.n_per_algo)
    print(f"  Total results to validate: {len(results_to_test)}")
    
    # Validate each result
    print("\nRunning validation...")
    validation_results = []
    
    for idx, (result_name, metadata, S_matrix) in enumerate(results_to_test, 1):
        print(f"  [{idx}/{len(results_to_test)}] {result_name}...", end=' ')
        
        result = validate_single_selection(
            result_name, metadata, S_matrix,
            F, B, W, probe_names, V_antenna_cases
        )
        
        validation_results.append(result)
        
        if result['status'] == 'SUCCESS':
            agg = result['aggregate_metrics']
            print(f"✓ RMSE={agg['rmse_mean']*1e6:.2f}±{agg['rmse_std']*1e6:.2f} μV")
        else:
            print(f"✗ {result.get('error', 'FAILED')}")
    
    # Generate reports
    print(f"\nGenerating reports in {args.output}...")
    generate_validation_report(validation_results, args.output, probe_names)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Reports saved to: {args.output}")
    print(f"  - validation_summary.csv: Overview of all results")
    print(f"  - detail_*.csv: Per-result detailed analysis")
    print(f"  - spatial_analysis.csv: Regional error patterns")
    print(f"  - validation_results.json: Machine-readable results")


if __name__ == '__main__':
    main()
