"""
comparison.py

Error computation and comparison logic for validation.
Compares three methods: SOLVER (ground truth), F_FULL, F_SUBSET
"""

import numpy as np
from typing import Dict, List, Tuple
import csv
from pathlib import Path


# EEG 10-20 regional grouping
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

# Reverse lookup
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


def load_solver_probes(probe_csv: Path) -> np.ndarray:
    """
    Load probe measurements from solver output CSV.
    
    Expected format: probe_name,value
    
    Returns (n_probes,) array
    """
    values = []
    with open(probe_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) >= 2:
                values.append(float(row[1]))
    return np.array(values)


def compute_full_forward(F: np.ndarray, B: np.ndarray, V_antenna: np.ndarray) -> np.ndarray:
    """
    Compute full forward model: V_probes = F @ B @ V_antenna
    
    Uses all 36 dipoles.
    
    Parameters
    ----------
    F : (n_probes, 36) array
    B : (36, 9) array
    V_antenna : (9,) array
    
    Returns
    -------
    V_probes : (n_probes,) array
    """
    return F @ B @ V_antenna


def compute_subset_forward(F: np.ndarray, S: np.ndarray, V_antenna: np.ndarray) -> np.ndarray:
    """
    Compute subset forward model: V_probes = F @ S @ V_antenna
    
    Uses only selected dipoles encoded in S matrix.
    
    Parameters
    ----------
    F : (n_probes, 36) array
    S : (36, 9) array - Selection matrix
    V_antenna : (9,) array
    
    Returns
    -------
    V_probes : (n_probes,) array
    """
    return F @ S @ V_antenna


def compute_errors(V_ref: np.ndarray, V_test: np.ndarray) -> Dict[str, float]:
    """
    Compute error metrics between reference and test measurements.
    
    Parameters
    ----------
    V_ref : (n_probes,) array - Reference (ground truth)
    V_test : (n_probes,) array - Test measurements
    
    Returns
    -------
    dict with:
        rmse : Root mean square error (V)
        max_abs_error : Maximum absolute error (V)
        mean_abs_error : Mean absolute error (V)
        relative_rmse : RMSE / RMS(V_ref) (dimensionless, %)
    """
    diff = V_ref - V_test
    
    rmse = np.sqrt(np.mean(diff**2))
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    
    rms_ref = np.sqrt(np.mean(V_ref**2))
    relative_rmse = (rmse / rms_ref * 100) if rms_ref > 1e-15 else np.inf
    
    return {
        'rmse': rmse,
        'max_abs_error': max_abs,
        'mean_abs_error': mean_abs,
        'relative_rmse': relative_rmse
    }


def compute_regional_errors(
    probe_names: List[str],
    errors: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Compute error statistics by brain region.
    
    Parameters
    ----------
    probe_names : list of str
    errors : (n_probes,) array - Error per probe
    
    Returns
    -------
    dict mapping region_name -> {mean_error, max_error, n_probes}
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


def compare_all_methods(
    V_solver: np.ndarray,
    V_full: np.ndarray,
    V_subset: np.ndarray,
    probe_names: List[str]
) -> Dict:
    """
    Compare all three methods and compute comprehensive metrics.
    
    Parameters
    ----------
    V_solver : (n_probes,) - Ground truth from FEM solver
    V_full : (n_probes,) - Full forward model (all 36 dipoles)
    V_subset : (n_probes,) - Subset model (selected dipoles only)
    probe_names : list of str
    
    Returns
    -------
    dict with:
        solver_vs_full : error metrics
        solver_vs_subset : error metrics (MOST IMPORTANT)
        full_vs_subset : error metrics (reference)
        regional_solver_vs_full : regional breakdown
        regional_solver_vs_subset : regional breakdown
        per_probe : detailed per-probe data
    """
    # Compute pairwise errors
    err_solver_full = compute_errors(V_solver, V_full)
    err_solver_subset = compute_errors(V_solver, V_subset)
    err_full_subset = compute_errors(V_full, V_subset)
    
    # Per-probe errors
    diff_solver_full = V_solver - V_full
    diff_solver_subset = V_solver - V_subset
    diff_full_subset = V_full - V_subset
    
    # Regional analysis
    regional_solver_full = compute_regional_errors(probe_names, diff_solver_full)
    regional_solver_subset = compute_regional_errors(probe_names, diff_solver_subset)
    
    # Per-probe detailed data
    per_probe = []
    for i, probe_name in enumerate(probe_names):
        region = PROBE_TO_REGION.get(probe_name, 'unknown')
        per_probe.append({
            'probe': probe_name,
            'region': region,
            'V_solver': V_solver[i],
            'V_full': V_full[i],
            'V_subset': V_subset[i],
            'error_solver_full': diff_solver_full[i],
            'error_solver_subset': diff_solver_subset[i],
            'error_full_subset': diff_full_subset[i]
        })
    
    return {
        'solver_vs_full': err_solver_full,
        'solver_vs_subset': err_solver_subset,
        'full_vs_subset': err_full_subset,
        'regional_solver_vs_full': regional_solver_full,
        'regional_solver_vs_subset': regional_solver_subset,
        'per_probe': per_probe
    }
