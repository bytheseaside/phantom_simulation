"""
Common utilities for dipole selection algorithms.
"""

import numpy as np
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any
import json
import sys

# Import from existing utils
sys.path.append(str(Path(__file__).parent.parent))
from model.utils import build_dipoles
from model.base_matrices.generate_base_matrices import build_s_matrix


def validate_matrix_shapes(F: np.ndarray, B: np.ndarray, W: np.ndarray = None) -> Tuple[int, int, int]:
    """
    Validate matrix shapes for dipole selection algorithms.
    
    Checks that matrix dimensions are compatible for multiplication:
    - F @ B requires F columns = B rows
    - W @ F requires W square with size = F rows
    
    Parameters
    ----------
    F : np.ndarray
        Dipole-to-scalp transfer matrix
    B : np.ndarray
        Antenna-to-dipole matrix
    W : np.ndarray, optional
        Probe weighting matrix
        
    Returns
    -------
    n_probes : int
        Number of probes (F rows)
    n_dipoles : int
        Number of dipoles (F columns)
    n_antennas : int
        Number of antennas (B columns)
        
    Raises
    ------
    ValueError
        If matrix dimensions are incompatible
    """
    f_rows, f_cols = F.shape  # probes × dipoles
    b_rows, b_cols = B.shape  # dipoles × antennas
    
    # Check F @ B compatibility
    if f_cols != b_rows:
        raise ValueError(
            f"Matrix multiplication F @ B requires F columns ({f_cols}) "
            f"to match B rows ({b_rows})"
        )
    
    # Check W @ F compatibility
    if W is not None:
        w_rows, w_cols = W.shape
        if w_rows != f_rows or w_cols != f_rows:
            raise ValueError(
                f"W must be square ({f_rows}×{f_rows}) to match F rows, "
                f"got ({w_rows}×{w_cols})"
            )
    
    return f_rows, f_cols, b_cols


def load_forbidden_triads(triads_path: Path) -> List[Set[Tuple[int, int]]]:
    """
    Load forbidden triads from NPY file once at start.
    Returns list of sets for efficient subset checking.
    """
    triads_raw = np.load(triads_path, allow_pickle=True)
    return [{tuple(pair) for pair in triad} for triad in triads_raw]


def check_triad_violation(
    dipole_list: List[Tuple[int, int]], 
    forbidden_triads: List[Set[Tuple[int, int]]]
) -> bool:
    """
    Check if dipole_list contains any forbidden triad.
    Use this by calling with selected + [candidate].
    """
    current = set(dipole_list)
    for triad in forbidden_triads:
        if triad.issubset(current):
            return True
    return False


def compute_condition_number(
    F: np.ndarray,
    B: np.ndarray,
    W: np.ndarray,
    selected_indices: List[int]
) -> float:
    """
    Compute condition number κ(M) where M = W·F_selected·B_selected.

    Uses ONLY the selected columns of F (not the full S matrix with zeros to avoid singularities).

    Parameters
    ----------
    F : (21, 36) array
    B : (36, 9) array
    W : (21, 21) array or None
    selected_indices : list of column indices in F
    
    Returns
    -------
    float : condition number (np.inf if singular/empty)
    """
    if len(selected_indices) == 0:
        return np.inf
    
    # Extract only selected columns of F
    F_selected = F[:, selected_indices]
    
    # Compute M = W·F_selected·B
    M = F_selected @ B[selected_indices, :]
    if W is not None:
        M = W @ M
    
    return np.linalg.cond(M)


def save_selection_results(
    S: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
    filename: str
) -> None:
    """
    Save selection matrix S and metadata.
    Only saves if condition_number is finite.
    
    Parameters
    ----------
    S : np.ndarray
        Selection matrix
    metadata : dict
        Algorithm metadata
    output_dir : Path
        Output directory
    filename : str
        Base filename (without extension)
    """
    # Only save results when condition number is finite
    cond = metadata.get('condition_number', None)
    
    if cond is None or not np.isfinite(cond):
        print(f"Condition number not finite ({cond}) — skipping save for {filename}")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save S matrix
    np.save(output_dir / f'{filename}.npy', S)
    
    # Save metadata as JSON (convert numpy types and handle infinity)
    def convert_to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            val = obj.item()
            if np.isinf(val):
                return "infinity" if val > 0 else "-infinity"
            elif np.isnan(val):
                return "nan"
            return val
        elif isinstance(obj, float):
            if np.isinf(obj):
                return "infinity" if obj > 0 else "-infinity"
            elif np.isnan(obj):
                return "nan"
            return obj
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json(item) for item in obj]
        else:
            return obj
    
    metadata_json = {k: convert_to_json(v) for k, v in metadata.items()}
    
    with open(output_dir / f'{filename}_metadata.json', 'w') as f:
        json.dump(metadata_json, f, indent=2)
    
    print(f"Results saved: {output_dir / filename}")
