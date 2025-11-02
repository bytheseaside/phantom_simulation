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


def load_base_matrices(base_path: Path) -> Dict[str, np.ndarray]:
    """Load F, B, W matrices from directory."""
    return {
        'F': np.load(base_path / 'F_matrix.npy'),
        'B': np.load(base_path / 'B_matrix.npy'),
        'W': np.load(base_path / 'W_matrix.npy')
    }


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


def save_selection_results(
    S: np.ndarray,
    metadata: Dict[str, Any],
    output_dir: Path,
    filename: str
) -> None:
    """
    Save selection matrix S and metadata.
    
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
