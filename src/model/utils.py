import json
import numpy as np
from pathlib import Path
from typing import List, Set, Tuple

def build_dipoles(n_electrodes: int = 9) -> List[Tuple[int, int]]:
    """
    Return list of ordered dipoles (i, j) with 1 <= i < j <= n_electrodes.
    Indexing here is 1-based to match the thesis notation.
    """
    dipoles = []
    for i in range(1, n_electrodes + 1):
        for j in range(i + 1, n_electrodes + 1):
            dipoles.append((i, j))
    return dipoles  # len == 36 for n_electrodes == 9 -> Order: (1,2), (1,3), ..., (7,8), (7,9), (8,9)


def contains_forbidden_triads(n_electrodes: int = 9, npy_path: Path = None) -> List[Set[Tuple[int, int]]]:
    """
    A forbidden triad is: ((i, j), (i, k), (j, k)) for 1 <= i < j < k <= n_electrodes.
    We return a list of triads, each triad is a set of dipole-tuples.

    This function saves the forbidden triads to a NPY file for efficient reuse.

    Math/Physics Context:
    ---------------------
    In the context of dipole modeling, a forbidden triad represents a set of three dipoles
    that share common electrodes, forming a closed loop. Such configurations are often
    undesirable in simulations due to redundancy or physical constraints. For example,
    if (i, j), (i, k), and (j, k) are all active dipoles, they form a triangle of connections
    that may violate the intended experimental setup.

    Parameters:
    - n_electrodes: Number of electrodes in the system.
    - csv_path: Path to save the forbidden triads as a CSV file (optional).
    - npy_path: Path to save the forbidden triads as an NPY file (optional).

    Returns:
    - triads: List of forbidden triads, where each triad is a set of dipole-tuples.
    """
    triads = []
    for i in range(1, n_electrodes + 1):
        for j in range(i + 1, n_electrodes + 1):
            for k in range(j + 1, n_electrodes + 1):
                triad = {(i, j), (i, k), (j, k)}
                triads.append(triad)

    # Save as NPY for efficient reuse if path is provided
    if npy_path:
        with npy_path.open('w') as f:
            for triad in triads:
                f.write(";".join([f"({i},{j})" for (i, j) in triad]) + "\n")
        np.save(npy_path, [list(triad) for triad in triads])

    return triads  # len == C(9,3) == 84 for n_electrodes == 9


def forms_forbidden_triad(dipole_list: List[Tuple[int, int]], forbidden_triads_path: Path) -> bool:
    """
    Check if the given list of dipoles forms any forbidden triad.

    Parameters:
    - dipole_list: List of dipoles [(i, j)].
    - forbidden_triads_path: Path to the NPY file containing forbidden triads.

    Returns:
    - True if any forbidden triad is formed, False otherwise.
    """
    # Load forbidden triads from the NPY file
    forbidden_triads = np.load(forbidden_triads_path, allow_pickle=True)
    forbidden_triads = [{tuple(pair) for pair in triad} for triad in forbidden_triads]

    current = set(dipole_list)

    # if any forbidden triad is subset of current → already faulty set
    for triad in forbidden_triads:
        if triad.issubset(current):
            return True
    return False