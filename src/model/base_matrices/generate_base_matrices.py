import numpy as np
from pathlib import Path    

def build_w_matrix(col_weights: np.ndarray, out: Path) -> np.ndarray:
    """
    Build the W matrix (weighting matrix) based on the provided column weights.

    Parameters
    ----------
    col_weights : np.ndarray
        A 1D array of weights for each column (probe).
    out : Path
        Path to save the resulting W matrix as a .npy file.

    Returns
    -------
    np.ndarray
        The constructed W matrix (diagonal matrix).
    """
    if col_weights.ndim != 1:
        raise ValueError("col_weights must be a 1D array")

    # Create a diagonal matrix with the weights
    w_matrix = np.diag(col_weights)

    # Save the matrix to the specified output path
    np.save(out, w_matrix)
    print(f"W matrix saved to {out}")

    return w_matrix

def generate_b_matrix(pairs: list[tuple[int, int]], num_electrodes: int, out: Path) -> np.ndarray:
    """
    Generate the B matrix for dipoles based on a list of electrode pairs.

    Parameters
    ----------
    pairs : list of tuple[int, int]
        A list of electrode pairs, where each pair (a, b) defines a dipole.
    num_electrodes : int
        The total number of electrodes.
    out : Path
        Path to save the resulting B matrix as a .npy file.

    Returns
    -------
    np.ndarray
        The constructed B matrix.
    """
    if num_electrodes < 2:
        raise ValueError("Number of electrodes must be at least 2 to form dipoles.")

    # Initialize the B matrix
    num_dipoles = len(pairs)
    b_matrix = np.zeros((num_dipoles, num_electrodes))

    # Fill the B matrix based on the provided pairs
    for dipole_index, (a, b) in enumerate(pairs):
        if not (0 <= a < num_electrodes and 0 <= b < num_electrodes):
            raise ValueError(f"Electrode indices {a}, {b} are out of bounds for {num_electrodes} electrodes.")
        # Convention: pair (a,b) means positive in X and negative in Y
        b_matrix[dipole_index, a] = 1  
        b_matrix[dipole_index, b] = -1

    # Save the matrix to the specified output path
    np.save(out, b_matrix)
    print(f"B matrix saved to {out}")

    return b_matrix

def build_s_matrix(selected_columns: list[int], total_columns: int, out: Path = None) -> np.ndarray:
    """
    Build the S matrix (selection matrix) based on the selected columns.
    Can optionally save to disk.

    Parameters
    ----------
    selected_columns : list[int]
        A list of column indices to keep in the matrix.
    total_columns : int
        The total number of columns in the original matrix.
    out : Path, optional
        Path to save the resulting S matrix as a .npy file.
        If None, matrix is not saved to disk.

    Returns
    -------
    np.ndarray
        The constructed S matrix.
    """
    if any(col < 0 or col >= total_columns for col in selected_columns):
        raise ValueError("Selected column indices are out of bounds.")

    s_matrix = np.zeros((total_columns, total_columns))
    for col_index in selected_columns:
        s_matrix[col_index, col_index] = 1 

    if out is not None:
        np.save(out, s_matrix)
        print(f"S matrix saved to {out}")

    return s_matrix



