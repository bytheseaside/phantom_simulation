# Dipole Selection Algorithms

This directory contains implementations of seven algorithms for selecting active dipoles in the phantom-head transfer model while respecting forbidden triad constraints.

## Overview

The phantom-head system has:
- **9 internal electrodes** (antennas) with controllable voltages
- **36 possible dipoles** (all pairs of electrodes)
- **21 scalp probes** measuring potentials
- **Forbidden triads**: Sets of three dipoles that form closed triangles (physically undesirable)

## System Equation

All algorithms work with the unified system equation:

```
V_scalp^(w) = W · F · S · B · V_antenna
```

Where:
- **V_antenna** (9×1): Antenna voltages (input)
- **B** (36×9): Antenna → dipole difference matrix
- **S** (36×36): **Selection matrix** (diagonal, returned by algorithms)
- **F** (21×36): Dipole → scalp transfer matrix
- **W** (21×21): Probe weighting matrix (diagonal, optional)
- **V_scalp** (21×1): Predicted scalp voltages (output)

## Algorithms Implemented

### Algorithm A — Linear Independence + Energy (Novelty First)
**Criterion**: Linearly independent dipoles, prioritized by energy (L2 norm)

**Best for**: Balanced selection with good numerical properties

**Parameters**:
- `eps_rel`: Relative tolerance for independence (default: 1e-6)
- `use_qr`: Use QR decomposition vs least-squares (default: True)

**Typical output**: 18-21 dipoles

---

### Algorithm B — Greedy by Norm (Energy Baseline)
**Criterion**: Highest-energy dipoles first

**Best for**: Quick baseline, maximum visibility on scalp

**Parameters**: None (just uses norms)

**Typical output**: 18-20 dipoles

---

### Algorithm C — SVD-Based Leverage Score (Principal-Mode Coverage)
**Criterion**: Dipoles contributing most to dominant singular modes

**Best for**: Variance-optimal selection, captures principal patterns

**Parameters**:
- `r_keep`: Number of singular modes to consider (default: 10)
- `selection_mode`: 'all', 'top_n', or 'threshold'
- `n_dipoles_max`: Max dipoles for 'top_n' mode
- `score_threshold`: Min score for 'threshold' mode

**Typical output**: 18-20 dipoles (mode-dependent)

---

### Algorithm D — Greedy by Low Correlation (Maximize Diversity)
**Criterion**: Mutually uncorrelated dipole patterns

**Best for**: Well-conditioned models, decorrelated measurements

**Parameters**:
- `rho_max`: Maximum allowed correlation (default: 0.85)

**Typical output**: 10-18 dipoles (depends on ρ_max)

---

### Algorithm E — Greedy-by-Norm with Condition Cap
**Criterion**: High energy with explicit condition number constraint

**Best for**: Energy + guaranteed numerical stability

**Parameters**:
- `kappa_max`: Maximum allowed condition number (default: 1e4)

**Typical output**: Variable (depends on κ_max)

---

### Algorithm F — Min-Condition Greedy (Best-Improvement)
**Criterion**: Iteratively minimize condition number κ(M)

**Best for**: Most numerically stable model possible

**Parameters**: None

**Typical output**: 18-20 dipoles

**Warning**: Computationally expensive (~700 SVD calls)

---

### Algorithm G — D-Optimal (log-det) Design
**Criterion**: Maximize log-determinant of information matrix

**Best for**: Information-theoretic optimality, experimental design

**Parameters**:
- `regularization`: Numerical stability parameter (default: 1e-8)

**Typical output**: 18-20 dipoles

**Warning**: Computationally expensive (similar to Algorithm F)

---

## Quick Start

### Running a Single Algorithm

```python
from pathlib import Path
from selection_algorithms.algorithm_B_greedy_norm.select import select_dipoles_algorithm_B
from selection_algorithms.common import load_base_matrices, load_forbidden_triads

# Load data
base_path = Path('src/model/base_matrices')
matrices = load_base_matrices(base_path)

triads_path = Path('src/model/forbidden_triads.npy')
forbidden_triads = load_forbidden_triads(triads_path)

# Run algorithm
result = select_dipoles_algorithm_B(
    F=matrices['F'],
    B=matrices['B'],
    W=matrices['W'],  # or None for unweighted
    forbidden_triads=forbidden_triads,
    verbose=True
)

# Access results
S = result['S']
selected_dipoles = result['selected_dipoles']
n_selected = result['n_selected']
condition_number = result['condition_number']
```

### Running All Algorithms (Comparison)

```python
from selection_algorithms.run_all import run_all_algorithms

results = run_all_algorithms(
    output_dir='results/comparison',
    verbose=True
)

# results is a dict: {'A': result_A, 'B': result_B, ...}
```

## Output Format

All algorithms return a dictionary with:

| Key | Type | Description |
|-----|------|-------------|
| `S` | np.ndarray (36×36) | Diagonal selection matrix |
| `selected_dipoles` | List[Tuple] | Selected dipole pairs (1-indexed) |
| `selected_indices` | List[int] | Selected indices (0-indexed) |
| `n_selected` | int | Number of dipoles selected |
| `condition_number` | float | κ(M = W·F·S·B) |
| `algorithm` | str | Algorithm identifier ('A', 'B', ..., 'G') |
| `parameters` | dict | Algorithm-specific parameters used |

Additional keys vary by algorithm (e.g., `norms`, `scores`, `correlations`, etc.).

## File Structure

```
selection_algorithms/
├── common.py                          # Shared utilities
├── algorithm_A_linear_independence/
│   ├── README.md                      # Detailed documentation
│   └── select.py                      # Implementation
├── algorithm_B_greedy_norm/
│   ├── README.md
│   └── select.py
├── algorithm_C_svd_leverage/
│   ├── README.md
│   └── select.py
├── algorithm_D_low_correlation/
│   ├── README.md
│   └── select.py
├── algorithm_E_condition_cap/
│   ├── README.md
│   └── select.py
├── algorithm_F_min_condition/
│   ├── README.md
│   └── select.py
├── algorithm_G_d_optimal/
│   ├── README.md
│   └── select.py
└── run_all.py                         # Batch runner
```

## Performance Comparison

| Algorithm | Complexity | Speed | Conditioning | Selection Size |
|-----------|-----------|-------|--------------|----------------|
| **A** | O(n k² m) | Fast | Moderate | 18-21 |
| **B** | O(n log n) | Very Fast | Variable | 18-20 |
| **C** | O(m² n) | Fast | Good | 18-20 |
| **D** | O(n² m) | Fast | Good | 10-18 |
| **E** | O(n² m²) | Medium | Controlled | Variable |
| **F** | O(n² k m²) | Slow | Best | 18-20 |
| **G** | O(n² k m) | Slow | Excellent | 18-20 |

Where: n=36 (dipoles), m=21 (probes), k≈20 (selected)

## Recommendations

- **Start with Algorithm B**: Fastest, good baseline
- **For balanced quality**: Algorithm A or C
- **For best conditioning**: Algorithm F or G (if you have time)
- **For decorrelation**: Algorithm D
- **For explicit stability**: Algorithm E

## Common Issues

### Import Errors
Make sure you're running from the workspace root:
```bash
cd /Users/brisarojas/Desktop/phantom_simulation
python -m src.selection_algorithms.algorithm_B_greedy_norm.select
```

### Missing Matrices
Ensure `F_matrix.npy`, `B_matrix.npy`, `W_matrix.npy` exist in `src/model/base_matrices/`.

### Forbidden Triads
Ensure `forbidden_triads.npy` exists in `src/model/`.

## References

- Experimental Design: Pukelsheim, F. (2006). *Optimal Design of Experiments*
- Column Subset Selection: Boutsidis et al. (2009). *An improved approximation algorithm for the column subset selection problem*
- SVD & Leverage Scores: Mahoney, M. W. (2011). *Randomized algorithms for matrices and data*

## Contributing

To add a new algorithm:
1. Create a new directory `algorithm_X_name/`
2. Add `README.md` with documentation
3. Add `select.py` with function `select_dipoles_algorithm_X()`
4. Follow the common output format
5. Update this main README

## License

Part of the phantom_simulation project.
