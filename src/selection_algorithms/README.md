# Selection Algorithms

**14 algorithms for optimal dipole selection**

See [ALGORITHM_INVENTORY.md](./ALGORITHM_INVENTORY.md) for comprehensive algorithm summary table.

---

## Quick Start

### Run Individual Algorithm

```bash
# Example: Greedy energy + QR independence
python3 -m src.selection_algorithms.greedy_energy_qr_independence.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path run/f_matrix/B_matrix.npy

# Example: Genetic optimization (slow but global)
python3 -m src.selection_algorithms.genetic_optimize_kappa.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path run/f_matrix/B_matrix.npy \
    --fitness-mode kappa \
    --n-generations 100
```

### Test All Algorithms

```bash
./src/selection_algorithms/test_all_algorithms.sh
```

### Run All Variants with Parameter Sweeps

```bash
./src/selection_algorithms/run_all_variants.sh
```

---

## Algorithm List

| # | Name | Strategy | Speed |
|---|------|----------|-------|
| 1 | `greedy_energy_qr_independence` | Energy + QR independence | Fast |
| 2 | `greedy_energy_simple` | Pure energy (baseline) | Very Fast |
| 3 | `svd_leverage_topk` | Forward leverage scores | Very Fast |
| 4 | `greedy_energy_condition_cap` | Energy + κ cap | Moderate |
| 5 | `greedy_min_condition` | Minimize κ greedily | Moderate |
| 6 | `greedy_max_logdet` | Maximize log-determinant | Slow |
| 7 | `rrqr_importance_ranking` | Rank-revealing QR | Very Fast |
| 8 | `rrqr_correlation_optimize` | Hybrid RRQR→SVD | Moderate |
| 9 | `greedy_max_independence` ⭐ | Pure independence | Fast |
| 10 | `greedy_low_correlation` ⭐ | Low correlation (2 modes) | Fast |
| 11 | `backward_eliminate_kappa` ⭐ | Remove to minimize κ | Very Slow |
| 12 | `backward_eliminate_leverage` ⭐ | Remove lowest importance | Moderate |
| 13 | `genetic_optimize_kappa` ⭐ | Evolutionary search | Very Slow |
| 14 | `greedy_joint_corr_kappa` ⭐ | Weighted joint criterion | Moderate |

⭐ = New algorithm

---

## Python Usage

```python
from pathlib import Path
import numpy as np
from selection_algorithms.greedy_energy_simple.select import select_dipoles_greedy_energy_simple
from selection_algorithms.common import load_forbidden_triads
from model.utils import build_dipoles

# Load matrices
F = np.load('run/f_matrix/F_matrix.npy')
B = np.load('run/f_matrix/B_matrix.npy')
W = np.load('run/f_matrix/W_matrix.npy')  # optional

triads = load_forbidden_triads(Path('src/model/forbidden_triads.npy'))
all_dipoles = build_dipoles()

# Run algorithm
result = select_dipoles_greedy_energy_simple(
    F=F, B=B, W=W,
    forbidden_triads=triads,
    all_dipoles=all_dipoles,
    verbose=True
)

# Access results
print(f"Selected {result['n_selected']} dipoles")
print(f"Condition number: {result['condition_number']:.2e}")
print(f"Dipoles: {result['selected_dipoles']}")
S_matrix = result['S']  # Selection matrix (36, k)
```

---

## Common Parameters

All algorithms support:
- `--run-dir`: Run directory (mesh-specific)
- `--f-matrix-path`: Path to F_matrix.npy
- `--b-matrix-path`: Path to B_matrix.npy
- `--w-matrix-path`: Path to W_matrix.npy (optional)
- `--forbidden-triads`: Path to forbidden triads file
- `--no-save`: Don't save results
- `--quiet`: Suppress output

Algorithm-specific parameters vary. See individual README files:
```
src/selection_algorithms/{algorithm_name}/README.md
```

---

## Key Features

All algorithms include:
- ✅ Forbidden triad checking
- ✅ Matrix shape validation
- ✅ CLI with argparse
- ✅ Result saving to `run/results/{algorithm_name}/`
- ✅ Comprehensive README documentation

---

## Documentation

- **[ALGORITHM_INVENTORY.md](./ALGORITHM_INVENTORY.md)**: Complete algorithm summary table
- **Individual READMEs**: Goal, use cases, parameters, complexity, examples
- **[common.py](./common.py)**: Shared utilities (validation, triad checking, etc.)
