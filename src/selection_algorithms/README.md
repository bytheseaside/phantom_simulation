# Quick Start

## Run Algorithms

```bash
# Individual algorithms
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select
python3 -m src.selection_algorithms.algorithm_B_greedy_norm.select
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select
python3 -m src.selection_algorithms.algorithm_F_min_condition.select      # slow
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select          # slow

# Test all algorithms
./src/selection_algorithms/test_all_algorithms.sh

# Run all with varied parameters
./src/selection_algorithms/run_all_variants.sh
```

## Python Usage

```python
from pathlib import Path
from selection_algorithms.algorithm_B_greedy_norm.select import select_dipoles_algorithm_B
from selection_algorithms.common import load_base_matrices, load_forbidden_triads

matrices = load_base_matrices(Path('src/model/base_matrices'))
triads = load_forbidden_triads(Path('src/model/forbidden_triads.npy'))

result = select_dipoles_algorithm_B(
    F=matrices['F'], B=matrices['B'], W=matrices['W'],
    forbidden_triads=triads, verbose=True
)

# result = {'S': array(36×36), 'selected_dipoles': [(1,8), ...], 'n_selected': 16, 'condition_number': 37.7, ...}
```

## CLI Parameters

Each algorithm has `--base-matrices`, `--forbidden-triads`, `--output-dir`, `--no-save`, `--quiet` plus:

- **A:** `--eps-rel` (rank tolerance, default 1e-6)
- **C:** `--r-keep`, `--selection-mode`, `--n-dipoles-max`, `--score-threshold`
- **D:** `--rho-max` (correlation threshold, default 0.3)
- **E:** `--kappa-max` (condition limit, default 1e4)
- **G:** `--regularization` (default 1e-8)

See individual READMEs in `src/selection_algorithms/algorithm_X_name/` for details.
