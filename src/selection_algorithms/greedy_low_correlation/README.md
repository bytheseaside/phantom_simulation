# Algorithm K: Greedy Low Correlation

## Goal
Select dipoles by **maintaining low correlation** with already-selected columns. Replaces deleted Algorithm D with proper implementation.

## Use Cases
- When you want uncorrelated (diverse) dipoles
- When you want to avoid redundant measurements
- When correlation matters more than energy
- Two different selection strategies available

## Parameters
- `rho_max` (float): Maximum allowed correlation with selected set (e.g., 0.7)
- `selection_order` (str, **REQUIRED**): 'energy_sorted' or 'greedy_min_corr'
- `start_col` (int, optional): Starting column (**only with greedy_min_corr mode**)
- `n_dipoles_max` (int, optional): Maximum number of dipoles to select

## How It Works (Recipe)

### Mode 1: energy_sorted
1. Sort all columns by energy (L2 norm)
2. Go through in descending energy order
3. For each candidate:
   - Compute max correlation with selected columns
   - Accept if max_corr < rho_max AND no triad violation
4. Stop when max_dipoles reached or no more candidates

### Mode 2: greedy_min_corr
1. Start with start_col (or highest energy if not specified)
2. Loop:
   - For each remaining candidate, compute max correlation with selected
   - Select candidate with **MINIMUM** max_correlation
   - Check if max_corr < rho_max (stop if exceeded)
   - Check triad violation
3. Stop when max_dipoles reached, rho_max exceeded, or no valid candidates

## Validation
- **start_col with energy_sorted**: RAISES ERROR (incompatible)
- **selection_order**: REQUIRED flag, must be one of two valid modes

## Complexity
- **energy_sorted**: O(nkm) where n=36, k≈20 selected, m=21 probes
- **greedy_min_corr**: O(n²km) - more expensive but globally optimal each step

## Example Usage
```bash
# Energy-sorted mode: high energy, low correlation
python -m src.selection_algorithms.greedy_low_correlation.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path src/model/base_matrices/B_matrix.npy \
    --w-matrix-path src/model/base_matrices/W_matrix.npy \
    --selection-order energy_sorted \
    --rho-max 0.7

# Greedy min-corr mode: select most uncorrelated
python -m src.selection_algorithms.greedy_low_correlation.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path src/model/base_matrices/B_matrix.npy \
    --w-matrix-path src/model/base_matrices/W_matrix.npy \
    --selection-order greedy_min_corr \
    --rho-max 0.5 \
    --n-dipoles-max 18

# With start column (only greedy_min_corr)
python -m src.selection_algorithms.greedy_low_correlation.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path src/model/base_matrices/B_matrix.npy \
    --w-matrix-path src/model/base_matrices/W_matrix.npy \
    --selection-order greedy_min_corr \
    --start-col 10 \
    --rho-max 0.6
```

## Notes
- Uses proper `np.corrcoef()` (not manual implementation like old Algorithm D)
- Two modes trade off energy vs correlation optimality
- energy_sorted: faster, energy-first
- greedy_min_corr: slower, correlation-first
- Different rho_max values give different selection sizes
