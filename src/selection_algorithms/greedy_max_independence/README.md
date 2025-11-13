# Algorithm J: Greedy Max Independence

## Goal
Select dipoles by **maximizing independence** from already-selected columns. Pure independence-driven selection.

## Use Cases
- When you want maximum diversity in column space
- When you need columns that span different directions
- When you want to avoid redundancy
- Starting from a specific important dipole

## Parameters
- `start_col` (int): Starting column index (0-35), forced as first selection
- `n_dipoles_max` (int, optional): Stop after N dipoles selected
- `eps_abs` (float, optional): Stop if residual norm < eps_abs (absolute threshold)
- `min_residual_ratio` (float, optional): Stop if residual/original < ratio (relative threshold)

## How It Works (Recipe)
1. **Initialize**: Start with `start_col` as first selection
2. **Loop**:
   - Compute QR decomposition of currently selected columns
   - For each remaining candidate column:
     - Project onto span of selected columns
     - Compute residual norm (how much is "new")
     - Skip if triad violation
   - Select column with **LARGEST** residual norm (most independent)
   - Check stop conditions
3. **Stop** when: max dipoles reached, no valid candidates, or thresholds met

## Stop Conditions (4 total)
1. `n_dipoles_max` reached
2. No more valid candidates (all remaining cause triads)
3. `eps_abs` threshold: residual norm < eps_abs (absolute measure)
4. `min_residual_ratio`: residual/original < ratio (relative measure)

## Complexity
- **Time**: O(nkm) where n=36 dipoles, k≈20 selected, m=21 probes
- **Space**: O(mk) for QR decomposition

## Example Usage
```bash
# Start from column 5, select up to 18 dipoles
python -m src.selection_algorithms.greedy_max_independence.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path src/model/base_matrices/B_matrix.npy \
    --w-matrix-path src/model/base_matrices/W_matrix.npy \
    --start-col 5 \
    --n-dipoles-max 18

# With absolute threshold
python -m src.selection_algorithms.greedy_max_independence.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path src/model/base_matrices/B_matrix.npy \
    --w-matrix-path src/model/base_matrices/W_matrix.npy \
    --start-col 0 \
    --eps-abs 1e-6

# With residual ratio threshold (stop when residual is < 10% of original)
python -m src.selection_algorithms.greedy_max_independence.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path src/model/base_matrices/B_matrix.npy \
    --w-matrix-path src/model/base_matrices/W_matrix.npy \
    --start-col 10 \
    --min-residual-ratio 0.1
```

## Notes
- Starts from specified column (unlike Algorithm A which sorts by energy first)
- Pure independence criterion (no energy weighting)
- Greedy: locally optimal at each step
- Different start_col will give different results
