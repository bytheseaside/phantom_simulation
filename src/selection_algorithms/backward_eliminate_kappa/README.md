# Algorithm L: Backward Eliminate Kappa

## Goal
**Backward elimination** to minimize condition number. Start with all 36 dipoles, iteratively remove the column whose removal most improves κ.

## Use Cases
- When you want optimal conditioning through elimination
- When starting from a complete set makes sense
- When computational cost is acceptable
- Global optimization of conditioning

## Parameters
- `n_dipoles_target` (int, default=18): Stop when this many dipoles remain
- `target_kappa` (float, optional): Stop if κ <= target_kappa achieved

## How It Works (Recipe)
1. **Initialize**: Start with all 36 dipoles
2. **Loop** (while more than n_dipoles_target):
   - For each remaining dipole:
     - Test removing it
     - Check if removal creates triad violation (skip if so)
     - Compute κ of remaining set
   - Remove the dipole that gives **smallest** κ
   - Record κ history
3. **Stop** when: target count reached, target κ reached, or no valid removal

## Triad Handling
**Option 1**: If removal creates triad, skip that candidate and try next

## Stop Conditions
1. `n_dipoles_target` reached (primary)
2. `target_kappa` achieved (optional early stop)
3. No valid removal (all remaining removals create triads)

## Complexity
- **Time**: O(n²k·min(m,k)) where n=36, k starts at 36 and decreases
- **Space**: O(mn) for condition number computation
- **WARNING**: Very expensive! ~630 condition number evaluations

## Example Usage
```bash
# Default: eliminate down to 18 dipoles
python -m src.selection_algorithms.backward_eliminate_kappa.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path src/model/base_matrices/B_matrix.npy \
    --w-matrix-path src/model/base_matrices/W_matrix.npy \
    --n-dipoles-target 18

# With target condition number (early stop)
python -m src.selection_algorithms.backward_eliminate_kappa.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path src/model/base_matrices/B_matrix.npy \
    --w-matrix-path src/model/base_matrices/W_matrix.npy \
    --target-kappa 1e3

# Both conditions (whichever comes first)
python -m src.selection_algorithms.backward_eliminate_kappa.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path src/model/base_matrices/B_matrix.npy \
    --w-matrix-path src/model/base_matrices/W_matrix.npy \
    --n-dipoles-target 20 \
    --target-kappa 5e3
```

## Notes
- Backward elimination: opposite of forward selection
- Computationally expensive (starts with full set)
- May find better solutions than greedy forward
- κ history tracked for analysis
- Different from Algorithm F (forward greedy min κ)
