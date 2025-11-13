# Algorithm O: Backward Eliminate Leverage

## Goal
**Backward elimination by leverage score**. Start with all 36 dipoles, iteratively remove column with lowest leverage (least important).

## Use Cases
- Backward elimination based on importance
- Remove least-contributing columns
- SVD-based importance metric
- Faster than Algorithm L (no condition number per removal)

## Parameters
- `n_dipoles_target` (int, default=18): Stop when this many remain
- `r_keep` (int, default=10): Number of top singular modes for leverage

## How It Works
1. Start with all 36 dipoles
2. Loop (while > n_dipoles_target):
   - Compute SVD of current selection
   - Compute leverage scores: Σ(r=1..r_keep) σ_r² × V[j,r]²
   - Find column with **lowest** leverage (least important)
   - Check triad violation (skip if creates triad)
   - Remove that column
3. Stop when target count reached

## Complexity
- **Time**: O(nk·min(m,k)) where n=36, k starts at 36
- **Space**: O(mk) for SVD
- Faster than Algorithm L (no κ computation per removal)

## Example Usage
```bash
python -m src.selection_algorithms.backward_eliminate_leverage.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path src/model/base_matrices/B_matrix.npy \
    --w-matrix-path src/model/base_matrices/W_matrix.npy \
    --n-dipoles-target 18 \
    --r-keep 10
```

## Notes
- Opposite of Algorithm C (forward leverage-based selection)
- Remove least important vs select most important
- Leverage = contribution to top-r singular modes
