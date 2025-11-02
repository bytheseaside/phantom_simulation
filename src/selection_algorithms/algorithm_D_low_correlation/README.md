# Algorithm D — Greedy by Low Correlation

**Concept**: Select dipoles ensuring max pairwise correlation < rho_max. Guarantees diversity for better conditioning.

**Complexity**: O(nk²) where k ≈ 20 selected, ~10k correlation checks  
**Parameters**: `--rho-max` (default: 0.85, max correlation threshold)  
**Output**: ~10-18 dipoles (depends on rho_max), κ typically improved

## Usage

```bash
# Default: max correlation 0.85
python -m src.selection_algorithms.algorithm_D_low_correlation.select

# Stricter (more diverse)
python -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.7

# More relaxed
python -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.95
```

## Trade-offs

✓ Explicit diversity control, intuitive parameter, good conditioning, fast  
✗ May select fewer dipoles, greedy (order-dependent), doesn't maximize any criterion
