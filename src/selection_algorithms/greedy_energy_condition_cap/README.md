# Algorithm E — Greedy-by-Norm with Condition Cap

**Concept**: Select by descending energy, accept only if κ(M) ≤ kappa_max. Trades selection size for guaranteed conditioning.

**Complexity**: O(n log n + nk²·min(m,k)) for sort + k SVD checks, k ≈ final count  
**Parameters**: `--kappa-max` (default: 1e4, max condition number)  
**Output**: Variable (~5-20 dipoles), κ ≤ kappa_max guaranteed

## Usage

```bash
# Default: κ_max = 1e4
python -m src.selection_algorithms.algorithm_E_condition_cap.select

# Stricter conditioning
python -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e3

# More relaxed
python -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e5
```

## Trade-offs

✓ Guaranteed conditioning, simple parameter, energy-prioritized, deterministic  
✗ May select few dipoles, expensive (k SVD calls), greedy (not optimal selection size)
