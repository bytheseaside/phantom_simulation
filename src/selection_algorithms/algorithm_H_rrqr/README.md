# Algorithm H: Rank-Revealing QR (RRQR)

**Concept:** QR decomposition with column pivoting naturally ranks columns by importance (R diagonal magnitude). Deterministic, linearly independent selection.

**Complexity:** O(nm²) for QR + O(nt) for triad checking  
**Parameters:** `--n-dipoles-max` (max dipoles to select, default 20)  
**Output:** Linearly independent set ordered by column importance

## Usage

```bash
# Default: up to 20 dipoles
python3 -m src.selection_algorithms.algorithm_H_rrqr.select

# Limit to 16 dipoles
python3 -m src.selection_algorithms.algorithm_H_rrqr.select --n-dipoles-max 16

# Save results
python3 -m src.selection_algorithms.algorithm_H_rrqr.select --output-dir results/rrqr_test
```

## Trade-offs

**Pros:** Deterministic, guaranteed linear independence, fast (single QR), interpretable importance ranking  
**Cons:** No explicit conditioning optimization, fixed ordering may miss better combinations
