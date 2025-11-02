# Algorithm C — SVD-Based Leverage Score

**Concept**: Select dipoles by contribution to top-r singular modes. Score[j] = Σ(σ_r² × V[j,r]²) measures participation in principal subspace.

**Complexity**: O(mn² + nr) for SVD + scoring ≈ 30k ops, deterministic  
**Parameters**: `--r-keep` (default: 10), `--selection-mode` (all/top_n/threshold), `--n-dipoles-max`, `--score-threshold`  
**Output**: Flexible (depends on mode), κ variable

## Usage

```bash
# Default: all non-triad, top 10 modes
python -m src.selection_algorithms.algorithm_C_svd_leverage.select

# Top 15 dipoles from top 12 modes
python -m src.selection_algorithms.algorithm_C_svd_leverage.select \
    --r-keep 12 --selection-mode top_n --n-dipoles-max 15

# Score threshold mode
python -m src.selection_algorithms.algorithm_C_svd_leverage.select \
    --selection-mode threshold --score-threshold 0.01
```

## Trade-offs

✓ Principal-mode focused, variance explained, flexible selection, interpretable scores  
✗ Doesn't guarantee independence, no conditioning control, parameter-dependent
