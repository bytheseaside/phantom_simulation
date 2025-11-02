# Algorithm A — Linear Independence + Energy

**Concept**: Select by descending energy, accept if residual norm after projection > eps_rel × original norm. Ensures linearly independent selection.

**Complexity**: O(n log n + nk²m) ≈ 700k ops, k ≈ 20 selected, deterministic  
**Parameters**: `--eps-rel` (default: 1e-6, relative independence tolerance)  
**Output**: ~18-21 dipoles, κ typically 10²-10⁴ (moderate)

## Usage

```bash
python -m src.selection_algorithms.algorithm_A_linear_independence.select \
    --base-matrices src/model/base_matrices \
    --forbidden-triads src/model/forbidden_triads.npy \
    --eps-rel 1e-6
```

## Trade-offs

✓ Numerically principled, enforces independence, efficient (QR), interpretable  
✗ Greedy (no global optimization), order-dependent, no explicit condition control
