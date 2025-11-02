# Algorithm B — Greedy by Norm (Energy Baseline)

**Concept**: Select dipoles by descending L2-norm (RMS scalp voltage), skipping forbidden triads. Simplest energy baseline.

**Complexity**: O(n log n) sort + O(ntk) triad checks ≈ 60k ops, deterministic  
**Parameters**: None (uses default matrices)  
**Output**: ~18-20 dipoles, κ typically 10²-10⁶ (uncontrolled)

## Usage

```bash
python -m src.selection_algorithms.algorithm_B_greedy_norm.select_v2 \
    --base-matrices src/model/base_matrices \
    --forbidden-triads src/model/forbidden_triads.npy
```

## Trade-offs

✓ Extremely simple, fast, reproducible, good baseline  
✗ No conditioning control, greedy (no global optimization), ignores linear dependence
