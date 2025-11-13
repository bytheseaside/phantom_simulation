# Algorithm F — Min-Condition Greedy

**Concept**: At each step, add dipole that minimizes κ(M). Global optimization for best conditioning.

**Complexity**: O(nk²·min(m,k)) ≈ 700 SVD calls, k ≈ 20 final, ~10 seconds  
**Parameters**: None  
**Output**: ~18-20 dipoles, κ minimized (typically 10¹-10²)

## Usage

```bash
python -m src.selection_algorithms.algorithm_F_min_condition.select

# WARNING: Computationally expensive (~700 SVD evaluations)
```

## Trade-offs

✓ Best conditioning achievable, no parameters, deterministic, principled  
✗ Very expensive (~10s), slow for large problems, greedy (not globally optimal)
