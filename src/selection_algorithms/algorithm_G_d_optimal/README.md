# Algorithm G — D-Optimal Design

**Concept**: Maximize log det(Φ) where Φ = (FS)ᵀ(FS) is information matrix. Information-theoretic optimal design.

**Complexity**: O(nk³) ≈ 700 log-det calls, k ≈ 20 final, ~10 seconds  
**Parameters**: `--regularization` (default: 1e-8, for numerical stability)  
**Output**: ~18-20 dipoles, log-det maximized, κ variable

## Usage

```bash
# Default: regularization = 1e-8
python -m src.selection_algorithms.algorithm_G_d_optimal.select

# Custom regularization
python -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-6

# WARNING: Computationally expensive (~700 log-det evaluations)
```

## Trade-offs

✓ Information-theoretic optimal, variance minimization, principled criterion  
✗ Very expensive (~10s), no direct κ control, sensitive to regularization
