# Algorithm G — D-Optimal Design

**Concept**: Maximize log det(Φ) where Φ = (FS)ᵀ(FS) is information matrix. Information-theoretic optimal design.

**Complexity**: O(nk³) ≈ 700 log-det calls, k ≈ 20 final, ~10 seconds  
**Parameters**: `--regularization` (default: 1e-8, for numerical stability)  
**Output**: ~18-20 dipoles, log-det maximized, κ variable

## Usage

```bash
# Default: regularization = 1e-8
python -m src.selection_algorithms.algorithm_G_d_optimal.select

# Test regularization range (recommended)
for reg in 1e-12 1e-10 1e-8 1e-6 1e-4; do
  python -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization $reg
done

# WARNING: Computationally expensive (~700 log-det evaluations)
```

## Regularization Parameter

**Recommended test range**: 1e-12, 1e-10, 1e-8, 1e-6, 1e-4

- **Too small (< 1e-10)**: Numerical instability, log-det may be unreliable
- **Just right (1e-10 to 1e-6)**: Stable computation, minimal bias
- **Too large (> 1e-4)**: Over-regularization, biases selection toward high-energy dipoles

## Trade-offs

✓ Information-theoretic optimal, variance minimization, principled criterion  
✗ Very expensive (~10s), no direct κ control, sensitive to regularization
