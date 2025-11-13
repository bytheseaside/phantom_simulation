# Algorithm I: Hybrid Greedy→SVD

**Concept:** Two-phase: (1) Greedy-by-Norm preselects ~20 triad-free dipoles, (2) SVD re-ranks by leverage scores to pick best subset. Combines speed with quality.

**Complexity:** O(n log n + nt) for greedy + O(k²m) for SVD on preselected (k≈20)  
**Parameters:** `--n-preselect` (Phase 1 size, default 20), `--n-final` (output size, default 16), `--r-keep` (SVD components, default 10)  
**Output:** High-quality subset from greedy preselection

## Usage

```bash
# Default: preselect 20, output 16, use 10 SVD components
python3 -m src.selection_algorithms.algorithm_I_hybrid.select

# More aggressive preselection
python3 -m src.selection_algorithms.algorithm_I_hybrid.select --n-preselect 25 --n-final 18

# Different SVD components
python3 -m src.selection_algorithms.algorithm_I_hybrid.select --r-keep 12

# Save results
python3 -m src.selection_algorithms.algorithm_I_hybrid.select --output-dir results/hybrid_test
```

## Trade-offs

**Pros:** Fast (greedy+small SVD), balances energy and variance, flexible parameters  
**Cons:** Preselection biases final result, two stages add complexity
