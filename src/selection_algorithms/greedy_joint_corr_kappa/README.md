# Algorithm N: Greedy Joint Correlation + Kappa

**Balances low correlation with good conditioning via weighted joint criterion.**

---

## Goal

Select columns by minimizing a joint score that combines:
- **Correlation**: Maximum correlation with already-selected columns
- **Condition number**: Matrix conditioning after adding candidate

The weighted combination allows tuning the tradeoff via `alpha`:
- `alpha=0`: Pure conditioning optimization (ignore correlation)
- `alpha=0.5`: Balanced (default)
- `alpha=1`: Pure correlation minimization (ignore conditioning)

---

## Use Cases

1. **Need balance**: When both low correlation AND good conditioning matter
2. **Tuneable tradeoff**: Sweep `alpha` to explore Pareto frontier
3. **Domain constraints**: When correlation caps are critical but κ still matters
4. **Comparative study**: Evaluate joint vs. sequential optimization

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `alpha` | float | 0.5 | Weight for correlation term (0=pure κ, 1=pure ρ) |
| `n_dipoles_max` | int | 20 | Maximum dipoles to select |
| `start_col` | int | None | Force first column (else highest energy) |

---

## Recipe

1. Start with `start_col` (if given) or highest-energy column
2. For each iteration:
   - For each candidate column `j`:
     - Compute `max_corr`: max correlation with selected columns
     - Compute `κ_test`: condition number with `j` added
     - Normalize: `κ_norm = κ_test / κ_init`
     - Joint score: `score = alpha * max_corr + (1-alpha) * κ_norm/100`
   - Select candidate with **minimum** joint score (no triad violations)
3. Stop when `n_dipoles_max` reached or no valid candidates

---

## Complexity

- **Time**: O(n²km) where n=36, k≈20 selected, m=21 probes
  - Each iteration: O(nm) for correlation + O(mk) for SVD
  - Similar to Algorithm K but with κ computation added
- **Space**: O(nm) for correlation matrices

Moderately expensive due to SVD per candidate, but practical for n=36.

---

## Examples

### Balanced mode (alpha=0.5)
```bash
python -m src.selection_algorithms.greedy_joint_corr_kappa.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path run/f_matrix/B_matrix.npy \
    --alpha 0.5 \
    --n-dipoles-max 20
```

### Correlation-focused (alpha=0.8)
When correlation constraints dominate:
```bash
python -m src.selection_algorithms.greedy_joint_corr_kappa.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path run/f_matrix/B_matrix.npy \
    --alpha 0.8 \
    --n-dipoles-max 18
```

### Conditioning-focused (alpha=0.2)
When matrix stability is paramount:
```bash
python -m src.selection_algorithms.greedy_joint_corr_kappa.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path run/f_matrix/B_matrix.npy \
    --alpha 0.2 \
    --n-dipoles-max 22 \
    --start-col 5
```

---

## Notes

- **Normalization**: κ scaled by initial κ and divided by 100 to match correlation scale (0-1)
- **Pareto frontier**: Sweep alpha ∈ [0,1] to explore correlation-conditioning tradeoff
- **Comparison**: Contrast with sequential approaches (optimize one, then the other)
- **Greedy limitation**: May not find global optimum of joint criterion
