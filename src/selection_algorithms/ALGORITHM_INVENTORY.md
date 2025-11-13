# Selection Algorithm Inventory

**14 algorithms for optimal dipole selection**

## Algorithm Summary Table

| ID     | Name                            | Folder                          | Strategy                                   | Key Parameters                                     | Complexity          |
| ------ | ------------------------------- | ------------------------------- | ------------------------------------------ | -------------------------------------------------- | ------------------- |
| **1**  | Greedy Energy + QR Independence | `greedy_energy_qr_independence` | Energy-sorted, QR independence             | `eps_rel`                                          | O(nkm)              |
| **2**  | Greedy Energy Simple            | `greedy_energy_simple`          | Pure energy (baseline)                     | -                                                  | O(n)                |
| **3**  | SVD Leverage Top-K              | `svd_leverage_topk`             | Forward leverage scores                    | `ensure_count`, `min_variance`, `start_col`        | O(nm²)              |
| **4**  | Greedy Energy + Condition Cap   | `greedy_energy_condition_cap`   | Energy with κ constraint                   | `kappa_max`                                        | O(n²km)             |
| **5**  | Greedy Min Condition            | `greedy_min_condition`          | Minimize κ greedily                        | `max_dipoles`, `target_kappa`, `delta_threshold`   | O(n²km)             |
| **6**  | Greedy Max LogDet               | `greedy_max_logdet`             | Maximize log-determinant                   | `regularization`                                   | O(n²k³)             |
| **7**  | RRQR Importance Ranking         | `rrqr_importance_ranking`       | Rank-revealing QR                          | -                                                  | O(nm²)              |
| **8**  | RRQR + Correlation Optimize     | `rrqr_correlation_optimize`     | Hybrid RRQR→SVD                            | `rho_max`, `selection_order`                       | O(nm² + n²k)        |
| **9**  | Greedy Max Independence ⭐      | `greedy_max_independence`       | Pure independence (residual norm)          | `start_col`, `n_dipoles_max`, `eps_abs`            | O(nkm)              |
| **10** | Greedy Low Correlation ⭐       | `greedy_low_correlation`        | Two modes: energy-sorted / greedy-min-corr | `rho_max`, `selection_order`                       | O(n²k)              |
| **11** | Backward Eliminate Kappa ⭐     | `backward_eliminate_kappa`      | Remove to minimize κ                       | `n_dipoles_target`, `target_kappa`                 | O(n²k·min(m,k))     |
| **12** | Backward Eliminate Leverage ⭐  | `backward_eliminate_leverage`   | Remove lowest importance                   | `n_dipoles_target`, `r_keep`                       | O(nk·min(m,k))      |
| **13** | Genetic Optimize Kappa ⭐       | `genetic_optimize_kappa`        | Evolutionary population-based              | `population_size`, `n_generations`, `fitness_mode` | O(G·P·k·m·min(m,k)) |
| **14** | Greedy Joint Corr+Kappa ⭐      | `greedy_joint_corr_kappa`       | Weighted joint criterion                   | `alpha`, `n_dipoles_max`                           | O(n²km)             |

⭐ = New algorithm (Phase 3)

---

## Quick Selection Guide

### Choose by Primary Goal

**Minimize Condition Number:**

- #5: `greedy_min_condition` (greedy, fast)
- #11: `backward_eliminate_kappa` (exhaustive, slow)
- #13: `genetic_optimize_kappa` (evolutionary, global)

**Maximize Independence:**

- #1: `greedy_energy_qr_independence` (energy-weighted)
- #9: `greedy_max_independence` (pure independence)

**Minimize Correlation:**

- #10: `greedy_low_correlation` (two modes)
- #8: `rrqr_correlation_optimize` (hybrid)

**Maximize Determinant:**

- #6: `greedy_max_logdet` (volume maximization)

**Balance Multiple Criteria:**

- #14: `greedy_joint_corr_kappa` (weighted corr+kappa)
- #4: `greedy_energy_condition_cap` (energy + kappa cap)

**Fast Baseline:**

- #2: `greedy_energy_simple` (pure energy, no optimization)
- #3: `svd_leverage_topk` (single SVD, very fast)

**Global Search:**

- #13: `genetic_optimize_kappa` (evolutionary, 3 fitness modes)

---

## Computational Cost Ranking

**Fast** (seconds):

- #2: `greedy_energy_simple` - O(n)
- #3: `svd_leverage_topk` - O(nm²) single SVD
- #7: `rrqr_importance_ranking` - O(nm²) single QR

**Medium** (tens of seconds):

- #1: `greedy_energy_qr_independence` - O(nkm)
- #9: `greedy_max_independence` - O(nkm)
- #10: `greedy_low_correlation` - O(n²k) correlation only
- #12: `backward_eliminate_leverage` - O(nk·min(m,k))

**Expensive** (minutes):

- #4: `greedy_energy_condition_cap` - O(n²km)
- #5: `greedy_min_condition` - O(n²km)
- #6: `greedy_max_logdet` - O(n²k³)
- #8: `rrqr_correlation_optimize` - O(nm² + n²k)
- #14: `greedy_joint_corr_kappa` - O(n²km)

**Very Expensive** (many minutes):

- #11: `backward_eliminate_kappa` - O(n²k·min(m,k)) ~630 SVDs
- #13: `genetic_optimize_kappa` - O(G·P·k·m·min(m,k)) ~5000 evals

---

## Parameter Reference

### Common Parameters

- `start_col`: Force first column selection (Algorithms 1, 3, 9, 10, 14)
- `n_dipoles_max`: Maximum dipoles to select (multiple)
- `rho_max`: Maximum correlation threshold (Algorithms 8, 10)
- `kappa_max`: Maximum condition number (Algorithm 4)
- `target_kappa`: Target condition number stop condition (Algorithms 5, 11)

### Algorithm-Specific

- **Algorithm 1**: `eps_rel` - Relative independence threshold
- **Algorithm 3**: `ensure_count`, `min_variance_explained` - Exact count mode
- **Algorithm 5**: `delta_threshold` - Relative κ improvement threshold
- **Algorithm 6**: `regularization` - Ridge regularization (1e-12 to 1e-4)
- **Algorithm 8**: `selection_order` - Mode: 'greedy' or 'energy_sorted'
- **Algorithm 9**: `eps_abs`, `min_residual_ratio` - Stop conditions
- **Algorithm 10**: `selection_order` - Mode: 'greedy_min_corr' or 'energy_sorted'
- **Algorithm 11**: `n_dipoles_target` - Target size for backward elimination
- **Algorithm 12**: `r_keep` - Number of singular values for leverage
- **Algorithm 13**: `population_size`, `n_generations`, `mutation_rate`, `crossover_rate`, `fitness_mode`, `tournament_size`
- **Algorithm 14**: `alpha` - Weight for correlation term (0=pure κ, 1=pure ρ)

---

## Implementation Notes

### All Algorithms Include:

- ✅ Triad checking (forbidden electrode combinations)
- ✅ Shape validation via `validate_matrix_shapes()`
- ✅ CLI with argparse
- ✅ Result saving to `run/results/{algorithm_name}/`
- ✅ README with goal, use cases, parameters, recipe, complexity, examples

### Output Structure:

```python
{
    'S': np.ndarray,              # Selection matrix (36, k)
    'selected_dipoles': list,     # List of (electrode_i, electrode_j) tuples
    'selected_indices': list,     # Column indices into F matrix
    'n_selected': int,            # Number of selected dipoles
    'condition_number': float,    # Final κ
    'algorithm': str,             # Algorithm ID
    'parameters': dict,           # All parameters used
    # Algorithm-specific fields...
}
```

---

## Changes from Original (Phase 1 & 2)

### Deleted:

- ❌ Algorithm D (`algorithm_D_low_correlation`) - Manual correlation implementation

### Added (Phase 1):

- ✅ `validate_matrix_shapes()` in `common.py`
- ✅ Enhanced parameters across 8 algorithms

### Renamed (Phase 2):

- A → `greedy_energy_qr_independence`
- B → `greedy_energy_simple`
- C → `svd_leverage_topk`
- E → `greedy_energy_condition_cap`
- F → `greedy_min_condition`
- G → `greedy_max_logdet`
- H → `rrqr_importance_ranking`
- I → `rrqr_correlation_optimize`

### Created (Phase 3):

- ⭐ Algorithm J → `greedy_max_independence`
- ⭐ Algorithm K → `greedy_low_correlation`
- ⭐ Algorithm L → `backward_eliminate_kappa`
- ⭐ Algorithm M → `genetic_optimize_kappa`
- ⭐ Algorithm N → `greedy_joint_corr_kappa`
- ⭐ Algorithm O → `backward_eliminate_leverage`

---

## Testing

Run all algorithms:

```bash
bash src/selection_algorithms/test_all_algorithms.sh
```

Run specific algorithm:

```bash
python -m src.selection_algorithms.{algorithm_folder}.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path run/f_matrix/B_matrix.npy \
    [algorithm-specific parameters]
```

---

## Next Steps

1. **Benchmarking**: Run all 14 algorithms, compare κ, correlation, runtime
2. **Parameter tuning**: Sweep key parameters (alpha, rho_max, kappa_max, etc.)
3. **Validation**: Test on real probe data
4. **Documentation**: Update main README with algorithm summary
5. **Integration**: Add to `run_all_variants.sh` and `rank_results.py`
