# Algorithm M: Genetic Algorithm Optimization

**Population-based evolutionary search for global optimization of multiple fitness criteria.**

---

## Goal

Use genetic algorithm to evolve optimal dipole selections by:
- Maintaining a population of candidate solutions
- Evolving via selection, crossover, mutation
- Optimizing toward one of three fitness modes:
  * **kappa**: Minimize condition number κ(F·B·S)
  * **logdet**: Maximize log-determinant log|det(G^T·G)|
  * **correlation**: Minimize maximum pairwise correlation

Unlike greedy algorithms, GA can:
- Escape local optima via mutation
- Explore solution space via crossover
- Balance exploration vs exploitation
- Find globally competitive solutions

---

## Use Cases

1. **Global optimization**: When greedy algorithms get stuck in local optima
2. **Multiple fitness modes**: Test different optimization criteria
3. **Computational budget**: When you can afford 100+ generations
4. **Benchmarking**: Compare greedy vs evolutionary approaches
5. **Research**: Study fitness landscape structure

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `population_size` | int | 50 | Number of individuals in population |
| `n_generations` | int | 100 | Number of generations to evolve |
| `mutation_rate` | float | 0.1 | Probability of mutating each gene |
| `crossover_rate` | float | 0.8 | Probability of performing crossover |
| `fitness_mode` | str | 'kappa' | Fitness: 'kappa', 'logdet', 'correlation' |
| `n_dipoles_target` | int | 18 | Target number of dipoles |
| `tournament_size` | int | 3 | Tournament size for selection |
| `random_seed` | int | None | Seed for reproducibility |

---

## Recipe

1. **Initialization**:
   - Create `population_size` random valid selections (no triads)
   - Evaluate fitness for each individual

2. **Evolution loop** (repeat for `n_generations`):
   - **Selection**: Tournament selection picks parents
   - **Crossover**: Uniform crossover (prob `crossover_rate`)
   - **Mutation**: Replace genes with prob `mutation_rate`
   - **Validation**: Reject if creates triads
   - **Elitism**: Best individual always survives

3. **Output**: Return best individual found

---

## Complexity

- **Time**: O(G·P·k·m·min(m,k)) 
  - G = generations (e.g., 100)
  - P = population size (e.g., 50)
  - k = dipoles (18), m = probes (21)
  - Each fitness evaluation: O(k·m·min(m,k)) for SVD
  - **Total**: ~5000 fitness evaluations for default params
  
- **Space**: O(P·n) where n=36 dipoles
  - Population storage is modest

**Very expensive** compared to greedy (minutes vs seconds), but can find better solutions.

---

## Examples

### Optimize condition number (default)
```bash
python -m src.selection_algorithms.genetic_optimize_kappa.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path run/f_matrix/B_matrix.npy \
    --fitness-mode kappa \
    --n-generations 100 \
    --population-size 50
```

### Optimize log-determinant (longer run)
```bash
python -m src.selection_algorithms.genetic_optimize_kappa.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path run/f_matrix/B_matrix.npy \
    --fitness-mode logdet \
    --population-size 100 \
    --n-generations 200
```

### Minimize correlation (reproducible)
```bash
python -m src.selection_algorithms.genetic_optimize_kappa.select \
    --run-dir run \
    --f-matrix-path run/f_matrix/F_matrix.npy \
    --b-matrix-path run/f_matrix/B_matrix.npy \
    --fitness-mode correlation \
    --mutation-rate 0.15 \
    --random-seed 42
```

---

## Notes

- **Computational cost**: Default params (50 pop, 100 gen) = ~5000 fitness evaluations
- **Convergence**: Track `fitness_history` - should improve over generations
- **Hyperparameters**: 
  - High mutation (0.2+): more exploration, slower convergence
  - Low mutation (0.05): faster convergence, risk of local optima
  - Population size: 50-100 is typical tradeoff
- **Reproducibility**: Use `--random-seed` for deterministic runs
- **Comparison**: Run multiple seeds, compare best-of-N vs greedy
- **Elitism**: Always keeps best solution (prevents regression)
