# Dipole Selection Validation

Validates dipole selections against **real FEM solver simulations**.

## What it does

Compares three methods for computing EEG probe voltages:
- **SOLVER**: FEM simulation (ground truth)
- **F_FULL**: Forward model with all 36 dipoles
- **F_SUBSET**: Forward model with selected dipoles only

Tells you if your dipole selection actually works in practice.

## Quick Start

### 1. Generate test cases

```bash
python src/validation/generate_test_cases.py \
  --n-cases 15 \
  --v-range 0.001,0.050 \
  --output test_cases.csv
```

Creates realistic antenna voltage patterns (1-50 mV range).

### 2. Run validation

```bash
python src/validation/validate.py \
  --test-cases test_cases.csv \
  --n-parallel 8
```

Runs solver once per test case (~10 min), then compares against all S matrices (fast).

Optional: `--validate-top-n 10` to only test top 10 selections per algorithm.

### 3. Check results

```
results/validation_results/
├── summary_all.csv                     ← All S matrices ranked
├── algorithm_comparison_boxplot.png    ← Algorithm performance
├── best_per_algorithm_bars.png         ← Best from each algorithm
└── algorithm_X/
    └── S_matrix_name/
        ├── report.md                   ← Human-readable summary
        ├── graphs/
        │   ├── overview_rmse_bars.png
        │   ├── regional_heatmap.png
        │   └── case_NNN_*.png
        └── per_case/
            └── case_NNN.csv            ← Detailed probe data
```

## What to look for

**In `summary_all.csv`:**
- Low `solver_vs_subset_rmse` = good selection
- Compare to `solver_vs_full_rmse` (baseline accuracy)

**In `report.md`:**
- ✅ Degradation <20% = excellent
- ⚠️ Degradation 20-50% = acceptable
- ❌ Degradation >50% = reconsider

**In graphs:**
- Scatter plot near diagonal = accurate
- Regional heatmap shows spatial errors
- Bar charts compare all three methods

## Requirements

- Conda environment activated (`conda activate opendihu`)
- FEM solver scripts in `src/steps/`
- Mesh and probes in `run/`
- F and B matrices in `results/`
- S matrices in `results/algorithm_X/`

## Notes

- Test cases run ONCE by solver, not per S matrix
- 15 test cases → 15 solver runs + 1,095 comparisons (if 73 S matrices)
- CSV data at full precision (%.15e) for post-analysis
- All graphs PNG with direct labels

## Troubleshooting

**"Mesh file not found"**: Run Step 1 (meshing) first  
**"F matrix not found"**: Run Step 2 (forward model) first  
**"No S matrices found"**: Run selection algorithms first  
**Solver fails**: Check conda env and run scripts manually to debug
