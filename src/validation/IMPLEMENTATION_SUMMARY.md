# Validation Framework - Implementation Summary

## ✅ Completed Implementation

### Core Modules (All Created)

1. **`generate_test_cases.py`** (190 lines)
   - Generates realistic antenna voltage test cases
   - 4 test types: single antenna, dipole pairs, uniform random, sparse random
   - Enforces `test_case_NNN` naming convention
   - Output: CSV with `case,v_1,...,v_9` columns
   - Configurable range (default: 1-50 mV)

2. **`comparison.py`** (220 lines)
   - `load_solver_probes()`: Load ground truth from solver CSV
   - `compute_full_forward()`: F @ B @ V_antenna (all 36 dipoles)
   - `compute_subset_forward()`: F @ S @ V_antenna (selected dipoles)
   - `compute_errors()`: RMSE, max, mean, relative errors
   - `compute_regional_errors()`: Breakdown by EEG 10-20 regions
   - `compare_all_methods()`: Three-way comparison

3. **`plotting.py`** (350+ lines)
   - All PNG graphs with direct labels (no Plotly mixing)
   - Per-case graphs:
     * `case_NNN_probe_comparison.png`: Side-by-side bars
     * `case_NNN_errors.png`: Error magnitude per probe
     * `case_NNN_scatter.png`: Solver vs subset correlation
   - Per-S-matrix aggregate:
     * `overview_rmse_bars.png`: 3-way RMSE comparison
     * `regional_heatmap.png`: Regions × Cases heatmap
   - Algorithm-level:
     * `algorithm_comparison_boxplot.png`: RMSE distributions
     * `best_per_algorithm_bars.png`: Best performance ranking

4. **`report_generator.py`** (240 lines)
   - `generate_summary_csv()`: All S matrices ranked by RMSE
   - `generate_s_matrix_report()`: Markdown report with:
     * Overall metrics table
     * Degradation analysis (F_subset vs F_full)
     * Regional breakdown
     * Per-case summary
     * Recommendations
   - `generate_case_detail_csv()`: Probe-by-probe data
   - `generate_combined_test_cases_csv()`: All data in one CSV
   - Full precision (%.15e) for programmatic analysis

5. **`validate_comprehensive.py`** (500+ lines)
   - Main orchestrator script
   - Creates `validation_solver_runs/` directory
   - Copies mesh, probes, test cases
   - Runs FEM solver (`run_step_03.sh`)
   - Extracts probe measurements (`run_probing.sh`)
   - Loads F, B matrices and all S matrices
   - For each S matrix and test case:
     * Computes F_full and F_subset
     * Loads solver ground truth
     * Compares all three methods
     * Generates graphs and reports
   - Command-line interface:
     * `--test-cases`: Required input CSV
     * `--validate-top-n`: Optional filter
     * `--n-parallel`: Solver parallelization
     * `--skip-solver`: Use existing results

6. **`README.md`**
   - Short and to the point (as requested)
   - Quick start guide
   - Output structure
   - What to look for in results
   - Troubleshooting

7. **`__init__.py`**
   - Makes validation module importable
   - Exports comparison, plotting, report_generator

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    Validation Workflow                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Generate test cases                                     │
│     └─> test_cases.csv (15-20 cases)                       │
│                                                              │
│  2. Run FEM solver (ONCE)                                   │
│     └─> validation_solver_runs/cases/probes_*.csv          │
│                                                              │
│  3. For each S matrix (73 in your case):                   │
│     ├─> Load S matrix                                       │
│     ├─> For each test case:                                 │
│     │   ├─> Compute F_full = F @ B @ V_antenna            │
│     │   ├─> Compute F_subset = F @ S @ V_antenna          │
│     │   ├─> Load V_solver from probes CSV                 │
│     │   └─> Compare all three                              │
│     │                                                        │
│     └─> Generate outputs:                                   │
│         ├─> graphs/ (PNG with direct labels)              │
│         ├─> per_case/ (detailed CSVs)                     │
│         ├─> report.md (human-readable)                    │
│         └─> test_cases_data.csv (combined)                │
│                                                              │
│  4. Generate summary reports:                               │
│     ├─> summary_all.csv (all S matrices ranked)           │
│     ├─> algorithm_comparison_boxplot.png                  │
│     └─> best_per_algorithm_bars.png                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions (User-Approved)

1. **Three-way comparison**: SOLVER (truth) vs F_FULL (baseline) vs F_SUBSET (selection)
2. **Solver runs**: ONCE per test case, not per S matrix (15 runs, not 1,095)
3. **Graph format**: All PNG with direct labels (no HTML/Plotly mixing)
4. **CSV precision**: %.15e format (no rounding) for post-analysis
5. **Test case format**: Enforce `test_case_NNN` naming (no description column)
6. **Output location**: `results/validation_results/algorithm_X/S_matrix_name/`
7. **Validation scope**: All S matrices by default, optional `--validate-top-n`
8. **Conda activation**: Manual by user (not programmatic in script)

## Runtime Estimates

- Test case generation: < 1 second
- Solver execution: ~10 minutes (15 cases, 8 parallel processes)
- Probe extraction: ~1 minute
- Comparison + plotting: ~30 seconds per S matrix
- **Total for 73 S matrices: ~45 minutes**

## Output Structure

```
results/validation_results/
├── summary_all.csv                           # Ranked list
├── algorithm_comparison_boxplot.png          # Overall comparison
├── best_per_algorithm_bars.png               # Best selections
│
├── greedy_selection/
│   ├── S_greedy_k12/
│   │   ├── report.md
│   │   ├── test_cases_data.csv
│   │   ├── graphs/
│   │   │   ├── overview_rmse_bars.png
│   │   │   ├── regional_heatmap.png
│   │   │   ├── test_case_001_probe_comparison.png
│   │   │   ├── test_case_001_errors.png
│   │   │   ├── test_case_001_scatter.png
│   │   │   └── ...
│   │   └── per_case/
│   │       ├── test_case_001.csv
│   │       └── ...
│   └── S_greedy_k24/
│       └── ...
│
├── random_selection/
│   └── ...
│
└── other_algorithms/
    └── ...
```

## Usage Examples

### Basic validation
```bash
# Generate test cases
python src/validation/generate_test_cases.py \
  --n-cases 15 \
  --v-range 0.001,0.050 \
  --output test_cases.csv

# Run validation
python src/validation/validate_comprehensive.py \
  --test-cases test_cases.csv \
  --n-parallel 8
```

### Validate only top performers
```bash
python src/validation/validate_comprehensive.py \
  --test-cases test_cases.csv \
  --validate-top-n 10 \
  --n-parallel 8
```

### Skip solver (use existing results)
```bash
python src/validation/validate_comprehensive.py \
  --test-cases test_cases.csv \
  --skip-solver
```

## Interpretation Guide

### In `summary_all.csv`
- Sort by `solver_vs_subset_rmse` (ascending = better)
- Low RMSE (<5 μV) = excellent
- Compare to `solver_vs_full_rmse` (baseline)
- `relative_rmse` shows percentage error

### In `report.md`
- **Degradation**: (RMSE_subset - RMSE_full) / RMSE_full × 100%
  * <10%: Excellent selection
  * 10-30%: Good selection
  * 30-50%: Moderate selection
  * >50%: Poor selection
- **Recommendations**: Based on RMSE and degradation

### In Graphs
- **Scatter plot**: Points near diagonal = accurate
- **Regional heatmap**: Color intensity = error magnitude
- **Bar charts**: Direct visual comparison of three methods

## Next Steps

To use the framework:

1. **Activate conda environment**:
   ```bash
   conda activate opendihu
   ```

2. **Generate test cases** (if needed):
   ```bash
   python src/validation/generate_test_cases.py --output test_cases.csv
   ```

3. **Run validation**:
   ```bash
   python src/validation/validate_comprehensive.py \
     --test-cases test_cases.csv \
     --n-parallel 8
   ```

4. **Review results**:
   - Start with `results/validation_results/summary_all.csv`
   - Check algorithm comparison graphs
   - Dive into individual S matrix reports for details

## Files You Can Delete (Obsolete)

- `validate_selections.py` (old version, mathematical only)
- `validate_with_solver.py` (intermediate version)
- `plot_validation_results.py` (replaced by plotting.py)

The new framework (`validate_comprehensive.py`) replaces all of these.
