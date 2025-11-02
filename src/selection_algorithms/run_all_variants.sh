#!/bin/bash

echo "=========================================="
echo "Running all algorithms with varied params"
echo "=========================================="
echo ""

# Algorithm A - different eps_rel values
echo "=== Algorithm A: Linear Independence ==="
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 1e-2
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 1e-3
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 5e-3
echo ""

# Algorithm B - baseline (no params)
echo "=== Algorithm B: Greedy by Norm ==="
python3 -m src.selection_algorithms.algorithm_B_greedy_norm.select
echo ""

# Algorithm C - different modes and r_keep
echo "=== Algorithm C: SVD Leverage ==="
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 21
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --selection-mode top_n
echo ""

# Algorithm D - different rho_max
echo "=== Algorithm D: Low Correlation ==="
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.3
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.25
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.2
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.15
echo ""

# Algorithm E - different kappa_max
echo "=== Algorithm E: Condition Cap ==="
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e4
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e3
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e5
echo ""

# Algorithm F - no params
echo "=== Algorithm F: Min Condition ==="
python3 -m src.selection_algorithms.algorithm_F_min_condition.select
echo ""

# Algorithm G - different regularization
echo "=== Algorithm G: D-Optimal ==="
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-8
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-6
echo ""

echo "=========================================="
echo "All variants completed!"
echo "=========================================="
