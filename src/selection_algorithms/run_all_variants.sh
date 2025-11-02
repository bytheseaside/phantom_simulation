#!/bin/bash

echo "=========================================="
echo "Running all algorithm variants"
echo "=========================================="
echo ""

echo "=== Algorithm A: Linear Independence ==="
echo "Very strict (small eps_rel)..."
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 1e-8
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 1e-7
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 1e-6

echo "Medium strict..."
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 1e-5
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 5e-5
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 1e-4

echo "Lax..."
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 5e-4
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 1e-3
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 5e-3
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --eps-rel 1e-2
echo ""

# Algorithm B - baseline (no params)
echo "=== Algorithm B: Greedy by Norm ==="
python3 -m src.selection_algorithms.algorithm_B_greedy_norm.select
echo ""

# Algorithm C - SVD leverage: r_keep and modes
echo "=== Algorithm C: SVD Leverage ==="
echo "Low r_keep (strict)..."
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 5 --selection-mode all
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 7 --selection-mode all
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 8 --selection-mode all

echo "Medium r_keep..."
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 10 --selection-mode all
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 12 --selection-mode all
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 15 --selection-mode all

echo "High r_keep (lax)..."
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 18 --selection-mode all
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 21 --selection-mode all

echo "Top-N mode with different thresholds..."
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 10 --selection-mode top_n --n-dipoles-max 10
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 10 --selection-mode top_n --n-dipoles-max 15
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 10 --selection-mode top_n --n-dipoles-max 20

echo "Different score thresholds..."
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 10 --selection-mode all --score-threshold 0.05
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 10 --selection-mode all --score-threshold 0.15
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --r-keep 10 --selection-mode all --score-threshold 0.2
echo ""

# Algorithm D - correlation: very strict to very lax
echo "=== Algorithm D: Low Correlation ==="
echo "Very strict (low correlation allowed)..."
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.1
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.15
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.2

echo "Medium..."
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.25
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.3
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.35
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.4

echo "Lax (high correlation allowed)..."
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.5
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.6
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.7
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.8
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --rho-max 0.9
echo ""

# Algorithm E - condition cap: very strict to very lax
echo "=== Algorithm E: Condition Cap ==="
echo "Very strict (low condition allowed)..."
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 100
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 500
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e3

echo "Medium..."
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 5e3
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e4
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 5e4

echo "Lax (high condition allowed)..."
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e5
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 5e5
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e6
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --kappa-max 1e7
echo ""

# Algorithm F - no params
echo "=== Algorithm F: Min Condition ==="
python3 -m src.selection_algorithms.algorithm_F_min_condition.select
echo ""

# Algorithm G - regularization: minimal to strong 
echo "=== Algorithm G: D-Optimal ==="
echo "Minimal regularization..."
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-10
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-9
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-8

echo "Light regularization..."
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 5e-8
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-7
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 5e-7

echo "Medium regularization..."
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-6
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 5e-6
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-5

echo "Strong regularization..."
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 5e-5
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --regularization 1e-4
echo ""

echo "=========================================="
echo "All cases completed!"
echo "=========================================="
