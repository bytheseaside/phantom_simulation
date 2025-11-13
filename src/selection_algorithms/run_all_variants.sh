#!/bin/bash

# Check if RUN_DIR is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <RUN_DIR>"
    echo "Example: $0 run"
    exit 1
fi

RUN_DIR="$1"

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Directory $RUN_DIR does not exist"
    exit 1
fi

# Define matrix paths
F_MATRIX="$RUN_DIR/f_matrix/F_matrix.npy"
B_MATRIX="src/model/base_matrices/B_matrix.npy"
W_MATRIX="src/model/base_matrices/W_matrix.npy"

# Check if matrices exist
if [ ! -f "$F_MATRIX" ]; then
    echo "Error: F matrix not found at $F_MATRIX"
    exit 1
fi

if [ ! -f "$B_MATRIX" ]; then
    echo "Error: B matrix not found at $B_MATRIX"
    exit 1
fi

if [ ! -f "$W_MATRIX" ]; then
    echo "Error: W matrix not found at $W_MATRIX"
    exit 1
fi

echo "=========================================="
echo "Running all algorithm variants"
echo "Run directory: $RUN_DIR"
echo "F matrix: $F_MATRIX"
echo "B matrix: $B_MATRIX"
echo "W matrix: $W_MATRIX"
echo "=========================================="
echo ""

# Calculate test cases per algorithm
TOTAL_CASES=$(grep -c "python3 -m src.selection_algorithms" "$0")
CASES_A=$(grep -c "python3 -m src.selection_algorithms.algorithm_A" "$0")
CASES_B=$(grep -c "python3 -m src.selection_algorithms.algorithm_B" "$0") 
CASES_C=$(grep -c "python3 -m src.selection_algorithms.algorithm_C" "$0")
CASES_D=$(grep -c "python3 -m src.selection_algorithms.algorithm_D" "$0")
CASES_E=$(grep -c "python3 -m src.selection_algorithms.algorithm_E" "$0")
CASES_F=$(grep -c "python3 -m src.selection_algorithms.algorithm_F" "$0")
CASES_G=$(grep -c "python3 -m src.selection_algorithms.algorithm_G" "$0")
CASES_H=$(grep -c "python3 -m src.selection_algorithms.algorithm_H" "$0")
CASES_I=$(grep -c "python3 -m src.selection_algorithms.algorithm_I" "$0")

echo "📊 TEST SUITE SUMMARY"
echo "─────────────────────────────────────────"
echo "Algorithm A (Linear Independence): $CASES_A cases"
echo "Algorithm B (Greedy by Norm): $CASES_B case(s)"
echo "Algorithm C (SVD Leverage): $CASES_C cases"
echo "Algorithm D (Low Correlation): $CASES_D cases"
echo "Algorithm E (Condition Cap): $CASES_E cases"
echo "Algorithm F (Min Condition): $CASES_F case(s)"
echo "Algorithm G (D-Optimal): $CASES_G cases"
echo "Algorithm H (RRQR): $CASES_H cases"
echo "Algorithm I (Hybrid Greedy→SVD): $CASES_I cases"
echo "─────────────────────────────────────────"
echo "TOTAL: $TOTAL_CASES test cases"
echo "=========================================="
echo ""

echo "=== Algorithm A: Linear Independence ==="
python3 -m src.selection_algorithms.greedy_energy_qr_independence.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --eps-rel 1e-8
python3 -m src.selection_algorithms.greedy_energy_qr_independence.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --eps-rel 1e-2
python3 -m src.selection_algorithms.greedy_energy_qr_independence.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --eps-rel 1e-1
python3 -m src.selection_algorithms.greedy_energy_qr_independence.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --eps-rel 15e-2
python3 -m src.selection_algorithms.greedy_energy_qr_independence.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --eps-rel 2e-1
python3 -m src.selection_algorithms.greedy_energy_qr_independence.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --eps-rel 25e-2
python3 -m src.selection_algorithms.greedy_energy_qr_independence.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --eps-rel 3e-1
echo ""

# Algorithm B - baseline (no params)
echo "=== Algorithm B: Greedy by Norm ==="
python3 -m src.selection_algorithms.greedy_energy_simple.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX"
echo ""

# Algorithm C - SVD leverage: comprehensive cross-product coverage
echo "=== Algorithm C: SVD Leverage (64 cases) ==="

echo "Mode 'all': r_keep × score_threshold (30 cases)..."
# r_keep=5
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 5 --selection-mode all --score-threshold 0.0
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 5 --selection-mode all --score-threshold 0.1
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 5 --selection-mode all --score-threshold 0.2

# r_keep=7
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 7 --selection-mode all --score-threshold 0.0
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 7 --selection-mode all --score-threshold 0.1
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 7 --selection-mode all --score-threshold 0.2

# r_keep=8
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode all --score-threshold 0.0
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode all --score-threshold 0.05
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode all --score-threshold 0.1
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode all --score-threshold 0.15
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode all --score-threshold 0.2

# r_keep=10 (comprehensive)
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode all --score-threshold 0.0
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode all --score-threshold 0.05
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode all --score-threshold 0.08
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode all --score-threshold 0.1
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode all --score-threshold 0.15
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode all --score-threshold 0.2

# r_keep=12
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 12 --selection-mode all --score-threshold 0.0
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 12 --selection-mode all --score-threshold 0.08
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 12 --selection-mode all --score-threshold 0.15
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 12 --selection-mode all --score-threshold 0.2

# r_keep=15
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 15 --selection-mode all --score-threshold 0.0
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 15 --selection-mode all --score-threshold 0.1
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 15 --selection-mode all --score-threshold 0.2

# r_keep=18
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 18 --selection-mode all --score-threshold 0.0
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 18 --selection-mode all --score-threshold 0.15

# r_keep=21
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 21 --selection-mode all --score-threshold 0.0
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 21 --selection-mode all --score-threshold 0.15

echo "Mode 'top_n': r_keep × n_dipoles_max (24 cases)..."
# r_keep=5
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 5 --selection-mode top_n --n-dipoles-max 10
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 5 --selection-mode top_n --n-dipoles-max 14
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 5 --selection-mode top_n --n-dipoles-max 18
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 5 --selection-mode top_n --n-dipoles-max 22

# r_keep=8
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode top_n --n-dipoles-max 10
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode top_n --n-dipoles-max 12
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode top_n --n-dipoles-max 16
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode top_n --n-dipoles-max 20

# r_keep=10
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode top_n --n-dipoles-max 10
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode top_n --n-dipoles-max 12
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode top_n --n-dipoles-max 15
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode top_n --n-dipoles-max 18
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode top_n --n-dipoles-max 20

# r_keep=12
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 12 --selection-mode top_n --n-dipoles-max 12
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 12 --selection-mode top_n --n-dipoles-max 16
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 12 --selection-mode top_n --n-dipoles-max 20

# r_keep=15
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 15 --selection-mode top_n --n-dipoles-max 14
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 15 --selection-mode top_n --n-dipoles-max 18
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 15 --selection-mode top_n --n-dipoles-max 22

# r_keep=18
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 18 --selection-mode top_n --n-dipoles-max 16
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 18 --selection-mode top_n --n-dipoles-max 20

echo "Mode 'threshold': strategic combinations (10 cases)..."
# r_keep=8
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode threshold --score-threshold 0.1
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 8 --selection-mode threshold --score-threshold 0.2

# r_keep=10
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode threshold --score-threshold 0.05
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode threshold --score-threshold 0.15
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 10 --selection-mode threshold --score-threshold 0.25

# r_keep=12
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 12 --selection-mode threshold --score-threshold 0.1
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 12 --selection-mode threshold --score-threshold 0.2
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 12 --selection-mode threshold --score-threshold 0.3

# r_keep=15
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 15 --selection-mode threshold --score-threshold 0.15
python3 -m src.selection_algorithms.svd_leverage_topk.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --r-keep 15 --selection-mode threshold --score-threshold 0.25
echo ""

# Algorithm D - correlation: very strict to very lax
echo "=== Algorithm D: Low Correlation ==="
echo "Very strict (low correlation allowed)..."
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.1
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.15
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.2

echo "Medium..."
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.25
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.3
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.35
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.4

echo "Lax (high correlation allowed)..."
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.5
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.6
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.7
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.8
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --rho-max 0.9
echo ""

# Algorithm E - condition cap: very strict to very lax
echo "=== Algorithm E: Condition Cap ==="
echo "Very strict (low condition allowed)..."
python3 -m src.selection_algorithms.greedy_energy_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --kappa-max 100
python3 -m src.selection_algorithms.greedy_energy_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --kappa-max 500
python3 -m src.selection_algorithms.greedy_energy_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --kappa-max 1e3

echo "Medium..."
python3 -m src.selection_algorithms.greedy_energy_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --kappa-max 5e3
python3 -m src.selection_algorithms.greedy_energy_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --kappa-max 1e4
python3 -m src.selection_algorithms.greedy_energy_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --kappa-max 5e4

echo "Lax (high condition allowed)..."
python3 -m src.selection_algorithms.greedy_energy_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --kappa-max 1e5
python3 -m src.selection_algorithms.greedy_energy_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --kappa-max 5e5
python3 -m src.selection_algorithms.greedy_energy_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --kappa-max 1e6
python3 -m src.selection_algorithms.greedy_energy_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --kappa-max 1e7
echo ""

# Algorithm F - no params
echo "=== Algorithm F: Min Condition ==="
python3 -m src.selection_algorithms.greedy_min_condition.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX"
echo ""

# Algorithm G - regularization: minimal to strong 
echo "=== Algorithm G: D-Optimal ==="
echo "Minimal regularization..."
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 1e-10
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 1e-9
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 1e-8

echo "Light regularization..."
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 5e-8
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 1e-7
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 5e-7

echo "Medium regularization..."
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 1e-6
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 5e-6
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 1e-5

echo "Strong regularization..."
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 5e-5
python3 -m src.selection_algorithms.greedy_max_logdet.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --regularization 1e-4
echo ""

# Algorithm H - RRQR: different max dipoles (10 cases)
echo "=== Algorithm H: RRQR (10 cases) ==="
echo "Extremes and full range..."
python3 -m src.selection_algorithms.rrqr_importance_ranking.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-dipoles-max 8
python3 -m src.selection_algorithms.rrqr_importance_ranking.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-dipoles-max 10
python3 -m src.selection_algorithms.rrqr_importance_ranking.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-dipoles-max 12
python3 -m src.selection_algorithms.rrqr_importance_ranking.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-dipoles-max 14
python3 -m src.selection_algorithms.rrqr_importance_ranking.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-dipoles-max 16
python3 -m src.selection_algorithms.rrqr_importance_ranking.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-dipoles-max 18
python3 -m src.selection_algorithms.rrqr_importance_ranking.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-dipoles-max 20
python3 -m src.selection_algorithms.rrqr_importance_ranking.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-dipoles-max 22
python3 -m src.selection_algorithms.rrqr_importance_ranking.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-dipoles-max 25
python3 -m src.selection_algorithms.rrqr_importance_ranking.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-dipoles-max 28
echo ""

# Algorithm I - Hybrid: comprehensive 3D parameter space (90 cases)
echo "=== Algorithm I: Hybrid Greedy→SVD (90 cases) ==="

echo "Strategy 1: Fixed n_preselect, vary n_final × r_keep..."
# n_preselect=15 (small) - 9 cases
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 15 --n-final 8 --r-keep 5
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 15 --n-final 8 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 15 --n-final 8 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 15 --n-final 10 --r-keep 5
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 15 --n-final 10 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 15 --n-final 10 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 15 --n-final 12 --r-keep 5
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 15 --n-final 12 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 15 --n-final 12 --r-keep 8

# n_preselect=20 (medium) - comprehensive 18 cases
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 10 --r-keep 5
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 10 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 10 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 12 --r-keep 5
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 12 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 12 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 12 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 14 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 14 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 14 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 16 --r-keep 5
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 16 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 16 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 16 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 16 --r-keep 15
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 18 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 18 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 20 --n-final 18 --r-keep 15

# n_preselect=25 (large) - 13 cases
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 12 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 12 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 15 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 15 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 18 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 18 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 18 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 18 --r-keep 15
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 20 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 20 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 20 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 20 --r-keep 15
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 20 --r-keep 18
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 22 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 22 --r-keep 15
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 22 --r-keep 18

echo "Strategy 2: Fixed n_final=16, vary n_preselect × r_keep (25 cases)..."
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 16 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 16 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 16 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 16 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 16 --r-keep 15
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 16 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 16 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 16 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 16 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 16 --r-keep 15
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 16 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 16 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 16 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 16 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 25 --n-final 16 --r-keep 15
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 16 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 16 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 16 --r-keep 10
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 16 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 16 --r-keep 15

echo "Strategy 3: Diagonal balanced & edge cases (30 cases)..."
# Diagonal
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 12 --n-final 10 --r-keep 5
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 15 --n-final 12 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 14 --r-keep 8
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 18 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 22 --r-keep 18

# Edge cases
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 12 --n-final 8 --r-keep 5
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 8 --r-keep 5
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 22 --r-keep 5
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 12 --n-final 10 --r-keep 18
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 22 --r-keep 18

# Additional strategic 3D samples
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 10 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 10 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 10 --r-keep 18
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 14 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 14 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 14 --r-keep 18
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 18 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 18 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 18 --n-final 18 --r-keep 18
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 10 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 10 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 10 --r-keep 18
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 14 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 14 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 14 --r-keep 18
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 18 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 18 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 22 --n-final 18 --r-keep 18
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 10 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 10 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 10 --r-keep 18
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 14 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 14 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 14 --r-keep 18
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 18 --r-keep 7
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 18 --r-keep 12
python3 -m src.selection_algorithms.rrqr_correlation_optimize.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --n-preselect 28 --n-final 18 --r-keep 18
echo ""

echo "📊 TEST SUITE COMPLETE"
echo "─────────────────────────────────────────"
echo "Algorithm A (Linear Independence): $CASES_A cases"
echo "Algorithm B (Greedy by Norm): $CASES_B case(s)"
echo "Algorithm C (SVD Leverage): $CASES_C cases"
echo "Algorithm D (Low Correlation): $CASES_D cases"
echo "Algorithm E (Condition Cap): $CASES_E cases"
echo "Algorithm F (Min Condition): $CASES_F case(s)"
echo "Algorithm G (D-Optimal): $CASES_G cases"
echo "Algorithm H (RRQR): $CASES_H cases"
echo "Algorithm I (Hybrid Greedy→SVD): $CASES_I cases"
echo "─────────────────────────────────────────"
echo "TOTAL: $TOTAL_CASES test cases"
echo "=========================================="
echo ""