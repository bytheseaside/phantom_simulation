#!/bin/bash

# Activate conda environment if not already active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate env
fi

# Default test paths
F_MATRIX="${F_MATRIX:-run/f_matrix/F_matrix.npy}"
B_MATRIX="${B_MATRIX:-src/model/base_matrices/B_matrix.npy}"
W_MATRIX="${W_MATRIX:-src/model/base_matrices/W_matrix.npy}"
RUN_DIR="${RUN_DIR:-run}"

echo "=========================================="
echo "Testing all 9 algorithms"
echo "F matrix: $F_MATRIX"
echo "B matrix: $B_MATRIX"
echo "W matrix: $W_MATRIX"
echo "Run dir: $RUN_DIR"
echo "=========================================="
echo ""

echo "1. Algorithm A (Linear Independence)..."
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm A PASSED"
else
    echo "✗ Algorithm A FAILED"
    exit 1
fi
echo ""

echo "2. Algorithm B (Greedy by Norm)..."
python3 -m src.selection_algorithms.algorithm_B_greedy_norm.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm B PASSED"
else
    echo "✗ Algorithm B FAILED"
    exit 1
fi
echo ""

echo "3. Algorithm C (SVD Leverage)..."
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm C PASSED"
else
    echo "✗ Algorithm C FAILED"
    exit 1
fi
echo ""

echo "4. Algorithm D (Low Correlation)..."
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm D PASSED"
else
    echo "✗ Algorithm D FAILED"
    exit 1
fi
echo ""

echo "5. Algorithm E (Condition Cap)..."
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm E PASSED"
else
    echo "✗ Algorithm E FAILED"
    exit 1
fi
echo ""

echo "6. Algorithm F (Min Condition)..."
python3 -m src.selection_algorithms.algorithm_F_min_condition.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm F PASSED"
else
    echo "✗ Algorithm F FAILED"
    exit 1
fi
echo ""

echo "7. Algorithm G (D-Optimal)..."
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm G PASSED"
else
    echo "✗ Algorithm G FAILED"
    exit 1
fi
echo ""

echo "8. Algorithm H (RRQR)..."
python3 -m src.selection_algorithms.algorithm_H_rrqr.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm H PASSED"
else
    echo "✗ Algorithm H FAILED"
    exit 1
fi
echo ""

echo "9. Algorithm I (Hybrid)..."
python3 -m src.selection_algorithms.algorithm_I_hybrid.select --run-dir "$RUN_DIR" --f-matrix-path "$F_MATRIX" --b-matrix-path "$B_MATRIX" --w-matrix-path "$W_MATRIX" --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm I PASSED"
else
    echo "✗ Algorithm I FAILED"
    exit 1
fi
echo ""

echo "=========================================="
echo "All 9 algorithms PASSED!"
echo "=========================================="


echo "1. Algorithm A (Linear Independence)..."
python3 -m src.selection_algorithms.algorithm_A_linear_independence.select --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm A PASSED"
else
    echo "✗ Algorithm A FAILED"
    exit 1
fi
echo ""

echo "2. Algorithm B (Greedy by Norm)..."
python3 -m src.selection_algorithms.algorithm_B_greedy_norm.select --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm B PASSED"
else
    echo "✗ Algorithm B FAILED"
    exit 1
fi
echo ""

echo "3. Algorithm C (SVD Leverage)..."
python3 -m src.selection_algorithms.algorithm_C_svd_leverage.select --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm C PASSED"
else
    echo "✗ Algorithm C FAILED"
    exit 1
fi
echo ""

echo "4. Algorithm D (Low Correlation)..."
python3 -m src.selection_algorithms.algorithm_D_low_correlation.select --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm D PASSED"
else
    echo "✗ Algorithm D FAILED"
    exit 1
fi
echo ""

echo "5. Algorithm E (Condition Cap)..."
python3 -m src.selection_algorithms.algorithm_E_condition_cap.select --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm E PASSED"
else
    echo "✗ Algorithm E FAILED"
    exit 1
fi
echo ""

echo "6. Algorithm F (Min Condition)..."
python3 -m src.selection_algorithms.algorithm_F_min_condition.select --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm F PASSED"
else
    echo "✗ Algorithm F FAILED"
    exit 1
fi
echo ""

echo "7. Algorithm G (D-Optimal)..."
python3 -m src.selection_algorithms.algorithm_G_d_optimal.select --no-save
if [ $? -eq 0 ]; then
    echo "✓ Algorithm G PASSED"
else
    echo "✗ Algorithm G FAILED"
    exit 1
fi
echo ""

echo "=========================================="
echo "All 7 algorithms PASSED!"
echo "=========================================="
