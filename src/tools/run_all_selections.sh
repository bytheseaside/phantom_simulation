#!/usr/bin/env bash
#
# Run all column selection algorithms on 9DOF and 18DOF F matrices.
#
# Usage:
#   ./run_all_selections.sh
#
# Outputs are saved to:
#   run_phantom/9dof/selection/{algorithm}/
#   run_phantom/18dof/selection/{algorithm}/
#

set -euo pipefail

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Python script path
SELECT_SCRIPT="$SCRIPT_DIR/selection_alg.py"

# Algorithms to run
ALGORITHMS=(
    "condition"
    "correlation"
    "coherence"
    "qr"
    "residual"
    "energy"
    "hybrid"
    "leverage"
)

# Names for each DOF configuration - order matters! - should be the same as in the F matrices
NAMES_9DOF="E1 E2 E3 E4 E5 E6 E7 E8 E9"

# 18DOF: lexicographic order (E1_R E1_T E2_R E2_T ...)
NAMES_18DOF="E1_R E1_T E2_R E2_T E3_R E3_T E4_R E4_T E5_R E5_T E6_R E6_T E7_R E7_T E8_R E8_T E9_R E9_T"

echo "============================================================"
echo "Column Selection: Running all algorithms"
echo "============================================================"
echo "Algorithms: ${ALGORITHMS[*]}"
echo ""

# --- 9 DOF ---
MATRIX_9DOF="$PROJECT_ROOT/run_phantom/9dof/F.npy"
if [[ -f "$MATRIX_9DOF" ]]; then
    echo ">>> 9 DOF Matrix: $MATRIX_9DOF"
    echo ""
    
    for algo in "${ALGORITHMS[@]}"; do
        OUT_DIR="$PROJECT_ROOT/run_phantom/9dof/selection/$algo"
        echo "  Running $algo..."
        python3 "$SELECT_SCRIPT" \
            --matrix "$MATRIX_9DOF" \
            --names $NAMES_9DOF \
            --algorithm "$algo" \
            --min-cols "8" \
            --out "$OUT_DIR" \
            2>&1 | grep -E "(Selected|Removed|condition_number|Saved)" | sed 's/^/    /'
        echo ""
    done
else
    echo "WARNING: 9DOF matrix not found: $MATRIX_9DOF"
fi

# --- 18 DOF ---
MATRIX_18DOF="$PROJECT_ROOT/run_phantom/18dof/F.npy"
if [[ -f "$MATRIX_18DOF" ]]; then
    echo ">>> 18 DOF Matrix: $MATRIX_18DOF"
    echo ""
    
    for algo in "${ALGORITHMS[@]}"; do
        OUT_DIR="$PROJECT_ROOT/run_phantom/18dof/selection/$algo"
        echo "  Running $algo..."
        python3 "$SELECT_SCRIPT" \
            --matrix "$MATRIX_18DOF" \
            --names $NAMES_18DOF \
            --algorithm "$algo" \
            --min-cols "12" \
            --out "$OUT_DIR" \
            2>&1 | grep -E "(Selected|Removed|condition_number|Saved)" | sed 's/^/    /'
        echo ""
    done
else
    echo "WARNING: 18DOF matrix not found: $MATRIX_18DOF"
fi

echo "============================================================"
echo "Done! Results saved to:"
echo "  run_phantom/9dof/selection/{algorithm}/"
echo "  run_phantom/18dof/selection/{algorithm}/"
echo "============================================================"
