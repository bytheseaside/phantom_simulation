#!/usr/bin/env bash
#
# Run all column selection algorithms on F matrices.
#
# Usage:
#   ./run_all_selections.sh
#
# Outputs are saved to:
#   run_phantom/selection/{algorithm}/
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
NAMES="E1 E2 E3 E4 E5 E6 E7 E8 E9"

echo "============================================================"
echo "Column Selection: Running all algorithms"
echo "============================================================"
echo "Algorithms: ${ALGORITHMS[*]}"
echo ""
TARGET_COLS=8


MATRIX="$PROJECT_ROOT/run_phantom/full_F/F.npy"
if [[ -f "$MATRIX" ]]; then
    echo ">>> Matrix: $MATRIX"
    echo ""
    
    for algo in "${ALGORITHMS[@]}"; do
        OUT_DIR="$PROJECT_ROOT/run_phantom/selection_${TARGET_COLS}_cols/$algo"
        echo "  Running $algo..."
        python3 "$SELECT_SCRIPT" \
            --matrix "$MATRIX" \
            --names $NAMES \
            --algorithm "$algo" \
            --min-cols "$TARGET_COLS" \
            --out "$OUT_DIR" \
            2>&1 | grep -E "(Selected|Removed|condition_number|Saved)" | sed 's/^/    /'
        echo ""
    done
else
    echo "WARNING: matrix not found: $MATRIX"
fi

echo "============================================================"
echo "Done! Results saved to:"
echo "  run_phantom/selection_${TARGET_COLS}_cols/{algorithm}/"
echo "============================================================"
