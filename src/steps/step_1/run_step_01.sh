#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_step_01.sh <path_to_step_file>
if [ $# -lt 1 ]; then
  echo "✗ Error: Missing STEP file path argument"
  echo "Usage: $0 <path_to_step_file>"
  exit 1
fi

STEP_FILE="$1"

# Check if file exists
if [ ! -f "$STEP_FILE" ]; then
  echo "✗ Error: File does not exist: $STEP_FILE"
  exit 1
fi

# Check if file has .step extension
if [[ "$STEP_FILE" != *.step ]]; then
  echo "✗ Error: File must have .step extension: $STEP_FILE"
  exit 1
fi

# Convert to absolute path
STEP_FILE="$(cd "$(dirname "$STEP_FILE")" && pwd)/$(basename "$STEP_FILE")"

TS="$(date +"%Y%m%d_%H%M%S")"
RUN_DIR="run_${TS}"
RUN_STEPS_DIR="$RUN_DIR/run_steps"
mkdir -p "$RUN_STEPS_DIR"
echo ">> Working dir: $RUN_DIR"

#  So the script can be run from anywhere
if [ -n "${BASH_SOURCE:-}" ]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
else
  SCRIPT_PATH="$0"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"


# Copy the frozen-IDs geo file from the script directory into the run_steps dir
cp "$SCRIPT_DIR/01_prep_freeze_ids.geo" "$RUN_STEPS_DIR"/
cd "$RUN_STEPS_DIR"


LOG="step1.log"
echo ">> Running Gmsh (prep model). Log: $LOG"
echo ">> Using STEP file: $STEP_FILE"

if ! STEP_FILE="$STEP_FILE" gmsh -v 5 -0 01_prep_freeze_ids.geo &> "$LOG"; then
  echo "✗ Error. See $RUN_STEPS_DIR/$LOG"
  exit 1
fi

# Gmsh may auto-save an unrolled of the input .geo; remove it to keep only our frozen artifacts
rm -f "01_prep_freeze_ids.geo" 2>/dev/null || true
rm -f "01_prep_freeze_ids.geo_unrolled.xao" 2>/dev/null || true


# Check artifacts: we only expect .geo_unrolled (+ .xao alongside)
if [ -f "prep.geo_unrolled" ] && [ -f "prep.geo_unrolled.xao" ]; then
  echo "✓ Prep OK. Artifacts:"
  echo "   - $RUN_STEPS_DIR/prep.geo_unrolled"
  echo "   - $RUN_STEPS_DIR/prep.geo_unrolled.xao"
  echo "   - $RUN_STEPS_DIR/$LOG"

  # Print instructions to check IDs
  echo "To check the IDs of the entities for STEP 2, run the following commands:"
  echo "cd $RUN_STEPS_DIR ; gmsh prep.geo_unrolled.xao"
else
  echo "✗ Frozen artifacts missing. Check $RUN_STEPS_DIR/$LOG"
  exit 1
fi
