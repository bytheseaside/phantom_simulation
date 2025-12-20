#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_step_01.sh --step <path_to_step_file> --out <path_to_output.xao>
STEP_FILE=""
OUT_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step)
      STEP_FILE="$2"
      shift 2
      ;;
    --out)
      OUT_FILE="$2"
      shift 2
      ;;
    *)
      echo "✗ Unknown option: $1"
      echo "Usage: $0 --step <path_to_step_file> --out <path_to_output.xao>"
      exit 1
      ;;
  esac
done

# Validate arguments
if [ -z "$STEP_FILE" ] || [ -z "$OUT_FILE" ]; then
  echo "✗ Error: Both --step and --out arguments required"
  echo "Usage: $0 --step <path_to_step_file> --out <path_to_output.xao>"
  exit 1
fi

# Validate STEP file exists and has .step extension
if [ ! -f "$STEP_FILE" ]; then
  echo "✗ Error: STEP file does not exist: $STEP_FILE"
  exit 1
fi

if [[ "$STEP_FILE" != *.step ]]; then
  echo "✗ Error: File must have .step extension: $STEP_FILE"
  exit 1
fi

# Convert to absolute paths
STEP_FILE="$(realpath "$STEP_FILE")"

# Ensure output directory exists and convert OUT_FILE to absolute path
OUT_DIR="$(dirname "$OUT_FILE")"
mkdir -p "$OUT_DIR"
OUT_FILE="$(realpath "$OUT_DIR")/$(basename "$OUT_FILE")"

# Get script directory
SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"

echo ">> Running gmsh prep (freeze IDs)..."
STEP_FILE="$STEP_FILE" OUT_PATH="$OUT_FILE" gmsh -v 4 - "$SCRIPT_DIR/01_prep_freeze_ids.geo"
