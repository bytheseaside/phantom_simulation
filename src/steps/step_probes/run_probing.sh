#!/usr/bin/env bash
set -euo pipefail

# Initialize variables
RUN_DIR=""
SINGLE_FILE=""
VAR="u"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --single-file)
      SINGLE_FILE="$2"
      shift 2
      ;;
    --probes)
      PROBES_PATH="$2"
      shift 2
      ;;
    --var)
      VAR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "${PROBES_PATH:-}" ]; then
  echo "✗ Missing required argument: --probes PATH. Please provide a valid probes.csv file."
  exit 1
fi

if [ -n "$RUN_DIR" ] && [ -n "$SINGLE_FILE" ]; then
  echo "✗ Cannot specify both --run-dir and --single-file. Choose one."
  exit 1
fi

if [ -z "$RUN_DIR" ] && [ -z "$SINGLE_FILE" ]; then
  echo "✗ Must specify either --run-dir or --single-file."
  exit 1
fi

if [ -n "$RUN_DIR" ]; then
  # Handle probing for a whole run
  [ -d "$RUN_DIR" ] || { echo "✗ Run directory does not exist: $RUN_DIR"; exit 1; }

  # Copy probe_vtu.py to run_steps
  SRC_PROBER="$(dirname "$0")/probe_vtu.py"
  DST_PROBER="$RUN_DIR/run_steps/probe_vtu.py"
  if [ -f "$SRC_PROBER" ]; then
    cp -f "$SRC_PROBER" "$DST_PROBER"
    echo ">> Copied probe_vtu.py -> $DST_PROBER"
  else
    echo "✗ probe_vtu.py not found next to this script: $SRC_PROBER"
    exit 1
  fi

  # Gather VTU files from RUN_DIR/cases
  shopt -s nullglob
  VTUS=( "$RUN_DIR/cases"/*.vtu )
  shopt -u nullglob

  if [ ${#VTUS[@]} -eq 0 ]; then
    echo "✗ No VTU files found in $RUN_DIR/cases."
    exit 1
  fi

  # Probe each VTU file
  TOTAL_FILES=${#VTUS[@]}
  COUNT=0
  for VTU in "${VTUS[@]}"; do
    COUNT=$((COUNT + 1))
    VBASE="$(basename "$VTU")"
    OUT="$RUN_DIR/cases/probes_${VBASE%.vtu}.csv"

    echo "[${COUNT}/${TOTAL_FILES}] Probing $VBASE -> $(basename "$OUT")"
    python "$DST_PROBER" --vtu "$VTU" --probes "$PROBES_PATH" --out "$OUT" --var "$VAR"
  done

elif [ -n "$SINGLE_FILE" ]; then
  # Handle probing for a single file
  [ -f "$SINGLE_FILE" ] || { echo "✗ File does not exist: $SINGLE_FILE"; exit 1; }

  VBASE="$(basename "$SINGLE_FILE")"
  OUT="probes_${VBASE%.vtu}.csv"

  echo "[1/1] Probing $VBASE -> $OUT"
  python "$(dirname "$0")/probe_vtu.py" --vtu "$SINGLE_FILE" --probes "$PROBES_PATH" --out "$OUT" --var "$VAR"
fi

echo "✓ Probing completed."
