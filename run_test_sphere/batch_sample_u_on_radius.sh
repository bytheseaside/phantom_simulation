#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# CONFIG
# ----------------------------
PYTHON_SCRIPT="/Users/brisarojas/Desktop/phantom_simulation/run_test_sphere/sample_u_on_radius.py"
PROBES_CSV="/Users/brisarojas/Desktop/phantom_simulation/run_test_sphere/sphere_probes.csv"
OUTDIR="/Users/brisarojas/Desktop/phantom_simulation/run_test_sphere/cases_sampled"
# ----------------------------

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <directory-with-xdmf-files>"
  exit 1
fi

CASEDIR="$1"

mkdir -p "$OUTDIR"

echo "Batch sampling..."
echo "  Cases from:  $CASEDIR"
echo "  Using probes: $PROBES_CSV"
echo "  Output dir:    $OUTDIR"
echo

shopt -s nullglob
for casefile in "$CASEDIR"/*.xdmf; do
  basename="$(basename "$casefile" .xdmf)"  # Remove .xdmf extension
  outfile="$OUTDIR/${basename}.csv"        # Append .csv to the base name

  echo "[RUN] $basename"
  pvpython "$PYTHON_SCRIPT" \
    --case "$casefile" \
    --probes "$PROBES_CSV" \
    --out "$outfile"

  echo "  → wrote $outfile"
  echo
done

echo "Done."
