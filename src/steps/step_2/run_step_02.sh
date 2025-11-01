#!/usr/bin/env bash
set -euo pipefail

PREP_DIR="${1:-}"
if [ -z "$PREP_DIR" ]; then
  echo "Usage: $0 /path/to/run_YYYYMMDD_HHMMSS"
  exit 1
fi
if [ ! -d "$PREP_DIR" ]; then
  echo "✗ Prep directory does not exist: $PREP_DIR"
  exit 1
fi
if [ ! -f "$PREP_DIR/prep.geo_unrolled.xao" ]; then
  echo "✗ Missing $PREP_DIR/prep.geo_unrolled.xao (Step 1 output)"
  exit 1
fi
if [ ! -f "$PREP_DIR/ids.inc.geo" ]; then
  echo "✗ Missing $PREP_DIR/ids.inc.geo (IDs file)"
  exit 1
fi

# Copy Step-2 GEO template into run dir
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "${SCRIPT_DIR}/02_label_and_mesh.geo" "$PREP_DIR/"

cd "$PREP_DIR"

LOG="step2.log"
echo ">> Running Gmsh Step 2 in: $PREP_DIR (log: $LOG)"
if ! gmsh -v 4 -3 02_label_and_mesh.geo -format msh4 -o ../mesh.msh &> "$LOG"; then
  echo "✗ Error. See $PREP_DIR/$LOG"
  exit 1
fi

# Optional cleanup: some builds auto-write *_geo_unrolled
[ -f "02_label_and_mesh.geo_unrolled" ] && rm -f "02_label_and_mesh.geo_unrolled"

PARENT_DIR="$(basename "$(cd "$PREP_DIR/.." && pwd)")"
echo "✓ OK. Results organized as follows:"
echo "$PARENT_DIR/"
echo "   ├── mesh.msh"
echo "   ├── ... "
echo "   └── $PREP_DIR/"
echo "       ├── step2.log"
echo "       └── ... "