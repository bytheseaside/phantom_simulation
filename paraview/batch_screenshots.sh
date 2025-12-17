#!/usr/bin/env bash
set -euo pipefail

# defaults
W=2000
H=2000

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --state)
      STATE="$2"; shift 2;;
    --output)
      OUTPUT_DIR="$2"; shift 2;;
    --blob)
      BLOB="$2"; shift 2;;
    --w)
      W="$2"; shift 2;;
    --h)
      H="$2"; shift 2;;
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

# sanity checks
: "${STATE:?--state is required}"
: "${OUTPUT_DIR:?--output is required}"
: "${BLOB:?--blob is required}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/screenshot_from_state.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
  echo "Cannot find screenshot_from_state.py next to this bash script"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# expand blob
shopt -s nullglob
FILES=( $BLOB )
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No files matched: $INPUT_DIR/$BLOB"
  exit 1
fi

echo "State  : $STATE"
echo "Blob   : $BLOB"
echo "Output : $OUTPUT_DIR"
echo "Size   : ${W}x${H}"
echo "Files  : ${#FILES[@]}"
echo

# iterate
for f in "${FILES[@]}"; do
  name="$(basename "$f")"
  stem="${name%.*}"
  out="$OUTPUT_DIR/$stem.png"

  echo "→ $name"
  pvpython "$PY_SCRIPT" \
    --state "$STATE" \
    --xdmf "$f" \
    --out "$out" \
    --w "$W" \
    --h "$H"
done
