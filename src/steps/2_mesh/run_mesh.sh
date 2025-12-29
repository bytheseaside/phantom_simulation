#!/usr/bin/env bash
set -euo pipefail

# Parse named arguments
GEO_FILE=""
XAO_FILE=""
OUT_FILE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --geo)
      GEO_FILE="$2"
      shift 2
      ;;
    --xao)
      XAO_FILE="$2"
      shift 2
      ;;
    --out)
      OUT_FILE="$2"
      shift 2
      ;;
    *)
      echo "✗ Unknown argument: $1"
      echo "Usage: $0 --geo <path> --xao <path> --out <path>"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$GEO_FILE" ] || [ -z "$XAO_FILE" ] || [ -z "$OUT_FILE" ]; then
  echo "✗ Error: Missing required arguments"
  echo "Usage: $0 --geo <path> --xao <path> --out <path>"
  exit 1
fi

# Validate GEO file exists
if [ ! -f "$GEO_FILE" ]; then
  echo "✗ Error: GEO file does not exist: $GEO_FILE"
  exit 1
fi

# Validate XAO file exists
if [ ! -f "$XAO_FILE" ]; then
  echo "✗ Error: XAO file does not exist: $XAO_FILE"
  exit 1
fi

# Convert to absolute paths
GEO_FILE="$(cd "$(dirname "$GEO_FILE")" && pwd)/$(basename "$GEO_FILE")"
XAO_FILE="$(cd "$(dirname "$XAO_FILE")" && pwd)/$(basename "$XAO_FILE")"

# Ensure output directory exists
OUT_DIR="$(dirname "$OUT_FILE")"
if [ ! -d "$OUT_DIR" ]; then
  mkdir -p "$OUT_DIR"
  echo ">> Created output directory: $OUT_DIR"
fi

# Convert OUT_FILE to absolute path
OUT_FILE="$(cd "$OUT_DIR" && pwd)/$(basename "$OUT_FILE")"

echo ">> Running Gmsh..."
echo "   GEO: $GEO_FILE"
echo "   XAO: $XAO_FILE"
echo "   OUT: $OUT_FILE"

if ! XAO_PATH="$XAO_FILE" OUT_PATH="$OUT_FILE" gmsh "$GEO_FILE" -v 4 -; then
  echo "✗ Error running Gmsh"
  exit 1
fi

echo "✓ OK. Mesh created: $OUT_FILE"
