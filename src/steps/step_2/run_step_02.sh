#!/usr/bin/env bash
set -euo pipefail
# Usage: ./run_step_02.sh --geo <path_to_geo_file>
GEO_FILE=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --geo)
      GEO_FILE="$2"
      shift 2
      ;;
    *)
      echo "✗ Unknown argument: $1"
      echo "Usage: $0 --geo <path>"
      exit 1
      ;;
  esac
done

# Validate required arguments
if  [ -z "$GEO_FILE" ]; then
  echo "✗ Error: Missing required arguments"
  echo "Usage: $0 --geo <path>"
  exit 1
fi


# Validate GEO file exists
if [ ! -f "$GEO_FILE" ]; then
  echo "✗ Error: GEO file does not exist: $GEO_FILE"
  exit 1
fi

# Convert paths to absolute
GEO_FILE="$(cd "$(dirname "$GEO_FILE")" && pwd)/$(basename "$GEO_FILE")"

echo ">> Running Gmsh..."
echo "   GEO: $GEO_FILE"

if ! gmsh "$GEO_FILE" -v 4 -; then
  echo "✗ Error running Gmsh"
  exit 1
fi
echo ">> Gmsh completed successfully."