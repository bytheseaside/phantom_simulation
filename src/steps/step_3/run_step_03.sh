#!/usr/bin/env bash
set -euo pipefail


# ------------------------------------------------------------------------------
# --- Per-run JIT cache isolation (macOS-friendly) ---
# Clear any global stale caches (safe; ignore if missing)
rm -rf "$HOME/.cache/fenics" "$HOME/.cache/ffcx" "$HOME/.cache/dolfinx" 2>/dev/null || true

# Use per-run caches in current working directory
export DOLFINX_JIT_CACHE_DIR="$PWD/__dolfinx_cache__"
export FFCX_CACHE_DIR="$PWD/__ffcx_cache__"
export XDG_CACHE_HOME="$PWD/__cache__"
export FFCX_REGENERATE=1
mkdir -p "$DOLFINX_JIT_CACHE_DIR" "$FFCX_CACHE_DIR" "$XDG_CACHE_HOME"

# --- macOS linker fix: avoid duplicate LC_RPATH from Conda flags/CFFI ---
# Remove possibly intrusive Conda flags that cause duplicate rpaths
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
# Use clang toolchain and build bundle-style shared modules for cffi
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
export ARCHFLAGS="-arch arm64"
# Key line: force cffi/distutils to build Python extension bundles w/o extra rpaths
export LDSHARED="$CC -bundle -undefined dynamic_lookup $ARCHFLAGS"
# Optional: see compile/link lines if something fails
export FFCX_JIT_LOG_LEVEL=DEBUG

# Run solver.py with all arguments passed through
SOLVER_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/solver.py"

echo ">> Running: python $SOLVER_SCRIPT $*"
python "$SOLVER_SCRIPT" "$@"
