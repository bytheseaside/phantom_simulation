#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Step 3 runner for solver.py (FEniCSx)
#
# Usage:
#   bash run_step_03.sh /path/to/run_dir [-n N]
#
# Input files required in run_dir:
#   - mesh.msh
#   - simulation_cases.csv
#
# Output:
#   - run_dir/run_steps/step3.log
#   - run_dir/cases/solution_<case>.xdmf
#   - run_dir/cases/solution_<case>.vtu
#   - run_dir/cases/solution_<case>.h5
# ------------------------------------------------------------------------------

RUN_DIR="${1:-}"
if [ -z "$RUN_DIR" ]; then
  echo "Usage: $0 /path/to/run_dir [-n N]"
  exit 1
fi
shift || true

# Optional: MPI process count
NP=0
if [[ "${1:-}" == "-n" ]]; then
  NP="${2:-0}"
  shift 2 || true
fi

# Resolve script dir robustly (works with bash or sh)
if [ -n "${BASH_SOURCE:-}" ]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
else
  SCRIPT_PATH="$0"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

# Sanity checks on inputs
[ -d "$RUN_DIR" ] || { echo "✗ Run directory does not exist: $RUN_DIR"; exit 1; }
[ -f "$RUN_DIR/mesh.msh" ] || { echo "✗ Missing: $RUN_DIR/mesh.msh"; exit 1; }

[ -f "$RUN_DIR/simulation_cases.csv" ] || { echo "✗ Missing: $RUN_DIR/simulation_cases.csv"; exit 1; }

# Ensure results directory exists
mkdir -p "$RUN_DIR/cases"

# Copy solver.py into the run dir (so results are self-contained)
SRC_SOLVER="${SCRIPT_DIR}/solver.py"
[ -f "$SRC_SOLVER" ] || { echo "✗ solver.py not found next to runner: $SRC_SOLVER"; exit 1; }
cp "$SRC_SOLVER" "$RUN_DIR/run_steps/"

cd "$RUN_DIR/run_steps" || exit
mkdir -p solutions

LOG="step3.log"
echo ">> Step 3 in $RUN_DIR  (log: $LOG)"

# --- Per-run JIT cache isolation (macOS-friendly) ---
# Clear any global stale caches (safe; ignore if missing)
rm -rf "$HOME/.cache/fenics" "$HOME/.cache/ffcx" "$HOME/.cache/dolfinx" 2>/dev/null || true
# Use per-run caches inside the run folder
export DOLFINX_JIT_CACHE_DIR="$PWD/__dolfinx_cache__"
export FFCX_CACHE_DIR="$PWD/__ffcx_cache__"
export XDG_CACHE_HOME="$PWD/__cache__"
export FFCX_REGENERATE=1
mkdir -p "$DOLFINX_JIT_CACHE_DIR" "$FFCX_CACHE_DIR" "$XDG_CACHE_HOME"

# --- macOS linker fix: avoid duplicate LC_RPATH from Conda flags/CFFI ---
# 1) Remove possibly intrusive Conda flags that cause duplicate rpaths
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
# 2) Use clang toolchain and build bundle-style shared modules for cffi
export CC="${CC:-clang}"
export CXX="${CXX:-clang++}"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-13.0}"
export ARCHFLAGS="-arch arm64"
# Key line: force cffi/distutils to build Python extension bundles w/o extra rpaths
export LDSHARED="$CC -bundle -undefined dynamic_lookup $ARCHFLAGS"
# Optional: see compile/link lines if something fails
export FFCX_JIT_LOG_LEVEL=DEBUG

# Pre-flight: check we can import dolfinx and gmsh from this env
python - <<'PY' 2>/dev/null || { echo '✗ Pre-check failed: activate env (micromamba activate env) and ensure the gmsh Python module is installed.'; exit 1; }
import sys, dolfinx, ufl
print("python:", sys.executable)
print("dolfinx:", dolfinx.__version__, "ufl:", ufl.__version__)
import gmsh
print("gmsh:", gmsh.__version__)
PY

LOG="step3.log"
echo ">> Step 3 in $RUN_DIR/run_steps  (log: $LOG)"

# If a previous bad JIT was produced in this run dir, clean local caches once
rm -rf "__cache__" "__ffcx_cache__" "__dolfinx_cache__" 2>/dev/null || true
mkdir -p "__cache__" "__ffcx_cache__" "__dolfinx_cache__"

# Stopwatch helper
ts() { date +%s; }

set +e
T0=$(ts)
# Run solver with updated paths
if [ "$NP" -gt 0 ] && command -v mpirun >/dev/null 2>&1; then
  echo ">> Running with MPI: mpirun -n $NP python solver.py" | tee "$LOG"
  mpirun -n "$NP" python solver.py >> "$LOG" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "!! MPI run failed (rc=$RC). Falling back to single process..." | tee -a "$LOG"
    python solver.py >> "$LOG" 2>&1
    RC=$?
  fi
else
  echo ">> Running single-process: python solver.py" | tee "$LOG"
  python solver.py >> "$LOG" 2>&1
  RC=$?
fi
T1=$(ts)
set -e

if [ $RC -ne 0 ]; then
  echo "✗ Error. See $RUN_DIR/run_steps/$LOG"
  exit $RC
fi

echo "✓ Done in $((T1 - T0)) s. See:"
echo "${RUN_DIR}/"
echo "├── cases/"
echo "│   ├── solution_<case>.xdmf *"
echo "│   ├── solution_<case>.vtu *"
echo "│   ├── solution_<case>.h5 *"
echo "│   ├── ..."
echo "├── run_steps/"
echo "│   ├── step3.log *"
echo "│   ├── ..."
echo "├── ..."