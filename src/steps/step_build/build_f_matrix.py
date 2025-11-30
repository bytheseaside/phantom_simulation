#!/usr/bin/env python3
"""
build_f_matrix.py

Build forward transfer matrix (F-matrix) for all cases in a VTU directory.
Samples all probes for each case and creates:
  - F_matrix.npy (raw matrix)
  - F_matrix.csv (normalized for readability)
  - F_matrix_heatmap.png (quick visual of measured values)
  - metadata.json (case info)

Usage:
  python build_f_matrix.py --vtu-dir run_1/solutions --probes probes_generated.csv --out f_matrix_output
"""
import argparse
import json
import sys
from pathlib import Path
import fnmatch
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyvista as pv


def extract_vtu_cases(vtu_dir: Path):
    """Find all VTU files in the specified directory and derive case names from their filenames."""
    vtu_files = sorted(vtu_dir.glob("solution_*.vtu"))
    cases = []
    for vtu in vtu_files:
        case_name = vtu.stem[len("solution_"):]  # Remove "solution_" prefix given by previous step (#3)
        cases.append((case_name, vtu))
    return cases


def read_probes_coords(path: Path):
    """Read probe coordinates from CSV (name,x,y,z)."""
    names, pts = [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            name = row[0]
            try:
                x, y, z = float(row[1]), float(row[2]), float(row[3])
                names.append(name)
                pts.append((x, y, z))
            except Exception:
                continue
    return names, np.array(pts, dtype=float)


def sample_probes_from_vtu(vtu_path: Path, probe_coords: np.ndarray, array_name='u'):
    """Sample voltage values at probe locations from VTU file."""
    grid = pv.read(str(vtu_path))
    
    # Extract head surface (region_id=2)
    try:
        head_vol = grid.threshold((1.5, 2.5), scalars='region_id')
        head_surface = head_vol.extract_surface()
    except Exception:
        head_surface = grid.extract_surface()
    
    if array_name not in head_surface.point_data:
        print(f"WARNING: Array '{array_name}' not found in {vtu_path.name}", file=sys.stderr)
        return np.full(len(probe_coords), np.nan)
    
    # Sample at each probe location
    values = np.zeros(len(probe_coords))
    for i, probe_coord in enumerate(probe_coords):
        closest_id = head_surface.find_closest_point(probe_coord)
        values[i] = head_surface.point_data[array_name][closest_id]
    
    return values


def build_f_matrix(cases, probe_coords, verbose=True):
    """
    Build F-matrix: rows = probes, columns = dipole cases.
    F[i, j] = voltage at probe i for dipole case j.
    """
    n_probes = len(probe_coords)
    n_cases = len(cases)
    
    F = np.zeros((n_probes, n_cases))
    case_names = []
    
    for j, (dipole_name, vtu_path) in enumerate(cases):
        if verbose:
            print(f"[{j+1}/{n_cases}] Sampling {dipole_name} from {vtu_path.name}")
        
        values = sample_probes_from_vtu(vtu_path, probe_coords, array_name='u')
        F[:, j] = values
        case_names.append(dipole_name)
    
    return F, case_names


def save_outputs(f_matrix, probe_names, case_names, output_dir: Path):
    """Save F-matrix in multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save raw NPY
    npy_path = output_dir / "F_matrix.npy"
    np.save(npy_path, f_matrix)
    print(f"Saved raw matrix: {npy_path}")
        
    # 2. Save metadata
    max_val = np.max(np.abs(f_matrix))
    metadata = {
        "n_probes": len(probe_names),
        "n_cases": len(case_names),
        "probe_names": probe_names,
        "cases": case_names,
        "matrix_shape": list(f_matrix.shape),
        "max_abs_value": float(max_val),
        "max_value": float(np.max(f_matrix)),
        "min_value": float(np.min(f_matrix)),
        "mean_abs_value": float(np.mean(np.abs(f_matrix)))
    }
    meta_path = output_dir / "metadata.json"
    with meta_path.open('w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    # 3. Create heatmap visualization with numbers (auto unit selection)
    _, ax = plt.subplots(figsize=(18, 12))

    # Auto-select display unit using 95th percentile
    p95 = np.percentile(np.abs(f_matrix), 95)
    if p95 >= 1.0:
        scale = 1.0
        unit = 'V'
        fmt = '{:.3f}'
    elif p95 >= 1e-3:
        scale = 1e3
        unit = 'mV'
        fmt = '{:.1f}'
    else:
        scale = 1e6
        unit = 'μV'
        fmt = '{:.0f}'

    f_disp = f_matrix * scale
    max_disp = np.max(np.abs(f_disp))

    im = ax.imshow(f_disp, aspect='auto', cmap='RdBu_r', vmin=-max_disp, vmax=max_disp)

    n_probes, n_cases = f_disp.shape    
    for i in range(n_probes):
        for j in range(n_cases):
            ax.text(j, i, fmt.format(f_disp[i, j]),
                    ha="center", va="center", color="black", fontsize=4)

    ax.set_xlabel('Case', fontsize=12)
    ax.set_ylabel('Probe', fontsize=12)
    ax.set_title(f'F-Matrix Heatmap ({unit})', fontsize=14)

    ax.set_xticks(range(len(case_names)))
    ax.set_xticklabels(case_names, rotation=45, fontsize=8)
    ax.xaxis.set_ticks_position('bottom')
    n_probes = len(probe_names)
    tick_stride = max(1, n_probes // 20)
    y_ticks = list(range(0, n_probes, tick_stride))
    y_labels = [probe_names[i] for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=6)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'Voltage ({unit})', fontsize=10)

    plt.tight_layout()
    heatmap_path = output_dir / "F_matrix_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vtu-dir', type=Path, required=True, 
                    help='Directory containing VTU files')
    ap.add_argument('--probes', type=Path, required=True,
                    help='Probe CSV file (name,x,y,z)')
    ap.add_argument('--out', type=Path, default=Path('f_matrix_output'),
                    help='Output directory')
    ap.add_argument('--pattern', type=str, default=None,
                    help='Glob pattern to include case names (e.g., "dipole_*")')
    args = ap.parse_args()

    if not args.vtu_dir.exists():
        print(f"ERROR: VTU directory not found: {args.vtu_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.probes.exists():
        print(f"ERROR: Probes file not found: {args.probes}", file=sys.stderr)
        sys.exit(1)

    # Find all VTU cases
    print(f"Scanning for VTU files in {args.vtu_dir}...")
    cases = extract_vtu_cases(args.vtu_dir)
    print(f"Found {len(cases)} cases")

    if len(cases) == 0:
        print("ERROR: No VTU files found", file=sys.stderr)
        sys.exit(1)

    # Optional filtering using a glob pattern (simple shell-style wildcard)
    if args.pattern:
        filtered = [c for c in cases if fnmatch.fnmatch(c[0], args.pattern)]
        print(f"Filtered with pattern '{args.pattern}': {len(filtered)} cases remain")
        cases = filtered

    # Read probes
    print(f"Reading probes from {args.probes}...")
    probe_names, probe_coords = read_probes_coords(args.probes)
    print(f"Loaded {len(probe_names)} probes")

    # Build F-matrix
    print("Building F-matrix...")
    f_result, case_names = build_f_matrix(cases, probe_coords, verbose=True)

    # Save outputs
    print(f"\nSaving outputs to {args.out}...")
    save_outputs(f_result, probe_names, case_names, args.out)

    print("\n=== Summary ===")
    print(f"F-matrix shape: {f_result.shape[0]} probes × {f_result.shape[1]} cases")
    print(f"Value range: [{np.min(f_result):.3e}, {np.max(f_result):.3e}] V")
    print(f"Mean |F|: {np.mean(np.abs(f_result)):.3e} V")
    print("Done!")


if __name__ == '__main__':
    try:
        pv.OFF_SCREEN = True
    except Exception:
        pass
    main()
