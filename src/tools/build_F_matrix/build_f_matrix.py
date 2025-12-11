#!/usr/bin/env python3
"""
build_f_matrix.py

Build forward transfer matrix (F-matrix) for stimulation cases in a VTU directory.
Samples voltage values at probe locations for each case and creates:
  - F_matrix.npy (raw matrix in Volts)
  - F_matrix.csv (matrix with probe and case names for inspection)
  - F_matrix_heatmap.svg (visual heatmap with auto-scaled units and optional annotations)
  - metadata.json (matrix dimensions, case names, probe names, and value statistics)

Usage:
  python build_f_matrix.py --vtu-dir run/solutions --probes run/probes.csv --out run/mono_F
  python build_f_matrix.py --vtu-dir run/solutions --probes run/probes.csv --out run/dip_F --pattern "D_*" --clim 200

Options:
  --vtu-dir <path>      Directory containing VTU solution files (solution_*.vtu).
  --probes <path>       CSV file with probe coordinates (name,x,y,z).
  --out <directory>     Output directory for F-matrix and visualizations.
  --pattern <glob>      Glob pattern to filter case names (e.g., "D_*" for dipolar only).
  --annotate            Annotate heatmap cells with numeric voltage values.
  --clim <value>        Symmetric colorbar limit in mV (e.g., 200 for ±200 mV).
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
    """Sample voltage values at probe locations from VTU file.
    
    Extracts head surface (region_id=2) and samples scalar field 'u' (voltage)
    at the closest surface point to each probe coordinate.
    
    Parameters:
        vtu_path: path to VTU solution file
        probe_coords: (n_probes, 3) array of probe coordinates
        array_name: name of scalar field to sample (default: 'u' for voltage)
    
    Returns:
        (n_probes,) array of sampled voltage values in Volts
    """
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


def save_outputs(f_matrix, probe_names, case_names, output_dir: Path, annotate=False, clim_mv=None):
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
    n_probes, n_cases = f_matrix.shape
    
    # Scale figure size generously to keep fixed font sizes readable
    fig_width = max(24, n_cases * 1.2)
    fig_height = max(18, n_probes * 1.2)
    _, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    # Auto-select display unit using 95th percentile for optimal readability
    # Converts from Volts to mV or μV based on data magnitude
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
    # Use symmetric limits centered at zero for diverging PRGn colormap
    # (purple=negative, green=positive)
    if clim_mv is not None:
        # Convert clim from mV to display units (clim is always in mV, scale converts V to display units)
        max_abs = clim_mv * (scale / 1e3)  # clim in mV, scale is 1.0 (V), 1e3 (mV), or 1e6 (μV)
    else:
        # Auto-detect from data
        max_abs = np.max(np.abs(f_disp))
    vmin, vmax = -max_abs, max_abs

    im = ax.imshow(f_disp, aspect='auto', cmap='PRGn', vmin=vmin, vmax=vmax)

    if annotate:
        import matplotlib.patheffects as path_effects
        cell_fontsize = 18
        for i in range(n_probes):
            for j in range(n_cases):
                text = ax.text(j, i, fmt.format(f_disp[i, j]),
                        ha="center", va="center", color="black", 
                        fontsize=cell_fontsize, fontweight='bold')
                text.set_path_effects([
                    path_effects.Stroke(linewidth=2, foreground='white'),
                    path_effects.Normal()
                ])

    tick_fontsize = 22
    ax.set_xlabel('Stimulation Case', fontsize=28, fontweight='bold', labelpad=20)
    ax.set_xticks(range(len(case_names)))
    ax.set_xticklabels(case_names, rotation=45, ha='right', fontsize=tick_fontsize)
    ax.xaxis.set_ticks_position('bottom')
    
    ax.set_ylabel('Probe', fontsize=28, fontweight='bold')
    ax.set_yticks(range(n_probes))
    ax.set_yticklabels(probe_names, fontsize=tick_fontsize)

    ax.set_title('Probe Voltage Response Map', fontsize=40, fontweight='bold', pad=30)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(f'Voltage ({unit})', fontsize=28, fontweight='bold')
    
    # Simple symmetric ticks from -max_abs to +max_abs
    ticks = np.linspace(-max_abs, max_abs, 9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([fmt.format(t) for t in ticks])
    cbar.ax.tick_params(labelsize=30)

    plt.tight_layout()
    heatmap_path = output_dir / "F_matrix_heatmap.svg"
    plt.savefig(heatmap_path, dpi=300)
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
    ap.add_argument('--annotate', action='store_true', 
                    help='Annotate heatmap cells with numeric values')
    ap.add_argument('--clim', type=float, default=None,
                    help='Symmetric colorbar limit in mV (e.g., 200 for ±200 mV)')
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
    save_outputs(f_result, probe_names, case_names, args.out, annotate=args.annotate, clim_mv=args.clim)

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
