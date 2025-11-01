#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_vtu.py — Probe scalar field 'u' from a VTU at arbitrary (x,y,z) points using PyVista.

Usage:
  python probe_vtu.py --vtu solutions/solution_e6.vtu --probes probes.csv --out solutions/probes_e6.csv
"""
import argparse
import csv
import math
from pathlib import Path
import sys
import numpy as np
import pyvista as pv

def read_probes(path: Path):
    names, pts = [], []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            parts = [p.strip() for p in (s.split(",") if "," in s else s.split())]
            if len(parts) < 4:
                continue
            name = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                continue
            names.append(name)
            pts.append((x, y, z))
    return names, np.array(pts, dtype=float)

def write_results(path: Path, names, pts, values):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "x", "y", "z", "u_volts"])
        for i, name in enumerate(names):
            x, y, z = pts[i]
            val = values[i]
            w.writerow([name, x, y, z, float(val) if val is not None else float("nan")])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtu", required=True, type=Path, help="Path to solution_<case>.vtu")
    ap.add_argument("--probes", required=True, type=Path, help="Path to probes.csv (name,x,y,z)")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV path")
    ap.add_argument("--var", type=str, default="u",
                    help="Name of the scalar array to sample (default: 'u')")
    args = ap.parse_args()

    if not args.vtu.exists():
        print(f"ERROR: VTU not found: {args.vtu}", file=sys.stderr); sys.exit(1)
    if not args.probes.exists():
        print(f"ERROR: probes file not found: {args.probes}", file=sys.stderr); sys.exit(1)

    names, pts = read_probes(args.probes)
    if len(names) == 0:
        write_results(args.out, [], np.empty((0, 3)), [])
        print(f"[write] {args.out} (0 probes)")
        return

    # Load VTU (UnstructuredGrid)
    grid = pv.read(str(args.vtu))

    head_vol = grid.threshold((1.5, 2.5), scalars='region_id')
    head_surface = head_vol.extract_surface()
    
    # Extract requested array using find_closest_point on surface
    array_name = args.var
    if array_name in head_surface.point_data:
        u_vals = np.zeros(len(pts))
        for i, probe_coord in enumerate(pts):
            closest_id = head_surface.find_closest_point(probe_coord)
            u_vals[i] = head_surface.point_data[array_name][closest_id]
    else:
        # Helpful diagnostic
        avail = list(head_surface.point_data.keys())
        print(f"WARNING: Array '{array_name}' not found in head surface point_data. Available: {avail}", file=sys.stderr)
        print(f"Source grid point_data: {list(grid.point_data.keys())}", file=sys.stderr)
        print(f"Source grid cell_data : {list(grid.cell_data.keys())}", file=sys.stderr)
        u_vals = np.full((pts.shape[0],), np.nan, dtype=float)

    values = [float(u_vals[i]) if (i < len(u_vals) and np.isfinite(u_vals[i])) else float("nan")
              for i in range(pts.shape[0])]

    write_results(args.out, names, pts, values)
    n_good = sum(1 for v in values if math.isfinite(v))
    print(f"[probe] {n_good}/{len(values)} points sampled from '{array_name}'")
    print(f"[write] {args.out}")

if __name__ == "__main__":
    try:
        pv.OFF_SCREEN = True  # avoid opening a render window
    except Exception:
        pass
    main()
