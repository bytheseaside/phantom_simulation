#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_screenshots.py — Automated batch screenshot generation for simulation cases

PURPOSE
-------
Automate the process of generating standardized screenshots for all simulation cases:
  • Load each VTU file
  • Set camera to specified view (default: front)
  • Save screenshot with case name
  • Process all cases without manual interaction

USAGE
-----
python batch_screenshots.py <cases_dir> [OPTIONS]

Example:
  python batch_screenshots.py run_lit_values/cases
  python batch_screenshots.py run_lit_values/cases --view back --clim 150000

Options:
  cases_dir         Directory containing solution_*.vtu files
  --view DIRECTION  Camera view: front, back, left, right, top, bottom (default: front)
  --output DIR      Output directory for screenshots (default: screenshots)
  --head MODE       Head display: surface, volume, off (default: surface)
  --gel MODE        Gel display: off, surface, volume (default: off)
  --no-elec         Hide electrodes
  --clim VALUE      Fixed symmetric color limit in μV (e.g., 150000 for ±150k μV)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pyvista as pv

# Allow empty meshes
pv.global_theme.allow_empty_mesh = True


def set_camera_view(plotter, direction='front'):
    """Set camera to specified view."""
    if direction == 'front':
        plotter.view_vector((0, -1, 0))  # Visual front (face)
    elif direction == 'back':
        plotter.view_vector((0, 1, 0))  # Visual back (occiput)
    elif direction == 'left':
        plotter.view_vector((-1, 0, 0))  # Left side
    elif direction == 'right':
        plotter.view_vector((1, 0, 0))  # Right side
    elif direction == 'top':
        plotter.view_vector((0, 0, 1))  # Top
    elif direction == 'bottom':
        plotter.view_vector((0, 0, -1))  # Bottom
    else:
        print(f"[warn] Unknown view direction '{direction}', using front")
        plotter.view_vector((0, -1, 0))


def process_case(vtu_path, view_direction, output_dir, head_mode='surface', gel_mode='off', show_elec=True, scale_factor=1e3, clim=None):
    """Process a single VTU file: load, configure view, save screenshot."""
    try:
        print(f"[processing] {vtu_path.name}")
        
        # Load the VTU file
        grid = pv.read(str(vtu_path))
        
        if 'u' not in grid.point_data:
            print(f"[error] No 'u' field in {vtu_path.name}, skipping")
            return False
        
        # Create a copy for scaled visualization
        grid_scaled = grid.copy()
        grid_scaled.point_data['u'] = grid.point_data['u'] * scale_factor
        
        # Separate volume cells (tetra) from surface cells (triangles)
        vol_cells = grid_scaled.extract_cells_by_type([pv.CellType.TETRA, pv.CellType.QUADRATIC_TETRA])
        tri_cells = grid_scaled.extract_cells_by_type([pv.CellType.TRIANGLE, pv.CellType.QUADRATIC_TRIANGLE])
        
        # Extract head and gel volumes
        if 'region_id' in vol_cells.cell_data:
            head_vol = vol_cells.threshold((1.5, 2.5), scalars='region_id')  # region_id=2
            gel_vol = vol_cells.threshold((0.5, 1.5), scalars='region_id')   # region_id=1
        else:
            head_vol = vol_cells
            gel_vol = None
        
        # Calculate color range (centered at zero for diverging colormap)
        if clim is not None:
            # Use fixed symmetric limit
            color_range = (-clim, clim)
        else:
            # Auto-scale per case
            u_data = grid_scaled.point_data['u']
            vmax = np.max(np.abs(u_data))
            color_range = (-vmax, vmax)
        
        # Create plotter (off-screen for batch processing)
        size = 800
        plotter = pv.Plotter(off_screen=True, window_size=(size, size))
        plotter.add_axes()
        
        # Add head mesh based on mode
        if head_mode != 'off' and head_vol and head_vol.n_cells > 0:
            if head_mode == 'surface':
                head_mesh = head_vol.extract_surface()
            else:  # volume
                head_mesh = head_vol
            
            plotter.add_mesh(
                head_mesh,
                scalars='u',
                cmap='PRGn',
                clim=color_range,
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Voltage (mV)',
                    'vertical': True,
                    'height': 0.6,
                    'position_x': 0.85,
                    'position_y': 0.2,
                    'title_font_size': 16,
                    'label_font_size': 14,
                }
            )
        
        # Add gel mesh based on mode
        if gel_mode != 'off' and gel_vol and gel_vol.n_cells > 0:
            if gel_mode == 'surface':
                gel_mesh = gel_vol.extract_surface()
            else:  # volume
                gel_mesh = gel_vol
            
            plotter.add_mesh(
                gel_mesh,
                scalars='u',
                cmap='PRGn',
                clim=color_range,
                opacity=0.5 if gel_mode == 'volume' else 1.0,
                show_scalar_bar=False
            )
        
        # Add electrode surfaces
        if show_elec and tri_cells and tri_cells.n_cells > 0:
            plotter.add_mesh(
                tri_cells,
                scalars='u',
                cmap='PRGn',
                clim=color_range,
                show_scalar_bar=False
            )
        
        # Set camera view
        set_camera_view(plotter, view_direction)
        
        # Reset camera to fit scene
        plotter.reset_camera()
        
        # Generate output filename (remove 'solution_' prefix)
        case_name = vtu_path.stem.replace('solution_', '')
        output_filename = f"{case_name}.png"
        output_path = output_dir / output_filename
        
        # Save screenshot (will overwrite if exists)
        plotter.screenshot(str(output_path))
        plotter.close()
        
        print(f"  ✓ Saved: {output_filename}")
        return True
        
    except Exception as e:
        print(f"[error] Failed to process {vtu_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate screenshots for simulation cases",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "cases_dir",
        type=Path,
        help="Directory containing solution_*.vtu files"
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=['front', 'back', 'left', 'right', 'top', 'bottom'],
        default='front',
        help="Camera view direction"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("screenshots"),
        help="Output directory for screenshots"
    )
    parser.add_argument(
        "--head",
        type=str,
        choices=['surface', 'volume', 'off'],
        default='surface',
        help="Head display mode"
    )
    parser.add_argument(
        "--gel",
        type=str,
        choices=['off', 'surface', 'volume'],
        default='off',
        help="Gel display mode"
    )
    parser.add_argument(
        "--no-elec",
        action='store_true',
        help="Hide electrodes"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="solution_*.vtu",
        help="File pattern to match in cases_dir"
    )
    parser.add_argument(
        "--clim",
        type=float,
        default=None,
        help="Fixed symmetric color limit in mV (e.g., 150 for ±150 mV). If not set, auto-scales per case."
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.cases_dir.exists():
        print(f"[error] Cases directory not found: {args.cases_dir}")
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Find all VTU files
    vtu_files = sorted(args.cases_dir.glob(args.pattern))
    
    if not vtu_files:
        print(f"[error] No files matching '{args.pattern}' found in {args.cases_dir}")
        sys.exit(1)
    
    print(f"[info] Found {len(vtu_files)} VTU files to process")
    print(f"[info] Output directory: {args.output}")
    print(f"[info] View: {args.view}, Head: {args.head}, Gel: {args.gel}, Electrodes: {not args.no_elec}")
    if args.clim:
        print(f"[info] Fixed color range: ±{args.clim:.1f} mV")
    else:
        print(f"[info] Color range: auto per case")
    print("-" * 60)
    
    # Process each file
    success_count = 0
    for vtu_file in vtu_files:
        if process_case(vtu_file, args.view, args.output, args.head, args.gel, not args.no_elec, 1e3, args.clim):
            success_count += 1
    
    print("-" * 60)
    print(f"[done] Successfully processed {success_count}/{len(vtu_files)} cases")
    print(f"[done] Screenshots saved to: {args.output}")


if __name__ == "__main__":
    main()
