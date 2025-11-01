#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
view_case.py — Interactive 3D visualization for FEniCSx electrostatic simulation results

PURPOSE
-------
Visualize voltage potential distributions from electrostatic simulations on head-phantom 
geometry with electrode arrays.

This viewer supports multi-actor rendering (head, gel, electrodes), interactive plane 
clipping, and probe validation for simulation quality assessment.

WHY THIS EXISTS
---------------
Electrostatic simulations require specialized visualization to:
  • Compare voltage distributions across tissue types (head vs gel)
  • Inspect internal potential fields via plane clipping
  • Validate electrode contact quality and current paths
  • Compare simulation results against probe measurements

KEY DESIGN DECISIONS
--------------------
Global vs Independent Ranges:
  • Global mode: Single colorbar for cross-tissue voltage comparison
  • Independent mode: Per-actor colorbars for enhanced local contrast
  WHY: Different analysis tasks need different perspectives

Static vs Dynamic Ranges:
  • Static: Locks ranges to full dataset (quantitative comparison during clipping)
  • Dynamic: Rescales to visible clipped data (maximizes local contrast)
  WHY: Preserves meaning during exploration vs optimizes detail viewing

Center-Zero with Diverging Colormap:
  • Symmetric ranges [-max, +max] with PRGn (purple=negative, green=positive)
  WHY: Bipolar current injection creates symmetric ±voltage patterns; diverging 
       colormap makes polarity and current direction visually obvious

Adaptive Probe Colors:
  • Colors change per colormap to ensure visibility
  WHY: White probes disappear on light backgrounds; complementary colors work across 
       entire colormap range

Range Calculation Strategy:
  • Always calculate from original unscaled data (Volts), then apply scale factor
  WHY: Prevents floating-point precision loss with small values (μV range)

DATA STRUCTURE
--------------
Input VTU from FEniCSx solver contains:
  • Volume cells (tetra10): point_data['u'] (voltage in V)
  • Cell tags: cell_data['region_id'] → 1=gel, 2=head
  • Surface cells (triangles): cell_data['facet_id'] → 1-9 (electrodes)

USAGE
-----
python view_case.py run_1/solutions/solution_balanced.vtu [OPTIONS]

Options:
  --head {surface,volume,off}       Head display mode (default: surface)
  --gel {off,surface,volume}        Gel display mode (default: off)
  --no-elec                         Hide electrode surfaces
  --pins <csv>                      Load probe positions (name,x,y,z in meters)
  --global-range                    Single colorbar across actors (default: True)
  --static-bar                      Lock ranges to full data (default: True)
  --center-zero                     Symmetric ±ranges with diverging cmap (default: True)
  --cmap-head <name>                Colormap (default: PRGn if center-zero, else viridis)

Keyboard Shortcuts:
  S           Save screenshot to ./screenshots/<case>/
  G           Toggle edge display
  H/J         Cycle head/gel display modes
  Q/Esc       Quit

UI Controls:
  • Actor toggles: Show/hide head, gel, electrodes, probe markers
  • Colorbar options: Global/independent, static/dynamic, center-zero
  • Plane clipping: Normal vector, position slider, anatomical plane presets
  
See VIEW_CASE_DOCUMENTATION.md for detailed technical documentation.
"""

# --- Pin CSV and Table Helpers ---
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import re
import csv

# PyVistaQt and PyQt5 for integrated 3D+table window
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QDockWidget
from PyQt5.QtCore import Qt

# Allow empty meshes (actors can clip to zero cells without error)
pv.global_theme.allow_empty_mesh = True
# ------------------------------------------------------------------------------

def parse_args():
    """Parse command-line options for the viewer."""
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="Path to solution_<case>.vtu produced by solver.py")
    p.add_argument("--head", choices=["surface", "volume", "off"], default="surface")
    p.add_argument("--gel",  choices=["off", "surface", "volume"], default="off")
    p.add_argument("--no-elec", action="store_true", help="Hide electrodes")

    p.add_argument("--cmap-head", default=None,
                   help="Colormap for head actor (default: viridis without centering, PRGn with centering)")
    p.add_argument("--cmap-gel",  default=None,
                   help="Colormap for gel/electrodes (default: same as head unless specified)")

    p.add_argument("--title", default=None,
                   help="Custom window title (defaults to derived case name)")
    p.add_argument("--pins", type=str, default=None,
                    help="CSV file with columns name,x,y,z for pin markers")
    
    p.add_argument("--static-bar", dest="static_bar", action="store_true", default=True,
                   help="Keep color bar ranges fixed to full dataset (default: True)")
    p.add_argument("--no-static-bar", dest="static_bar", action="store_false",
                   help="Allow color bars to rescale based on visible clipped data")
    
    p.add_argument("--global-range", dest="global_range", action="store_true", default=True,
                   help="Use single color bar with global min/max across all actors (default: True)")
    
    p.add_argument("--center-zero", dest="center_zero", action="store_true", default=True,
                   help="Center colormap at zero with symmetric range (e.g., [-2, +2]). Uses diverging colormap (default: True).")
    
    return p.parse_args()

# ------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------
def read_pins_csv(pins_path):
    pins_data = []
    try:
        with open(pins_path, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith("#"): continue
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) < 4: continue
                try:
                    x, y, z = map(float, parts[1:4])
                    name = parts[0]
                    pins_data.append((name, x, y, z))
                except ValueError:
                    continue
    except Exception as e:
        print(f"[warn] Failed to read pins CSV: {e}")
    return pins_data

def get_probe_colors_for_cmap(cmap_name):
    """
    Return (unselected_color, selected_color) pair optimized for the given colormap.
    
    Colors are chosen to have good visibility and contrast against the colormap.
    
    Parameters
    ----------
    cmap_name : str
        Colormap name (e.g., 'viridis', 'PRGn', 'RdBu_r')
    
    Returns
    -------
    tuple of str
        (unselected_color_hex, selected_color_hex)
    """
    # Common diverging colormaps (purple/green, red/blue, etc.)
    diverging = ['PRGn', 'PiYG', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdBu_r', 
                 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    
    # Common sequential colormaps
    sequential_cool = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                       'Blues', 'Purples', 'Greens', 'BuGn', 'GnBu', 'PuBu', 'YlGnBu']
    sequential_warm = ['hot', 'Reds', 'Oranges', 'YlOrRd', 'YlOrBr', 'OrRd']
    
    # For diverging colormaps: use cyan and orange (neutral, not in the diverging extremes)
    if any(d in cmap_name for d in diverging):
        return "#00BCD4", "#FF6F00"  # Cyan (Material Cyan 500) and Amber (Material Amber 800)
    
    # For cool sequential colormaps: use warm colors (orange and red)
    elif any(c in cmap_name for c in sequential_cool):
        return "#FF9800", "#D32F2F"  # Orange (Material Orange 500) and Red (Material Red 700)
    
    # For warm sequential colormaps: use cool colors (cyan and blue)
    elif any(w in cmap_name for w in sequential_warm):
        return "#00BCD4", "#1976D2"  # Cyan (Material Cyan 500) and Blue (Material Blue 700)
    
    # Default: cyan and red/bordeaux (works reasonably with most colormaps)
    else:
        return "#00BCD4", "#C62828"  # Teal cyan and Deep red

def add_pin_spheres_and_table(plotter, original_grid, pins_data, cmap_name="viridis"):
    """
    Add probe spheres to the 3D scene and create a probe selection table in the right dock.
    
    WHY THIS EXISTS
    ---------------
    Probes are physical validation points in the simulation. This function:
    - Visualizes probes as 3D spheres with colormap-adaptive colors
    - Interpolates solution values directly from original grid (preserving units)
    - Provides interactive table for probe selection/comparison
    - Highlights selected probe with contrasting color for easy identification
    
    KEY DESIGN DECISIONS
    --------------------
    - Interpolation from original_grid (not scaled): Preserves original μV values without rescaling artifacts
    - Adaptive colors per colormap: Ensures probe visibility against any colormap background
    - Table shows V (not μV): Prevents confusion with scaled visualization
    - Radio button selection: Enforces single-probe focus, highlights selection in 3D
    
    Parameters
    ----------
    plotter : pyvista.Plotter
        PyVista plotter instance for adding meshes
    original_grid : pv.UnstructuredGrid
        UNSCALED original mesh with solution data in 'u' (used for interpolation)
    pins_data : list of tuples
        [(label, x, y, z), ...] - probe coordinates in meters
    cmap_name : str
        Current colormap name (for adaptive color selection)
    
    Returns
    -------
    tuple[list, list]
        (sphere_actor_names, spheres) - Names and meshes for show/hide functionality
    """
    if not pins_data:
        return [], []  # Return empty lists if no probes
    
    # Get appropriate colors for this colormap
    unselected_color, selected_color = get_probe_colors_for_cmap(cmap_name)
    
    pts = np.array([(x, y, z) for _, x, y, z in pins_data])
    
    # Sample from HEAD SURFACE using find_closest_point (not volume!)
    # For EEG, probes measure voltage on the scalp surface
    head_vol = original_grid.threshold((1.5, 2.5), scalars='region_id') # region_id=2=head
    head_surface = head_vol.extract_surface()
    
    # Use find_closest_point on surface (NOT sample interpolation)
    u_vals = np.zeros(len(pts))
    if 'u' in head_surface.point_data:
        for i, probe_coord in enumerate(pts):
            closest_id = head_surface.find_closest_point(probe_coord)
            u_vals[i] = head_surface.point_data['u'][closest_id]
    else:
        # fallback: all NaN if something goes wrong
        u_vals = np.array([float('nan')] * len(pts))
    
    pin_names = [name for name, *_ in pins_data]
    # Keep values in V (no scaling), show in scientific notation
    u_vals_disp = u_vals
    # Spheres and their base names
    spheres = []
    sphere_actor_names = []
    for _, x, y, z in pins_data:
        s = pv.Sphere(radius=0.003, center=(x, y, z))
        spheres.append(s)
        name = f"pin_{x:.3f}_{y:.3f}_{z:.3f}"
        plotter.add_mesh(s, color=unselected_color, name=name, ambient=1.0, diffuse=0.0, specular=0.0, opacity=1.0)
        sphere_actor_names.append(name)
    from PyQt5.QtWidgets import QRadioButton, QButtonGroup, QSizePolicy
    try:
        table_widget = QTableWidget()
        table_widget.setColumnCount(3)
        # Always keep header in V, values will be in scientific notation
        table_widget.setHorizontalHeaderLabels(["", "Label", "u (V)"])
        table_widget.setRowCount(len(pin_names))
        table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Column sizing: select (radio) and label minimal, value stretches
        header = table_widget.horizontalHeader()
        header.setSectionResizeMode(0, header.ResizeToContents)
        header.setSectionResizeMode(1, header.ResizeToContents)
        header.setSectionResizeMode(2, header.Stretch)
        # Don't stretch last section - keep all rows the same height
        table_widget.verticalHeader().setStretchLastSection(False)
        radio_group = QButtonGroup(table_widget)
        radio_group.setExclusive(True)
        def highlight_probe(idx):
            for i, (s, base_name) in enumerate(zip(spheres, sphere_actor_names)):
                plotter.remove_actor(base_name, reset_camera=False, render=False)
                plotter.remove_actor(base_name + '_highlighted', reset_camera=False, render=False)
                if i == idx:
                    plotter.add_mesh(s, color=selected_color, name=base_name + '_highlighted', ambient=1.0, diffuse=0.0, specular=0.0, opacity=1.0)
                else:
                    plotter.add_mesh(s, color=unselected_color, name=base_name, ambient=1.0, diffuse=0.0, specular=0.0, opacity=1.0)
            plotter.render()
        for i, (label, uval) in enumerate(zip(pin_names, u_vals_disp)):
            radio = QRadioButton()
            radio_group.addButton(radio, i)
            table_widget.setCellWidget(i, 0, radio)
            item_label = QTableWidgetItem(str(label))
            item_label.setFlags(item_label.flags() & ~Qt.ItemIsEditable)
            table_widget.setItem(i, 1, item_label)
            # Always show sign and format in scientific notation (V units)
            item_uval = QTableWidgetItem(f"{uval:+.3e}")
            item_uval.setFlags(item_uval.flags() & ~Qt.ItemIsEditable)
            table_widget.setItem(i, 2, item_uval)
        def on_radio_selected(id):
            highlight_probe(id)
        radio_group.idClicked.connect(on_radio_selected)
        if radio_group.buttons():
            radio_group.buttons()[0].setChecked(True)
            highlight_probe(0)
        dock = QDockWidget("Probe Table", plotter.app_window)
        dock.setWidget(table_widget)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setFloating(False)
        dock.show()
        plotter.app_window.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.raise_()
    except Exception as e:
        print(f"[error] Failed to create or dock table: {e}")
    
    # Return sphere info for show/hide functionality
    return sphere_actor_names, spheres

def infer_title(path: Path) -> str:
    """
    Infer a compact window title from the input filename.

    Examples
    --------
    - 'solution_v9_on.vtu' → 'v9_on'
    - 'head.vtu'           → 'head'
    """
    stem = path.stem
    return stem[len("solution_"):] if stem.startswith("solution_") and len(stem) > len("solution_") else stem

def outer_surface_of(vol_grid: pv.UnstructuredGrid) -> pv.PolyData | None:
    """
    Extract the external surface of a volumetric grid.

    Returns
    -------
    pv.PolyData or None
        The extracted surface, or None if extraction fails or the grid is empty.
    """
    if vol_grid is None or vol_grid.n_cells == 0:
        return None
    try:
        return vol_grid.extract_surface(pass_pointid=True, pass_cellid=True)
    except Exception:
        return None

def extract_by_cell_tag(grid: pv.UnstructuredGrid, name: str, value: int) -> pv.UnstructuredGrid | None:
    """
    Extract a subgrid containing only cells whose cell_data[name] equals `value`.

    Parameters
    ----------
    grid : pv.UnstructuredGrid
        Input grid.
    name : str
        Cell data array name (e.g., 'region_id').
    value : int
        Desired tag value (e.g., 1=gel, 2=head).

    Returns
    -------
    pv.UnstructuredGrid or None
        Subgrid with matching cells, or None if not found.
    """
    if grid is None or name not in grid.cell_data:
        return None
    mask = (grid.cell_data[name] == value)
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return None
    return grid.extract_cells(idx)

def split_blocks(grid: pv.UnstructuredGrid):
    """
    Split a mixed VTU grid into volume vs triangle blocks.

    Returns
    -------
    (pv.UnstructuredGrid or None, pv.UnstructuredGrid or None)
        vol_grid: tetra/tetra10 cells (volume)
        tri_grid: triangle cells (internal electrode surfaces)
    """
    vol_idx = np.nonzero(np.isin(grid.celltypes, [10, 24]))[0]
    tri_idx = np.nonzero(grid.celltypes == 5)[0]
    vol_grid = grid.extract_cells(vol_idx) if vol_idx.size else None
    tri_grid = grid.extract_cells(tri_idx) if tri_idx.size else None
    return vol_grid, tri_grid

def safe_name(s: str) -> str:
    """Sanitize a string to be filesystem-safe (letters, numbers, underscore, dash)."""
    s = re.sub(r"[^\w\-]+", "_", (s or "").strip())
    return s.strip("_")

def case_folder_from_title(title: str) -> Path:
    """
    Return ./screenshots/<case_name>, creating it if missing.
    Uses the (sanitized) window title as folder. If it sanitizes to empty, use a timestamp.
    """
    shots_root = Path("screenshots")
    shots_root.mkdir(exist_ok=True)
    name = safe_name(title)
    if not name:
        name = datetime.now().strftime("case_%Y%m%d_%H%M%S")
    folder = shots_root / name
    folder.mkdir(exist_ok=True)
    return folder

def unique_shot_path(folder: Path, base: str) -> Path:
    """Build <base>_<YYYYmmdd_HHMMSS_mmm>.png inside the given folder."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
    fname = f"{safe_name(base) or 'shot'}_{ts}.png"
    return folder / fname


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


from typing import List, Tuple, Dict, Any

def create_actors_spec(
    args: argparse.Namespace,
    head_surf: pv.PolyData | None,
    head_vol: pv.UnstructuredGrid | None,
    gel_surf: pv.PolyData | None,
    gel_vol: pv.UnstructuredGrid | None,
    tri_grid: pv.UnstructuredGrid | None
) -> list:
    """
    Build the list of actors to render based on CLI arguments and available geometry.
    
    WHY THIS EXISTS
    ---------------
    Centralizes actor specification logic to make the rendering pipeline declarative.
    Instead of scattered add_mesh() calls, we build a structured list of (key, dataset, scalar,
    cmap, clim, label) tuples that can be processed uniformly by the rendering loop.
    
    KEY DESIGN DECISIONS
    --------------------
    1. Adaptive colormap selection:
       - center_zero mode → diverging colormap (PRGn) for data centered at zero
       - Normal mode → sequential colormap (viridis) for positive-dominant data
    
    2. Global range mode behavior:
       - All actors use HEAD's colormap (ensures visual consistency)
       - Gel and electrodes inherit head_cmap (they're on the same scale)
    
    3. Electrodes follow gel colormap:
       - Electrodes are part of the gel-electrode system
       - Sharing colormap emphasizes they're on the same voltage scale
    
    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments (head mode, gel mode, colormaps, global_range, etc.)
    head_surf, head_vol : pv.PolyData | None
        Head surface/volume meshes extracted from solution
    gel_surf, gel_vol : pv.PolyData | None
        Gel surface/volume meshes
    tri_grid : pv.UnstructuredGrid | None
        Triangle mesh for electrode surfaces
    
    Returns
    -------
    list of tuples
        Each tuple: (key, dataset, scalar, cmap, clim, label)
        - key: unique identifier like "head_surf", "gel_vol", "elec"
        - dataset: PyVista mesh object
        - scalar: array name for coloring (usually "u")
        - cmap: colormap name
        - clim: color limits (None = auto from data)
        - label: human-readable label for UI
    """
    # Determine default colormap based on center_zero setting
    if args.center_zero:
        default_cmap = "PRGn"  # Diverging colormap for centered data
    else:
        default_cmap = "viridis"  # Sequential colormap for general data
    
    # Head colormap: use specified or default
    head_cmap = args.cmap_head if args.cmap_head is not None else default_cmap
    
    # Gel colormap: use specified, or default to head's colormap
    gel_cmap = args.cmap_gel if args.cmap_gel is not None else head_cmap
    
    # With global_range, all actors use the same colormap (head's colormap)
    if args.global_range:
        gel_cmap = head_cmap
    
    # (key, dataset, scalar, cmap, clim, label)
    actors_spec = []
    if args.head == "surface" and head_surf is not None:
        actors_spec.append(("head_surf", head_surf, "u", head_cmap, None, "Head potential (V)"))
    elif args.head == "volume" and head_vol is not None:
        actors_spec.append(("head_vol", head_vol, "u", head_cmap, None, "Head potential (V)"))
    if args.gel == "surface" and gel_surf is not None:
        actors_spec.append(("gel_surf", gel_surf, "u", gel_cmap, None, "Gel potential (V)"))
    elif args.gel == "volume" and gel_vol is not None:
        actors_spec.append(("gel_vol", gel_vol, "u", gel_cmap, None, "Gel potential (V)"))
    if not args.no_elec and tri_grid is not None and tri_grid.n_cells > 0:
        # Electrodes always use the same colormap as gel (they're part of the same system)
        actors_spec.append(("elec", tri_grid, "u", gel_cmap, None, "Electrodes (V)"))
    return actors_spec

def add_actor(
    plotter: BackgroundPlotter,
    ds: pv.DataSet,
    scalar: str,
    cmap: str,
    clim: tuple | None,
    label: str,
    edge_state: dict,
    show_scalar_bar: bool = True
) -> Any:
    """
    Add a single dataset (actor) to the plotter with proper rendering settings.
    
    WHY THIS EXISTS
    ---------------
    Centralizes actor creation with consistent rendering properties. Key concerns:
    - VTK's default transparency settings often cause semi-transparent, washed-out surfaces
    - Lighting coefficients need tuning to avoid overly dark or overly bright meshes
    - Scalar bars need consistent styling and positioning
    - Edge visibility must be controllable across all actors uniformly
    
    KEY DESIGN DECISIONS
    --------------------
    1. Force opacity=1.0 explicitly:
       - VTK sometimes applies partial transparency despite opacity=1.0 in add_mesh()
       - Explicitly setting prop.opacity and backface_params ensures true opacity
    
    2. Lighting coefficients (diffuse=1.0, ambient=0.2, specular=0.0):
       - diffuse=1.0: Primary lighting component, provides depth perception
       - ambient=0.2: Prevents completely black shadows
       - specular=0.0: Eliminates distracting highlights on irregular meshes
    
    3. Scalar bar always attached to actor:
       - Allows independent control when global_range=False
       - Provides actor-specific colormaps and ranges
    
    Parameters
    ----------
    plotter : BackgroundPlotter
        PyVista plotter instance
    ds : pv.DataSet
        Mesh (PolyData or UnstructuredGrid) to render
    scalar : str
        Name of point/cell data array for coloring (usually "u")
    cmap : str
        Matplotlib colormap name
    clim : tuple | None
        (vmin, vmax) color limits, or None for auto-scaling
    label : str
        Scalar bar title (e.g., "Head potential (V)")
    edge_state : dict
        {"on": bool} - whether to show mesh edges
    show_scalar_bar : bool
        Whether to display scalar bar for this actor
    
    Returns
    -------
    VTK actor object
        The added mesh actor (for later manipulation)
    """
    # Build add_mesh kwargs
    has_scalars = scalar in ds.array_names or scalar in ds.point_data
    mesh_kwargs = {
        'scalars': scalar if has_scalars else None,
        'cmap': cmap,
        'show_edges': edge_state["on"],
        'show_scalar_bar': show_scalar_bar if has_scalars else False,
        'scalar_bar_args': {"title": label} if has_scalars and show_scalar_bar else None,
        'nan_opacity': 1.0,
        'use_transparency': False,
        'opacity': 1.0,
    }
    # Include clim if provided to lock the color range
    if clim is not None:
        mesh_kwargs['clim'] = clim
    
    a = plotter.add_mesh(ds, **mesh_kwargs)
    # Explicitly force VTK property values to ensure full opacity and
    # reasonable lighting so surfaces don't look 'washed out' or semi-
    # transparent due to lighting/material configuration.
    try:
        prop = a.prop
        prop.opacity = 1.0
        # Make backfaces opaque as well
        prop.backface_params = {"opacity": 1.0}
        # Set lighting coefficients to sensible defaults for opaque surfaces
        prop.lighting = True
        prop.diffuse = 1.0
        prop.ambient = 0.2
        prop.specular = 0.0
    except Exception:
        # If actor.prop is unavailable for some reason, ignore and continue
        pass
    if clim is not None and (scalar in ds.array_names or scalar in ds.point_data):
        a.mapper.scalar_range = tuple(clim)
    return a

def setup_plotter_and_actors(
    actors_spec: List[Tuple[str, Any, str, str, Any, str]],
    args: argparse.Namespace,
    title: str,
    static_ranges: dict = None,
    unit: str = "V"
) -> Tuple[BackgroundPlotter, Dict[str, Any], dict]:
    """Create the BackgroundPlotter, add actors, and return plotter and actors_added dict."""
    plotter = BackgroundPlotter(window_size=(1200, 900), title=title, show=True, auto_update=True)
    plotter.remove_bounding_box()
    plotter.add_axes()
    
    # ──────────────────────────────────────────────────────────────────────────
    # CAMERA BUTTON LABEL REFLECTION
    # ──────────────────────────────────────────────────────────────────────────
    # WHY: The CAD model has Y-axis inverted relative to anatomical orientation
    # RESULT: "+Y direction" in CAD points toward anatomical FRONT (face)
    #         "-Y direction" in CAD points toward anatomical BACK (occiput)
    # SOLUTION: Reflect button labels so users see anatomically correct directions
    #           - "Back (+Y)" button → renamed to "Front (+Y)"
    #           - "Front (-Y)" button → renamed to "Back (-Y)"
    #           - "Isometric" view → also negated to match reflection
    # ──────────────────────────────────────────────────────────────────────────
    try:
        from PyQt5.QtWidgets import QToolBar, QAction
        toolbars = plotter.app_window.findChildren(QToolBar)
        for tb in toolbars:
            for action in tb.actions():
                text = action.text()
                if text == "Back (+Y)":
                    action.setText("Front (+Y)")
                elif text == "Front (-Y)":
                    action.setText("Back (-Y)")
                elif text == "Isometric":
                    # Override isometric view to be reflected for backwards head model
                    action.triggered.disconnect()
                    action.triggered.connect(lambda: plotter.view_isometric(negative=True))
    except Exception as e:
        print(f"[warn] Could not update camera button labels: {e}")
    edge_state = {"on": False}
    actors_added = {}
    
    # Determine which actors are enabled (present in spec)
    gel_enabled = any(k.startswith("gel") for k, _, _, _, _, _ in actors_spec)
    elec_enabled = any(k == "elec" for k, _, _, _, _, _ in actors_spec)
    
    # ──────────────────────────────────────────────────────────────────────────
    # ADAPTIVE COLORBAR LABELS
    # ──────────────────────────────────────────────────────────────────────────
    # WHY: Reduce UI clutter and provide intuitive labeling based on what's visible
    #
    # STRATEGY:
    # - Global range mode: Single colorbar "Voltage (μV)" for all actors
    # - Independent mode: Smart labels adapt to enabled actors:
    #   * Head always gets its own bar: "Head (μV)"
    #   * Gel + Electrodes share a bar when both enabled: "Gel + Elec (μV)"
    #   * Each gets separate bar when alone: "Gel (μV)" or "Elec (μV)"
    #
    # RATIONALE:
    # - Global mode: All actors on same scale → single bar sufficient
    # - Independent mode: Each region has different scale → separate bars needed
    # - Gel+Elec share bar: They're physically coupled → same voltage scale
    # ──────────────────────────────────────────────────────────────────────────
    for i, spec in enumerate(actors_spec):
        key, ds, scalar, cmap, clim, original_label = spec
        
        # Use static range if available, otherwise use clim from spec
        if static_ranges and args.static_bar:
            actor_clim = static_ranges.get(key, clim)
        else:
            actor_clim = clim
        
        # Determine smart adaptive labels and scalar bar visibility
        if args.global_range:
            # Global range mode: only show bar for first actor (all use same scale)
            show_scalar_bar = (i == 0)
            label = f"Voltage ({unit})"
        else:
            # Independent ranges mode: smart adaptive labels based on enabled actors
            if key.startswith("head"):
                show_scalar_bar = True
                label = f"Head ({unit})"
            elif key.startswith("gel"):
                show_scalar_bar = True
                # Gel label adapts based on whether electrodes are enabled
                if elec_enabled:
                    label = f"Gel + Elec ({unit})"
                else:
                    label = f"Gel ({unit})"
            elif key == "elec":
                # Electrodes share gel's bar only if gel is enabled
                show_scalar_bar = not gel_enabled
                # Elec label adapts based on whether gel is enabled
                if gel_enabled:
                    label = f"Gel + Elec ({unit})"
                else:
                    label = f"Elec ({unit})"
            else:
                show_scalar_bar = True
                label = original_label
        
        a = add_actor(plotter, ds, scalar, cmap, actor_clim, label, edge_state, show_scalar_bar=show_scalar_bar)
        if actor_clim is not None and (scalar in ds.array_names or scalar in ds.point_data):
            a.mapper.scalar_range = tuple(actor_clim)
        actors_added[key] = a
    
    return plotter, actors_added, edge_state
    return plotter, actors_added, edge_state

def main():
    """
    Entry point: load VTU, prepare actors, and launch the PyVista window.
    Loads the mesh, sets up actors and plotter, adds pins/table if requested, and starts the Qt event loop.
    """
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"✗ File not found: {in_path}")
        sys.exit(1)
    grid = pv.read(in_path)
    # Keep a copy of original grid for probe table (unscaled values)
    original_grid = grid.copy()
    title = args.title or infer_title(in_path)
    shots_folder = case_folder_from_title(title)
    vol_grid, tri_grid = split_blocks(grid)
    head_vol = extract_by_cell_tag(vol_grid, "region_id", 2)
    gel_vol  = extract_by_cell_tag(vol_grid, "region_id", 1)
    head_surf = outer_surface_of(head_vol) if head_vol is not None else None
    gel_surf  = outer_surface_of(gel_vol)  if gel_vol  is not None else None
    actors_spec = create_actors_spec(args, head_surf, head_vol, gel_surf, gel_vol, tri_grid)
    if not actors_spec:
        print("[warn] Nothing to show. Are 'u' and 'region_id' present?")
        print(f" arrays: point_data={list(grid.point_data.keys())}, cell_data={list(grid.cell_data.keys())}")
        sys.exit(0)
    # Determine appropriate scale for colorbar display, keep table in V (scientific)
    if args.pins:
        pins_data = read_pins_csv(args.pins)
    
    # Determine best unit/scale for colorbar based on data range
    u_all_values = []
    if 'u' in grid.point_data:
        u_all_values.extend(grid.point_data['u'])
    if 'u' in grid.cell_data:
        u_all_values.extend(grid.cell_data['u'])
    
    abs_max = max(abs(float(u)) for u in u_all_values) if u_all_values else 0
    if abs_max < 1e-3:
        scale = 1e6
        unit = "μV"
    elif abs_max < 1e-1:
        scale = 1e3
        unit = "mV"
    else:
        scale = 1.0
        unit = "V"
    
    # Scale the 'u' data for better colorbar readability
    for arr in [grid.point_data, grid.cell_data]:
        if 'u' in arr:
            arr['u'] = arr['u'] * scale
    
    # Also scale in all actor datasets
    for spec in actors_spec:
        ds = spec[1]
        if hasattr(ds, 'point_data') and 'u' in ds.point_data:
            ds.point_data['u'] = ds.point_data['u'] * scale
        if hasattr(ds, 'cell_data') and 'u' in ds.cell_data:
            ds.cell_data['u'] = ds.cell_data['u'] * scale
    
    # Update colorbar labels to appropriate unit
    new_actors_spec = []
    for spec in actors_spec:
        key, ds, scalar, cmap, clim, label = spec
        # Change colorbar labels to appropriate unit
        if '(V)' in label:
            label = label.replace('(V)', f'({unit})')
        new_actors_spec.append((key, ds, scalar, cmap, clim, label))
    
    # Capture static color ranges (if enabled) to preserve during clipping.
    # IMPORTANT: Calculate ranges from ORIGINAL unscaled data, then apply scale factor.
    static_ranges = {}
    if args.static_bar:
        # Build temp specs with ORIGINAL unscaled data for range calculation
        original_vol_grid, original_tri_grid = split_blocks(original_grid)
        original_head_vol = extract_by_cell_tag(original_vol_grid, "region_id", 2)
        original_gel_vol = extract_by_cell_tag(original_vol_grid, "region_id", 1)
        original_head_surf = outer_surface_of(original_head_vol) if original_head_vol is not None else None
        original_gel_surf = outer_surface_of(original_gel_vol) if original_gel_vol is not None else None
        original_actors_spec = create_actors_spec(args, original_head_surf, original_head_vol, 
                                                   original_gel_surf, original_gel_vol, original_tri_grid)
        
        if args.global_range:
            # Calculate global min/max across ALL actors from ORIGINAL unscaled data
            global_min = float('inf')
            global_max = float('-inf')
            for spec in original_actors_spec:
                key, ds, scalar, cmap, clim, label = spec
                if scalar in getattr(ds, 'point_data', {}):
                    arr = ds.point_data[scalar]
                    global_min = min(global_min, float(np.min(arr)))
                    global_max = max(global_max, float(np.max(arr)))
                elif scalar in getattr(ds, 'cell_data', {}):
                    arr = ds.cell_data[scalar]
                    global_min = min(global_min, float(np.min(arr)))
                    global_max = max(global_max, float(np.max(arr)))
                elif scalar in ds.array_names:
                    arr = ds[scalar]
                    global_min = min(global_min, float(np.min(arr)))
                    global_max = max(global_max, float(np.max(arr)))
            
            # Apply scale factor AFTER calculating range from original data
            global_min *= scale
            global_max *= scale
            
            # If center_zero is enabled, make range symmetric around zero
            if args.center_zero:
                max_abs = max(abs(global_min), abs(global_max))
                global_min, global_max = -max_abs, max_abs
                print(f"[center-zero] Using symmetric global range: [{global_min:.4f}, {global_max:.4f}] {unit}")
            else:
                print(f"[global-range] Using global range: [{global_min:.4f}, {global_max:.4f}] {unit}")
            
            # Apply global range to all actors
            for spec in new_actors_spec:
                key = spec[0]
                static_ranges[key] = (global_min, global_max)
        else:
            # Independent ranges: Calculate per-actor from ORIGINAL unscaled data
            for original_spec, new_spec in zip(original_actors_spec, new_actors_spec):
                original_key, original_ds, scalar, _, _, _ = original_spec
                key = new_spec[0]
                
                # Calculate range from original unscaled data
                if scalar in getattr(original_ds, 'point_data', {}):
                    arr = original_ds.point_data[scalar]
                    range_min, range_max = float(np.min(arr)), float(np.max(arr))
                elif scalar in getattr(original_ds, 'cell_data', {}):
                    arr = original_ds.cell_data[scalar]
                    range_min, range_max = float(np.min(arr)), float(np.max(arr))
                elif scalar in original_ds.array_names:
                    arr = original_ds[scalar]
                    range_min, range_max = float(np.min(arr)), float(np.max(arr))
                else:
                    continue
                
                # Apply scale factor AFTER calculating range from original data
                range_min *= scale
                range_max *= scale
                
                # If center_zero is enabled, make range symmetric around zero
                if args.center_zero:
                    max_abs = max(abs(range_min), abs(range_max))
                    range_min, range_max = -max_abs, max_abs
                
                static_ranges[key] = (range_min, range_max)
    
    plotter, actors_added, edge_state = setup_plotter_and_actors(new_actors_spec, args, title, static_ranges, unit)
    
    # Determine colormap for probe colors (use head colormap)
    if args.center_zero:
        probe_cmap = args.cmap_head if args.cmap_head is not None else "PRGn"
    else:
        probe_cmap = args.cmap_head if args.cmap_head is not None else "viridis"
    
    # Add probe spheres if available
    probe_sphere_names = []
    if args.pins:
        probe_sphere_names = add_pin_spheres_and_table(plotter, original_grid, pins_data, probe_cmap)

    # Pre-compute electrode centers for quick plane positioning
    electrode_centers = {}
    if tri_grid is not None and 'facet_id' in tri_grid.cell_data:
        for elec_id in range(1, 10):  # Electrodes 1-9
            facet_ids = tri_grid.cell_data['facet_id']
            elec_mask = (facet_ids == elec_id)
            if elec_mask.any():
                elec_cells = tri_grid.extract_cells(np.nonzero(elec_mask)[0])
                if elec_cells.n_points > 0:
                    electrode_centers[elec_id] = elec_cells.points.mean(axis=0)
                    print(f"[init] Electrode {elec_id} center: {electrode_centers[elec_id]}")

    # --- Clip logic: plane clip with point-normal specification ---
    bounds = grid.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
    bbox_center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
    bbox_extents = [(bounds[1]-bounds[0]), (bounds[3]-bounds[2]), (bounds[5]-bounds[4])]
    
    print("[bbox] Bounding box dimensions (in cm):")
    print(f"  X extent: {bbox_extents[0]*100:.2f} cm (width)")
    print(f"  Y extent: {bbox_extents[1]*100:.2f} cm (height)")
    print(f"  Z extent: {bbox_extents[2]*100:.2f} cm (depth)")
    print(f"  3D diagonal (corner-to-corner): {np.linalg.norm(bbox_extents)*100:.2f} cm")
    
    clip_state = {
        "plane_on": False,
        "normal": [1.0, 0.0, 0.0],     # normal vector (a, b, c)
        "origin": list(bbox_center),   # point on the plane (x, y, z)
        "use_bbox_mode": True,         # if True, use normalized position; if False, use explicit origin
        "pos": 0.5,                     # normalized 0..1 position (only used when use_bbox_mode=True)
    }
    bbox_diagonal = np.linalg.norm(bbox_extents)
    
    def get_bbox_projection_length(normal: list):
        """Calculate the total length of bbox projection along the normal direction."""
        # Normalize normal
        n = np.array(normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            n = np.array([1.0, 0.0, 0.0])
        else:
            n = n / norm
        
        # Calculate the half-extents from center
        half_extents = np.array(bbox_extents) / 2.0
        
        # The projection of the bbox along the normal is the sum of absolute projections
        # of the half-extents in each axis direction
        projection_length = 2.0 * np.sum(np.abs(half_extents * n))
        
        return projection_length
    
    def world_coord_for_normal(normal: list, pos: float):
        """Return the origin point for the clip plane along the normal direction."""
        # Normalize the normal vector for calculation
        n = np.array(normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            n = np.array([1.0, 0.0, 0.0])
        else:
            n = n / norm
        # Map normalized position [0,1] to a range along the normal
        # from one side of the bbox to the other
        offset = (pos - 0.5) * bbox_diagonal
        origin = bbox_center + n * offset
        return origin

    def apply_clip_to_ds(ds: pv.DataSet) -> pv.DataSet:
        """Return a clipped copy of dataset according to clip_state."""
        if not clip_state['plane_on']:
            return ds
        
        result = ds
        
        try:
            # Apply plane clip
            normal = clip_state['normal']
            if clip_state['use_bbox_mode']:
                # Use normalized position mode (slider-based)
                origin = world_coord_for_normal(normal, clip_state['pos'])
            else:
                # Use explicit origin point
                origin = clip_state['origin']
            
            print(f"[clip] Plane clip: normal={normal}, origin={origin}")
            result = result.clip(normal=tuple(normal), origin=tuple(origin))
            
            return result
            
        except Exception as e:
            print(f"[clip] clip failed: {e}")
            return ds

    def update_actors_for_clip():
        """Recreate actors using clipped datasets to reflect current clip state."""
        # Show status message during clipping
        if clip_state['plane_on']:
            try:
                plotter.add_text("Applying clip...", position='upper_left', font_size=14, name='clip_status', color='yellow')
                plotter.render()
            except Exception:
                pass
        
        # Determine which actors are enabled (present in spec)
        gel_enabled = any(k.startswith("gel") for k, _, _, _, _, _ in new_actors_spec)
        elec_enabled = any(k == "elec" for k, _, _, _, _, _ in new_actors_spec)
        
        # For dynamic ranges with center_zero, need to calculate ranges from clipped data
        dynamic_ranges = {}
        if not args.static_bar and args.center_zero:
            # Calculate symmetric ranges from clipped data
            if args.global_range:
                # Global range across all clipped actors
                global_min = float('inf')
                global_max = float('-inf')
                for spec in new_actors_spec:
                    key, ds, scalar, _, _, _ = spec
                    ds_clipped = apply_clip_to_ds(ds)
                    if ds_clipped.n_points == 0:
                        continue
                    if scalar in getattr(ds_clipped, 'point_data', {}):
                        arr = ds_clipped.point_data[scalar]
                        global_min = min(global_min, float(np.min(arr)))
                        global_max = max(global_max, float(np.max(arr)))
                    elif scalar in getattr(ds_clipped, 'cell_data', {}):
                        arr = ds_clipped.cell_data[scalar]
                        global_min = min(global_min, float(np.min(arr)))
                        global_max = max(global_max, float(np.max(arr)))
                
                # Make symmetric
                max_abs = max(abs(global_min), abs(global_max))
                global_min, global_max = -max_abs, max_abs
                
                # Apply to all actors
                for spec in new_actors_spec:
                    dynamic_ranges[spec[0]] = (global_min, global_max)
            else:
                # Independent ranges per actor
                for spec in new_actors_spec:
                    key, ds, scalar, _, _, _ = spec
                    ds_clipped = apply_clip_to_ds(ds)
                    if ds_clipped.n_points == 0:
                        continue
                    if scalar in getattr(ds_clipped, 'point_data', {}):
                        arr = ds_clipped.point_data[scalar]
                        range_min, range_max = float(np.min(arr)), float(np.max(arr))
                    elif scalar in getattr(ds_clipped, 'cell_data', {}):
                        arr = ds_clipped.cell_data[scalar]
                        range_min, range_max = float(np.min(arr)), float(np.max(arr))
                    else:
                        continue
                    
                    # Make symmetric
                    max_abs = max(abs(range_min), abs(range_max))
                    range_min, range_max = -max_abs, max_abs
                    dynamic_ranges[key] = (range_min, range_max)
        
        for i, spec in enumerate(new_actors_spec):
            key, ds, scalar, cmap, clim, original_label = spec
            try:
                ds_clipped = apply_clip_to_ds(ds)
                # Skip if clipped dataset is empty (keep last valid actor state)
                if ds_clipped.n_points == 0:
                    print(f"[clip] Skipping {key} (empty after clip)")
                    continue
                old = actors_added.get(key)
                # remove old actor (object or name) without resetting camera
                try:
                    if old is not None:
                        plotter.remove_actor(old, reset_camera=False, render=False)
                    else:
                        plotter.remove_actor(key, reset_camera=False, render=False)
                except Exception:
                    pass
                
                # Determine which range to use
                if args.static_bar:
                    # Static range mode: use pre-calculated static ranges
                    actor_clim = static_ranges.get(key, clim)
                elif args.center_zero:
                    # Dynamic range mode with center-zero: use calculated symmetric ranges
                    actor_clim = dynamic_ranges.get(key, clim)
                else:
                    # Dynamic range mode without center-zero: let PyVista auto-calculate
                    actor_clim = clim
                
                # Determine smart adaptive labels and scalar bar visibility (same logic as setup)
                if args.global_range:
                    # Global range mode: only show bar for first actor
                    show_scalar_bar = (i == 0)
                    label = f"Voltage ({unit})"
                else:
                    # Independent ranges mode: smart adaptive labels based on enabled actors
                    if key.startswith("head"):
                        show_scalar_bar = True
                        label = f"Head ({unit})"
                    elif key.startswith("gel"):
                        show_scalar_bar = True
                        # Gel label adapts based on whether electrodes are enabled
                        if elec_enabled:
                            label = f"Gel + Elec ({unit})"
                        else:
                            label = f"Gel ({unit})"
                    elif key == "elec":
                        # Electrodes share gel's bar only if gel is enabled
                        show_scalar_bar = not gel_enabled
                        # Elec label adapts based on whether gel is enabled
                        if gel_enabled:
                            label = f"Gel + Elec ({unit})"
                        else:
                            label = f"Elec ({unit})"
                    else:
                        show_scalar_bar = True
                        label = original_label
                
                a = add_actor(plotter, ds_clipped, scalar, cmap, actor_clim, label, edge_state, show_scalar_bar=show_scalar_bar)
                if actor_clim is not None and (scalar in ds_clipped.array_names or scalar in getattr(ds_clipped, 'point_data', {})):
                    a.mapper.scalar_range = tuple(actor_clim)
                actors_added[key] = a
            except Exception as e:
                print(f"[clip] could not update actor {key}: {e}")
        
        # Remove status message
        try:
            plotter.remove_actor('clip_status', reset_camera=False, render=False)
        except Exception:
            pass
        
        plotter.render()

    def clip_status():
        status = "[clip]"
        if clip_state['plane_on']:
            if clip_state['use_bbox_mode']:
                origin = world_coord_for_normal(clip_state['normal'], clip_state['pos'])
                status += f" Plane: normal={clip_state['normal']} pos={clip_state['pos']:.2f} origin={origin}"
            else:
                status += f" Plane: normal={clip_state['normal']} origin={clip_state['origin']}"
        else:
            status += " OFF"
        return status
    
    def recalculate_static_ranges():
        """
        Recalculate static color ranges based on current global_range and center_zero settings.
        
        WHY CALCULATE FROM ORIGINAL GRID
        ---------------------------------
        CRITICAL: Ranges are calculated from original_grid (unscaled Volts), then multiplied by scale.
        This is essential for precision when working with μV values:
        
        - μV values are tiny (e.g., 1e-6 V = 1 μV)
        - Calculating ranges from already-scaled data compounds floating-point errors
        - Strategy: Calculate in Volts (clean range), then scale to display units
        - Example: [1e-6, 5e-6] V → calculate range → multiply by 1e6 → [1, 5] μV
        
        This ensures color ranges are numerically stable and match the original simulation precision.
        """
        nonlocal static_ranges
        static_ranges = {}
        
        # STEP 1: Extract ORIGINAL unscaled datasets (region_id for volumes, facet_id for surfaces)
        # WHY: Working with original Volt values prevents precision loss from premature scaling
        original_vol_grid, original_tri_grid = split_blocks(original_grid)
        temp_actors = []
        if args.head != 'off':
            # Extract head region (region_id=2) from volume grid
            head_ds = original_vol_grid.threshold((1.5, 2.5), scalars='region_id', method='threshold')
            if head_ds.n_points > 0:
                temp_actors.append(('head', head_ds))
        if args.gel != 'off':
            # Extract gel region (region_id=1) from volume grid
            gel_ds = original_vol_grid.threshold((0.5, 1.5), scalars='region_id', method='threshold')
            if gel_ds.n_points > 0:
                temp_actors.append(('gel', gel_ds))
        if not args.no_elec:
            # Extract electrode surfaces from triangle grid (all facet_ids 1-9)
            elec_ds = original_tri_grid.threshold((0.5, 9.5), scalars='facet_id', method='threshold')
            if elec_ds.n_points > 0:
                temp_actors.append(('elec', elec_ds))
        
        # STEP 2: Calculate color ranges based on mode
        if args.global_range:
            # GLOBAL RANGE MODE: Single range across all actors
            # WHY: Ensures visual consistency - same color = same voltage across all geometry
            # HOW: Find min/max across all enabled actors, then apply to everyone
            global_min = float('inf')
            global_max = float('-inf')
            for key, ds in temp_actors:
                if 'u' in ds.point_data:
                    arr = ds.point_data['u']
                    global_min = min(global_min, float(np.min(arr)))
                    global_max = max(global_max, float(np.max(arr)))
                elif 'u' in ds.cell_data:
                    arr = ds.cell_data['u']
                    global_min = min(global_min, float(np.min(arr)))
                    global_max = max(global_max, float(np.max(arr)))
            
            # STEP 3: Apply scale factor AFTER calculating from original Volt data
            # WHY: Maintains numerical precision for tiny μV values
            global_min *= scale
            global_max *= scale
            
            # STEP 4: Apply center-zero symmetry if requested
            # WHY: Diverging colormaps need symmetric ranges to display zero at colormap center
            if args.center_zero:
                max_abs = max(abs(global_min), abs(global_max))
                global_min, global_max = -max_abs, max_abs
                print(f"[center-zero] Recalculated symmetric global range: [{global_min:.4f}, {global_max:.4f}] {unit}")
            else:
                print(f"[global-range] Recalculated global range: [{global_min:.4f}, {global_max:.4f}] {unit}")
            
            # STEP 5: Apply global range to all possible actor keys
            # WHY: Ensures consistent coloring even if actors are toggled on/off later
            for key in ['head_surf', 'head_vol', 'gel_surf', 'gel_vol', 'elec']:
                static_ranges[key] = (global_min, global_max)
        else:
            # INDEPENDENT RANGE MODE: Each actor gets its own optimal range
            # WHY: Maximizes color contrast within each geometry (better for comparing regions)
            # TRADEOFF: Same color may represent different voltages in different actors
            for key, ds in temp_actors:
                if 'u' in ds.point_data:
                    arr = ds.point_data['u']
                    range_min, range_max = float(np.min(arr)), float(np.max(arr))
                elif 'u' in ds.cell_data:
                    arr = ds.cell_data['u']
                    range_min, range_max = float(np.min(arr)), float(np.max(arr))
                else:
                    continue
                
                # Apply scale factor AFTER calculating range from original data
                range_min *= scale
                range_max *= scale
                
                # If center_zero is enabled, make range symmetric around zero
                if args.center_zero:
                    max_abs = max(abs(range_min), abs(range_max))
                    range_min, range_max = -max_abs, max_abs
                
                range_val = (range_min, range_max)
                
                # Map temp keys to actual actor keys based on surface/volume mode
                if key == 'head':
                    if args.head == 'surface':
                        static_ranges['head_surf'] = range_val
                    else:
                        static_ranges['head_vol'] = range_val
                elif key == 'gel':
                    if args.gel == 'surface':
                        static_ranges['gel_surf'] = range_val
                    else:
                        static_ranges['gel_vol'] = range_val
                elif key == 'elec':
                    static_ranges['elec'] = range_val

    def update_actors_spec():
        """Rebuild actors_spec based on current args state, then update all actors"""
        nonlocal new_actors_spec
        new_actors_spec = []
        
        # Determine default colormap based on center_zero setting
        if args.center_zero:
            default_cmap = "PRGn"  # Diverging colormap for centered data
        else:
            default_cmap = "viridis"  # Sequential colormap for general data
        
        # Head colormap: use specified or default
        head_cmap = args.cmap_head if args.cmap_head is not None else default_cmap
        
        # Gel colormap: use specified, or default to head's colormap
        gel_cmap = args.cmap_gel if args.cmap_gel is not None else head_cmap
        
        # With global_range, all actors use the same colormap (head's colormap)
        if args.global_range:
            gel_cmap = head_cmap
        
        # Head
        if args.head == 'surface':
            head_surf = vol_grid.threshold((1.5, 2.5), scalars='region_id', method='threshold')
            if head_surf.n_points > 0:
                head_surf = head_surf.extract_surface()
                clim = static_ranges.get('head_surf') if args.static_bar else None
                new_actors_spec.append(('head_surf', head_surf, 'u', head_cmap, clim, 'Head'))
        elif args.head == 'volume':
            head_vol = vol_grid.threshold((1.5, 2.5), scalars='region_id', method='threshold')
            if head_vol.n_points > 0:
                clim = static_ranges.get('head_vol') if args.static_bar else None
                new_actors_spec.append(('head_vol', head_vol, 'u', head_cmap, clim, 'Head'))
        
        # Gel
        if args.gel == 'surface':
            gel_surf = vol_grid.threshold((0.5, 1.5), scalars='region_id', method='threshold')
            if gel_surf.n_points > 0:
                gel_surf = gel_surf.extract_surface()
                clim = static_ranges.get('gel_surf') if args.static_bar else None
                new_actors_spec.append(('gel_surf', gel_surf, 'u', gel_cmap, clim, 'Gel'))
        elif args.gel == 'volume':
            gel_vol = vol_grid.threshold((0.5, 1.5), scalars='region_id', method='threshold')
            if gel_vol.n_points > 0:
                clim = static_ranges.get('gel_vol') if args.static_bar else None
                new_actors_spec.append(('gel_vol', gel_vol, 'u', gel_cmap, clim, 'Gel'))
        
        # Electrodes (surface only, from triangle mesh)
        # Electrodes always use the same colormap as gel (they're part of the same system)
        if not args.no_elec:
            elec_grid = tri_grid.threshold((0.5, 9.5), scalars='facet_id', method='threshold')
            if elec_grid.n_points > 0:
                clim = static_ranges.get('elec') if args.static_bar else None
                new_actors_spec.append(('elec', elec_grid, 'u', gel_cmap, clim, 'Electrodes'))
        
        # Remove old actors
        for key in actors_added.keys():
            plotter.remove_actor(actors_added[key])
        actors_added.clear()
        
        # Add new actors (always apply clip state)
        update_actors_for_clip()
    
    # Loading indicator helpers
    def show_loading(message="Processing..."):
        """Show loading indicator with message"""
        loading_label.setText(f"⏳ {message}")
        plotter.app.processEvents()  # Force UI update
    
    def hide_loading():
        """Hide loading indicator"""
        loading_label.setText('')
        plotter.app.processEvents()  # Force UI update
    
    def update_warning_visibility():
        """Show/hide warning based on whether all actors are disabled"""
        all_off = (args.head == 'off' and args.gel == 'off' and args.no_elec)
        warning_label.setVisible(all_off)
    
    def change_head_mode():
        """Update head display mode based on radio button selection"""
        show_loading("Updating head display...")
        if head_off.isChecked():
            args.head = 'off'
        elif head_surf.isChecked():
            args.head = 'surface'
        else:
            args.head = 'volume'
        print(f"[actor] Head mode changed to: {args.head}")
        
        # Recalculate static ranges if needed (especially for global range mode)
        if args.static_bar:
            recalculate_static_ranges()
        
        update_actors_spec()
        update_warning_visibility()
        hide_loading()
    
    def change_gel_mode():
        """Update gel display mode based on radio button selection"""
        show_loading("Updating gel display...")
        if gel_off.isChecked():
            args.gel = 'off'
        elif gel_surf.isChecked():
            args.gel = 'surface'
        else:
            args.gel = 'volume'
        print(f"[actor] Gel mode changed to: {args.gel}")
        
        # Recalculate static ranges if needed (especially for global range mode)
        if args.static_bar:
            recalculate_static_ranges()
        
        update_actors_spec()
        update_warning_visibility()
        hide_loading()
    
    def change_elec_visibility():
        """Update electrode visibility based on checkbox"""
        show_loading("Updating electrodes...")
        args.no_elec = not elec_cb.isChecked()
        print(f"[actor] Electrodes: {'shown' if not args.no_elec else 'hidden'}")
        
        # Recalculate static ranges if needed (especially for global range mode)
        if args.static_bar:
            recalculate_static_ranges()
        
        update_actors_spec()
        update_warning_visibility()
        hide_loading()
    
    def change_probe_visibility():
        """Toggle probe marker visibility based on checkbox"""
        show_probes = probe_cb.isChecked()
        print(f"[probe] Probe markers: {'shown' if show_probes else 'hidden'}")
        
        # Show/hide all probe spheres (both regular and highlighted)
        # PyVista stores actors by name in the renderer
        for name in probe_sphere_names:
            # Check both regular and highlighted versions
            for actor_name in [name, name + '_highlighted']:
                actor = plotter.renderer.actors.get(actor_name)
                if actor is not None:
                    actor.SetVisibility(show_probes)
        
        plotter.render()
    
    def change_global_range():
        """Toggle global range mode and recalculate static ranges"""
        show_loading("Updating color bar...")
        args.global_range = global_range_cb.isChecked()
        print(f"[color] Global range: {'enabled' if args.global_range else 'disabled'}")
        
        # Recalculate static ranges based on new mode
        if args.static_bar:
            recalculate_static_ranges()
        
        # Rebuild actors with new color scheme
        update_actors_spec()
        hide_loading()
    
    def change_static_bar():
        """Toggle static/dynamic range mode"""
        show_loading("Updating ranges...")
        args.static_bar = static_bar_cb.isChecked()
        print(f"[color] Static ranges: {'enabled' if args.static_bar else 'disabled (dynamic)'}")
        
        # If switching to static, recalculate ranges
        if args.static_bar:
            recalculate_static_ranges()
        
        # Rebuild actors with new range mode
        update_actors_spec()
        hide_loading()
    
    def change_center_zero():
        """Toggle center-zero symmetric range mode"""
        show_loading("Updating colormap...")
        args.center_zero = center_zero_cb.isChecked()
        print(f"[color] Center at zero: {'enabled' if args.center_zero else 'disabled'}")
        
        # Recalculate static ranges with new centering mode
        if args.static_bar:
            recalculate_static_ranges()
        
        # Rebuild actors with new colormap and range mode
        update_actors_spec()
        hide_loading()

    def set_clip_normal(a: float, b: float, c: float):
        show_loading("Updating clip plane...")
        clip_state['normal'] = [a, b, c]
        print(clip_status())
        update_actors_for_clip()
        try:
            if clip_ui_update:
                clip_ui_update()
        except Exception:
            pass
        hide_loading()
    
    def invert_clip_normal():
        show_loading("Inverting clip plane...")
        clip_state['normal'] = [-x for x in clip_state['normal']]
        print(clip_status())
        update_actors_for_clip()
        try:
            if clip_ui_update:
                clip_ui_update()
        except Exception:
            pass
        hide_loading()

    # Main controls dock with actor controls, plane and electrode clip modes
    clip_ui_update = None
    try:
        from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
                                      QPushButton, QDockWidget, QLineEdit, QFrame, QDoubleSpinBox, QRadioButton, QButtonGroup)
        from PyQt5.QtCore import Qt
        ctrl = QWidget()
        v = QVBoxLayout()
        
        # === ACTOR CONTROLS SECTION ===
        v.addWidget(QLabel('<b>Actor Controls</b>'))
        
        # Head mode selection
        head_row = QHBoxLayout()
        head_row.addWidget(QLabel('Head:'))
        head_group = QButtonGroup()
        head_off = QRadioButton('Off')
        head_surf = QRadioButton('Surface')
        head_vol = QRadioButton('Volume')
        head_group.addButton(head_off, 0)
        head_group.addButton(head_surf, 1)
        head_group.addButton(head_vol, 2)
        head_row.addWidget(head_off)
        head_row.addWidget(head_surf)
        head_row.addWidget(head_vol)
        head_row.addStretch()
        v.addLayout(head_row)
        
        # Set initial head state
        if args.head == 'off':
            head_off.setChecked(True)
        elif args.head == 'surface':
            head_surf.setChecked(True)
        else:  # 'volume'
            head_vol.setChecked(True)
        
        # Gel mode selection
        gel_row = QHBoxLayout()
        gel_row.addWidget(QLabel('Gel:'))
        gel_group = QButtonGroup()
        gel_off = QRadioButton('Off')
        gel_surf = QRadioButton('Surface')
        gel_vol = QRadioButton('Volume')
        gel_group.addButton(gel_off, 0)
        gel_group.addButton(gel_surf, 1)
        gel_group.addButton(gel_vol, 2)
        gel_row.addWidget(gel_off)
        gel_row.addWidget(gel_surf)
        gel_row.addWidget(gel_vol)
        gel_row.addStretch()
        v.addLayout(gel_row)
        
        # Set initial gel state
        if args.gel == 'off':
            gel_off.setChecked(True)
        elif args.gel == 'surface':
            gel_surf.setChecked(True)
        else:  # 'volume'
            gel_vol.setChecked(True)
        
        # Electrodes checkbox
        from PyQt5.QtWidgets import QCheckBox
        elec_cb = QCheckBox('Show Electrodes')
        elec_cb.setTristate(False)  # Disable tri-state to avoid multiple state changes
        elec_cb.setChecked(not args.no_elec)
        v.addWidget(elec_cb)
        
        # Probe markers checkbox
        probe_cb = QCheckBox('Show Probe Markers')
        probe_cb.setTristate(False)
        probe_cb.setChecked(True)  # Default: visible
        probe_cb.setEnabled(bool(probe_sphere_names))  # Only enable if probes exist
        v.addWidget(probe_cb)
 
        # Warning label for all-actors-off scenario (shown/hidden dynamically)
        warning_label = QLabel()
        warning_label.setText('All actors are OFF - nothing to display')
        warning_label.setStyleSheet("color: #FF8C00; font-weight: bold; font-size: 10pt; padding: 5px; background-color: #FFF3E0;")
        warning_label.setWordWrap(True)
        warning_label.setAlignment(Qt.AlignCenter)
        warning_label.setVisible(False)  # Hidden by default
        v.addWidget(warning_label)
        
        # Separator after actor controls
        sep_actor = QFrame()
        sep_actor.setFrameShape(QFrame.HLine)
        sep_actor.setFrameShadow(QFrame.Sunken)
        v.addWidget(sep_actor)
        
        # === COLOR BAR OPTIONS SECTION ===
        v.addWidget(QLabel('<b>Color Bar Options</b>'))
        
        # Global range checkbox with clearer label
        global_range_cb = QCheckBox('Global Range (compare across actors)')
        global_range_cb.setTristate(False)  # Disable tri-state to avoid multiple state changes
        global_range_cb.setChecked(args.global_range)
        v.addWidget(global_range_cb)
        
        # Static help text
        global_range_help = QLabel('  → When enabled: compare different actors. When disabled: enhance per-actor local changes')
        global_range_help.setStyleSheet("color: gray; font-size: 9pt;")
        global_range_help.setWordWrap(True)
        v.addWidget(global_range_help)
        
        # Static bar checkbox
        static_bar_cb = QCheckBox('Static Ranges (locked to full data)')
        static_bar_cb.setTristate(False)  # Disable tri-state to avoid multiple state changes
        static_bar_cb.setChecked(args.static_bar)
        v.addWidget(static_bar_cb)
        
        static_bar_help = QLabel('  → Ranges stay fixed even when clipping (based on enabled actors)')
        static_bar_help.setStyleSheet("color: gray; font-size: 9pt;")
        v.addWidget(static_bar_help)
        
        # Center zero checkbox
        center_zero_cb = QCheckBox('Center at Zero (symmetric ranges)')
        center_zero_cb.setTristate(False)  # Disable tri-state to avoid multiple state changes
        center_zero_cb.setChecked(args.center_zero)
        v.addWidget(center_zero_cb)
        
        center_zero_help = QLabel('  → Symmetric ranges like [-2, +2] with diverging colormap (PRGn)')
        center_zero_help.setStyleSheet("color: gray; font-size: 9pt;")
        v.addWidget(center_zero_help)
        
        # Separator after color bar options
        sep_colorbar = QFrame()
        sep_colorbar.setFrameShape(QFrame.HLine)
        sep_colorbar.setFrameShadow(QFrame.Sunken)
        v.addWidget(sep_colorbar)
        
        # === PLANE CLIP SECTION ===
        plane_header = QHBoxLayout()
        plane_header.addWidget(QLabel('<b>Plane Clip</b>'))
        btn_plane_toggle = QPushButton('Disable Clipping' if clip_state['plane_on'] else 'Enable Clipping')
        btn_plane_toggle.setCheckable(True)
        btn_plane_toggle.setChecked(clip_state['plane_on'])
        plane_header.addWidget(btn_plane_toggle)
        v.addLayout(plane_header)
        
        # Normal vector inputs with Flip button inline
        normal_row = QHBoxLayout()
        normal_row.addWidget(QLabel('Normal:'))
        edit_a = QLineEdit(str(clip_state['normal'][0]))
        edit_b = QLineEdit(str(clip_state['normal'][1]))
        edit_c = QLineEdit(str(clip_state['normal'][2]))
        edit_a.setFixedWidth(50)
        edit_b.setFixedWidth(50)
        edit_c.setFixedWidth(50)
        normal_row.addWidget(edit_a)
        normal_row.addWidget(edit_b)
        normal_row.addWidget(edit_c)
        btn_invert = QPushButton('Flip')
        btn_invert.setFixedWidth(50)
        normal_row.addWidget(btn_invert)
        v.addLayout(normal_row)
        
        # ──────────────────────────────────────────────────────────────────────
        # QUICK ANATOMICAL PLANE BUTTONS
        # ──────────────────────────────────────────────────────────────────────
        # WHY: Medical/neuroscience users think in anatomical planes, not Cartesian axes
        # Sagittal (YZ): Divides left/right hemispheres (slices parallel to midline)
        # Coronal (XZ): Divides front/back (slices parallel to face)
        # Axial (XY): Divides top/bottom (horizontal slices)
        # ──────────────────────────────────────────────────────────────────────
        v.addWidget(QLabel('Quick Planes:'))
        quick_row = QHBoxLayout()
        btn_x = QPushButton('Sagittal (YZ)')
        btn_y = QPushButton('Coronal (XZ)')
        btn_z = QPushButton('Axial (XY)')
        btn_x.setMinimumWidth(110)
        btn_y.setMinimumWidth(110)
        btn_z.setMinimumWidth(110)
        quick_row.addWidget(btn_x)
        quick_row.addWidget(btn_y)
        quick_row.addWidget(btn_z)
        v.addLayout(quick_row)
        
        # === TABS FOR POSITIONING MODES ===
        from PyQt5.QtWidgets import QTabWidget
        tabs = QTabWidget()
        
        # Tab 1: BBox Mode
        bbox_tab = QWidget()
        bbox_layout = QVBoxLayout()
        
        # Position slider + text entry
        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel('Position:'))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.blockSignals(True)
        slider.setValue(int(clip_state['pos']*100))
        slider.blockSignals(False)
        edit_pos = QLineEdit(f"{clip_state['pos']:.2f}")
        edit_pos.setFixedWidth(50)
        pos_row.addWidget(slider)
        pos_row.addWidget(edit_pos)
        bbox_layout.addLayout(pos_row)
        
        # Distance display label (shows current position in cm and total length)

        lbl_distance = QLabel()
        lbl_distance.setStyleSheet("color: gray; font-size: 10pt;")
        bbox_layout.addWidget(lbl_distance)
        
        # Reset button
        btn_bbox_reset = QPushButton('Reset to Center')
        bbox_layout.addWidget(btn_bbox_reset)
        
        bbox_layout.addStretch()
        
        bbox_tab.setLayout(bbox_layout)
        
        # Tab 2: Explicit Origin Mode
        explicit_tab = QWidget()
        explicit_layout = QVBoxLayout()
        
        # Origin point inputs (x, y, z) in centimeters
        origin_row = QHBoxLayout()
        origin_row.addWidget(QLabel('Origin (cm):'))
        edit_ox = QLineEdit(f"{clip_state['origin'][0]*100:.2f}")  # Convert m to cm
        edit_oy = QLineEdit(f"{clip_state['origin'][1]*100:.2f}")
        edit_oz = QLineEdit(f"{clip_state['origin'][2]*100:.2f}")
        edit_ox.setFixedWidth(60)
        edit_oy.setFixedWidth(60)
        edit_oz.setFixedWidth(60)
        origin_row.addWidget(edit_ox)
        origin_row.addWidget(edit_oy)
        origin_row.addWidget(edit_oz)
        explicit_layout.addLayout(origin_row)
        
        # Quick electrode buttons (no Center button here)
        explicit_layout.addWidget(QLabel('Quick Electrodes:'))
        elec_quick_row1 = QHBoxLayout()
        elec_quick_row2 = QHBoxLayout()
        elec_quick_buttons = []
        for i in range(1, 10):
            btn = QPushButton(f'E{i}')
            btn.setFixedWidth(40)
            if i <= 5:
                elec_quick_row1.addWidget(btn)
            else:
                elec_quick_row2.addWidget(btn)
            elec_quick_buttons.append((i, btn))
        
        explicit_layout.addLayout(elec_quick_row1)
        explicit_layout.addLayout(elec_quick_row2)
        explicit_layout.addStretch()
        
        explicit_tab.setLayout(explicit_layout)
        
        # Add tabs to widget
        tabs.addTab(bbox_tab, "BBox Mode")
        tabs.addTab(explicit_tab, "Explicit Origin Mode")
        
        # Set initial tab based on mode
        if clip_state['use_bbox_mode']:
            tabs.setCurrentIndex(0)
        else:
            tabs.setCurrentIndex(1)
        
        v.addWidget(tabs)
        
        # Add a separator before the loading indicator
        sep_loading = QFrame()
        sep_loading.setFrameShape(QFrame.HLine)
        sep_loading.setFrameShadow(QFrame.Sunken)
        v.addWidget(sep_loading)
        
        # Loading indicator at the bottom
        loading_label = QLabel('')
        loading_label.setStyleSheet("color: gray; font-size: 9pt; font-style: italic;")
        loading_label.setWordWrap(True)
        v.addWidget(loading_label)
        
        v.addStretch()  # Push blank space to bottom
        ctrl.setLayout(v)
        
        # Callbacks
        def update_normal_from_inputs():
            try:
                a = float(edit_a.text())
                b = float(edit_b.text())
                c = float(edit_c.text())
                set_clip_normal(a, b, c)
            except Exception:
                pass
        
        edit_a.editingFinished.connect(update_normal_from_inputs)
        edit_b.editingFinished.connect(update_normal_from_inputs)
        edit_c.editingFinished.connect(update_normal_from_inputs)
        
        btn_x.clicked.connect(lambda: set_clip_normal(1, 0, 0))
        btn_y.clicked.connect(lambda: set_clip_normal(0, 1, 0))
        btn_z.clicked.connect(lambda: set_clip_normal(0, 0, 1))
        btn_invert.clicked.connect(invert_clip_normal)
        
        def on_slider(val:int):
            show_loading("Adjusting clip position...")
            clip_state['pos'] = val/100.0
            edit_pos.setText(f"{clip_state['pos']:.2f}")
            update_actors_for_clip()
            hide_loading()
        
        def on_pos_edit():
            try:
                show_loading("Adjusting clip position...")
                p = float(edit_pos.text())
                clip_state['pos'] = max(0.0, min(1.0, p))
                slider.setValue(int(clip_state['pos']*100))
                update_actors_for_clip()
                hide_loading()
            except Exception:
                hide_loading()
        
        slider.valueChanged.connect(on_slider)
        edit_pos.editingFinished.connect(on_pos_edit)
        
        # Origin point callbacks (convert from cm to meters)
        def update_origin_from_inputs():
            try:
                show_loading("Updating clip origin...")
                x_cm = float(edit_ox.text())
                y_cm = float(edit_oy.text())
                z_cm = float(edit_oz.text())
                # Convert cm to meters
                clip_state['origin'] = [x_cm/100.0, y_cm/100.0, z_cm/100.0]
                clip_state['use_bbox_mode'] = False
                print(clip_status())
                update_actors_for_clip()
                if clip_ui_update:
                    clip_ui_update()
                hide_loading()
            except Exception:
                hide_loading()
        
        edit_ox.editingFinished.connect(update_origin_from_inputs)
        edit_oy.editingFinished.connect(update_origin_from_inputs)
        edit_oz.editingFinished.connect(update_origin_from_inputs)
        
        # Electrode quick buttons
        def set_origin_to_electrode(elec_id: int):
            show_loading(f"Setting origin to E{elec_id}...")
            if elec_id in electrode_centers:
                clip_state['origin'] = list(electrode_centers[elec_id])
                clip_state['use_bbox_mode'] = False
                print(f"[clip] Origin set to electrode {elec_id}: {clip_state['origin']}")
                update_actors_for_clip()
                if clip_ui_update:
                    clip_ui_update()
            else:
                print(f"[clip] Electrode {elec_id} not found")
            hide_loading()
        
        for elec_id, btn in elec_quick_buttons:
            btn.clicked.connect(lambda checked, eid=elec_id: set_origin_to_electrode(eid))
        
        # Tab switching callback
        def on_tab_changed(index):
            show_loading("Switching clip mode...")
            if index == 0:
                # Switched to BBox Mode
                clip_state['use_bbox_mode'] = True
                print("[clip] Switched to BBox Mode")
            else:
                # Switched to Explicit Mode
                clip_state['use_bbox_mode'] = False
                print("[clip] Switched to Explicit Origin Mode")
            print(clip_status())
            update_actors_for_clip()
            if clip_ui_update:
                clip_ui_update()
            hide_loading()
        
        tabs.currentChanged.connect(on_tab_changed)
        
        # BBox reset button
        def reset_bbox():
            show_loading("Resetting to center...")
            clip_state['pos'] = 0.5
            print(clip_status())
            update_actors_for_clip()
            if clip_ui_update:
                clip_ui_update()
            hide_loading()
        
        btn_bbox_reset.clicked.connect(reset_bbox)
        
        # Toggle plane button
        def toggle_plane():
            show_loading("Toggling clip plane...")
            clip_state['plane_on'] = not clip_state['plane_on']
            print(clip_status())
            # Enable/disable tabs based on plane state
            tabs.setEnabled(clip_state['plane_on'])
            update_actors_for_clip()
            if clip_ui_update:
                clip_ui_update()
            hide_loading()
        
        # Actor control connections
        head_group.buttonClicked.connect(lambda: change_head_mode())
        gel_group.buttonClicked.connect(lambda: change_gel_mode())
        elec_cb.toggled.connect(lambda: change_elec_visibility())
        probe_cb.toggled.connect(lambda: change_probe_visibility())
        
        # Color bar control connections
        global_range_cb.toggled.connect(lambda: change_global_range())
        static_bar_cb.toggled.connect(lambda: change_static_bar())
        center_zero_cb.toggled.connect(lambda: change_center_zero())
        
        btn_plane_toggle.clicked.connect(toggle_plane)
        
        # Initialize warning visibility based on current state
        update_warning_visibility()

        # UI update function
        def _update_ui():
            try:
                plane_on = clip_state['plane_on']
                
                # Update button labels and states
                btn_plane_toggle.setText('Disable Clipping' if plane_on else 'Enable Clipping')
                btn_plane_toggle.setChecked(plane_on)
                
                # Enable/disable normal vector controls based on plane state
                edit_a.setEnabled(plane_on)
                edit_b.setEnabled(plane_on)
                edit_c.setEnabled(plane_on)
                btn_x.setEnabled(plane_on)
                btn_y.setEnabled(plane_on)
                btn_z.setEnabled(plane_on)
                btn_invert.setEnabled(plane_on)
                
                # Enable/disable tabs based on plane state
                tabs.setEnabled(plane_on)
                
                # Update plane controls
                edit_a.setText(str(clip_state['normal'][0]))
                edit_b.setText(str(clip_state['normal'][1]))
                edit_c.setText(str(clip_state['normal'][2]))
                edit_pos.setText(f"{clip_state['pos']:.2f}")
                slider.setValue(int(clip_state['pos']*100))
                
                # Update distance display in BBox mode (recalculate based on current normal)
                total_length = get_bbox_projection_length(clip_state['normal'])  # Length along current normal
                current_distance = (clip_state['pos'] - 0.5) * total_length  # Offset from center
                current_pos = total_length / 2.0 + current_distance  # Absolute position from start
                total_cm = total_length * 100  # Convert to cm
                current_cm = current_pos * 100  # Convert to cm
                lbl_distance.setText(f"Distance: {current_cm:.1f} cm  (Total: {total_cm:.1f} cm)")
                
                # Update origin controls (display in cm)
                edit_ox.setText(f"{clip_state['origin'][0]*100:.2f}")
                edit_oy.setText(f"{clip_state['origin'][1]*100:.2f}")
                edit_oz.setText(f"{clip_state['origin'][2]*100:.2f}")
                
                # Update tab selection based on mode
                if clip_state['use_bbox_mode']:
                    tabs.setCurrentIndex(0)
                else:
                    tabs.setCurrentIndex(1)
            except Exception:
                pass
        
        clip_ui_update = _update_ui
        _update_ui()  # Initialize UI state

        dock = QDockWidget('Viewer controls', plotter.app_window)
        dock.setWidget(ctrl)
        plotter.app_window.addDockWidget(Qt.LeftDockWidgetArea, dock)
    except Exception as e:
        print(f"[ui] Clip controls not available: {e}")

    plotter.add_key_event("q", lambda: sys.exit(0))
    plotter.add_key_event("Q", lambda: sys.exit(0))
    plotter.add_key_event("Escape", lambda: sys.exit(0))
    print(f"[info] Loaded VTU: {in_path} with {grid.n_points} points and {grid.n_cells} cells.")
    
    # Print bounding box dimensions
    bounds = grid.bounds
    bbox_extents_cm = [(bounds[1]-bounds[0])*100, (bounds[3]-bounds[2])*100, (bounds[5]-bounds[4])*100]
    diagonal_cm = np.sqrt(bbox_extents_cm[0]**2 + bbox_extents_cm[1]**2 + bbox_extents_cm[2]**2)
    print("[bbox] Bounding box dimensions (in cm):")
    print(f"  X extent: {bbox_extents_cm[0]:.2f} cm (width)")
    print(f"  Y extent: {bbox_extents_cm[1]:.2f} cm (height)")
    print(f"  Z extent: {bbox_extents_cm[2]:.2f} cm (depth)")
    print(f"  3D diagonal (corner-to-corner): {diagonal_cm:.2f} cm")
    
    if tri_grid is None or tri_grid.n_cells == 0:
        print("[info] No triangle (electrode) block in VTU — check solver export if you expected electrodes.")
    try:
        from PyQt5.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.exec_()
        else:
            print("[debug] No QApplication instance found; window may close immediately.")
    except Exception as e:
        print(f"[error] Exception in Qt event loop: {e}")


if __name__ == "__main__":
    main()
