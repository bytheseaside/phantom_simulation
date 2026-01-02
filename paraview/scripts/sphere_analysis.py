import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for ParaView
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import MaxNLocator, MultipleLocator, AutoMinorLocator
from paraview.simple import FindSource, PlotOverLine, Delete
from vtk.util.numpy_support import vtk_to_numpy
from paraview import servermanager
import os
import csv

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Plot settings
BIN_COUNT = 10              # Number of bins for histogram
POL_RESOLUTION = 1000        # Points in plots

# Save settings
SAVE_PATH_BASE = '/Users/brisarojas/Desktop/phantom_simulation/paraview/plots/'  # Output directory
SAVE_DPI = 150
SAVE_FORMAT = 'svg'  # 'svg', 'png', etc.

# Plot style
FIGSIZE = (14, 7)
GRID_ALPHA = 0.3

# Color scheme
COLOR_ANALYTICAL = '#6A8EDB'   # Purple-ish blue
COLOR_NUMERICAL = '#F77F00'    # Orange
COLOR_ERROR = '#2A9D8F'        # Teal/cyan-green
COLOR_VOLUME1 = '#66C2A5'      # Soft green (inner sphere)
COLOR_VOLUME2 = '#8DA0CB'      # Soft blue-purple (outer sphere)
COLOR_REFLINES = '0.2'         # Gray for reference lines (r0, r1, r2)
REFLINE_WIDTH = 0.7              # Width of reference lines
REFLINE_STYLE = '--'           # Dashed style for reference lines

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def _get_case_name_from_source(base_source_name):
    src = FindSource(base_source_name)
    if src is None:
        raise RuntimeError(f'Base source "{base_source_name}" not found.')

    proxy = src.SMProxy  # vtkSMSourceProxy

    p = proxy.GetProperty("FileName")
    if p is not None and p.GetNumberOfElements() > 0:
        fn = p.GetElement(0)
        return os.path.splitext(os.path.basename(fn))[0]

    p = proxy.GetProperty("FileNames")
    if p is not None and p.GetNumberOfElements() > 0:
        fn = p.GetElement(0)
        return os.path.splitext(os.path.basename(fn))[0]

    raise RuntimeError(f'Source "{base_source_name}" has no FileName/FileNames property.')

def _fetch_params_table(params_name="PARAM INJECTOR"):
    psrc = FindSource(params_name)
    if psrc is None:
        raise RuntimeError(f'Params source "{params_name}" not found.')
    return servermanager.Fetch(psrc)

def _get_param(tbl, name: str) -> float:
    col = tbl.GetColumnByName(name)
    if col is None:
        raise RuntimeError(f'Param "{name}" not found in params table.')
    return float(vtk_to_numpy(col)[0])

def _get_vlines_in_range(r_min_mm, r_max_mm, r0_mm, r1_mm, r2_mm, ax):
    ymin, ymax = ax.get_ylim()
    y_text = (ymax + ymin) * 0.5 
    for x in (r0_mm, r1_mm, r2_mm):
        if x >= r_min_mm and x <= r_max_mm:
            ax.axvline(x, linestyle=REFLINE_STYLE, linewidth=REFLINE_WIDTH, color=COLOR_REFLINES)
            ax.text(x, y_text, f"r₀ " if x == r0_mm else (f"r₁ " if x == r1_mm else "r₂ "), ha="right", va="top", fontsize=10, color="0.6")

def _get_x_range_for_mask(xlim_mm, r0, r2):
    if xlim_mm is not None:
        r_min = xlim_mm[0] / 1000.0
        r_max = xlim_mm[1] / 1000.0
    else:
        r_min = r0
        r_max = r2
    return r_min, r_max
# ----------------------------
# Loading stage
# ----------------------------
def load_paraview_data(params_name="PARAM INJECTOR"):
    # --- params ---
    pt = _fetch_params_table(params_name=params_name)
    params = {
        "r0": _get_param(pt, "r0"),
        "r1": _get_param(pt, "r1"),
        "r2": _get_param(pt, "r2"),
        "sigma1": _get_param(pt, "sigma1"),
        "sigma2": _get_param(pt, "sigma2"),
        "V0": _get_param(pt, "V0"),
        "V2": _get_param(pt, "V2"),
    }

    # --- dataset arrays (numerical, from active source) ---
    ds_src = FindSource("FINAL")
    if ds_src is None:
        raise RuntimeError('Source "FINAL" not found. Rename your last pipeline node to "FINAL".')

    # sanity prints
    print("Loaded params:", params)

    return params, ds_src

def setup_plot_style():
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams['axes.titlepad'] = 30

def style_axis_offset_text(ax, color="0.5", fontsize=10):
    ax.xaxis.get_offset_text().set_color(color)
    ax.xaxis.get_offset_text().set_fontsize(fontsize)
    ax.yaxis.get_offset_text().set_color(color)
    ax.yaxis.get_offset_text().set_fontsize(fontsize)

def finish_axes(ax,  r_min_mm, r_max_mm, r0_mm, r1_mm, r2_mm, fig=None):
    """Apply standard finishing touches to all plots: xlim, autoscaling, grid, ticks, legend."""
    # Standard tick locators
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    style_axis_offset_text(ax)
            
    # Grid
    ax.grid(True, alpha=GRID_ALPHA)
    
    # Legend with automatic column count based on number of items
    handles, labels = ax.get_legend_handles_labels()
    ncol = len(handles) if len(handles) > 0 else 1
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=True, ncol=ncol)
    
    _get_vlines_in_range(r_min_mm, r_max_mm, r0_mm, r1_mm, r2_mm, ax=ax)

    # Adjust figure layout to make room for legend below
    if fig is not None:
        fig.subplots_adjust(bottom=0.18)

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def plot_u_vs_r(save_dir, params, coeffs, src_proxy, show_error=True, xlim_mm=None):

    r0, r1, r2 = params["r0"], params["r1"], params["r2"]
    A1, B1, A2, B2 = coeffs["A1"], coeffs["B1"], coeffs["A2"], coeffs["B2"]

    # --- sample numerical u along x-axis ---
    pol = PlotOverLine(Input=src_proxy)
    pol.Point1 = [0.0, 0.0, 0.0]
    pol.Point2 = [r2, 0.0, 0.0]
    pol.Resolution = POL_RESOLUTION
    pol.UpdatePipeline()

    line_ds = servermanager.Fetch(pol)

    arc = vtk_to_numpy(line_ds.GetPointData().GetArray("arc_length"))
    u_num = vtk_to_numpy(line_ds.GetPointData().GetArray("u"))

    # arc_length == radius because Point1 is origin; crop to [r0, r2]
    r_num = arc
    r_min, r_max = _get_x_range_for_mask(xlim_mm, r0, r2)
    mask_volume = (r_num >= r_min) & (r_num <= r_max)
    r_num = r_num[mask_volume]
    u_num = u_num[mask_volume]

    # --- analytic continuous curve on a dense grid ---
    r_grid = np.linspace(r_min, r_max, 2000)
    u_ana = np.empty_like(r_grid)
    mask_v1 = r_grid <= r1
    u_ana[mask_v1] = A1 / r_grid[mask_v1] + B1
    u_ana[~mask_v1] = A2 / r_grid[~mask_v1] + B2

    # --- convert x-axis to mm for plotting ---
    r_num_mm  = 1000.0 * r_num
    r_grid_mm = 1000.0 * r_grid
    r1_mm     = 1000.0 * r1
    r0_mm = 1000.0 * r0
    r2_mm = 1000.0 * r2

    # --- plot ---
    os.makedirs(save_dir, exist_ok=True)
    # Modify the save path to include xlim if provided
    if xlim_mm is not None:
        xlim_str = f"_xlim_{xlim_mm[0]}_{xlim_mm[1]}"
        out_path = os.path.join(save_dir, f"u_vs_r{xlim_str}.{SAVE_FORMAT}")
    else:
        out_path = os.path.join(save_dir, f"u_vs_r.{SAVE_FORMAT}")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(r_grid_mm, u_ana, color=COLOR_ANALYTICAL, label="Analytical")
    ax.plot(r_num_mm, u_num, linestyle="None", marker="o", markersize=1, color=COLOR_NUMERICAL, label="Numerical")

    if show_error:
        # also plot error as shaded area
        err_abs = vtk_to_numpy(line_ds.GetPointData().GetArray("err_abs"))
        err_abs = err_abs[mask_volume]
        ax.plot(
            r_num_mm, err_abs,
            linestyle=":", linewidth=0.8, marker="o", markersize=1,
            color=COLOR_ERROR,
            label="Absolute error"
        )

    ax.set_xlabel("r [mm]")
    ax.set_ylabel("u [V]")
    # Modify the title of the plot to include xlim if provided
    if xlim_mm is not None:
        ax.set_title(f"Potential - u(r) [{xlim_mm[0]} < r < {xlim_mm[1]} mm]")
    else:
        ax.set_title("Potential - u(r)")
    
    finish_axes(ax, r_min_mm=r_min * 1000.0, r_max_mm=r_max * 1000.0, r0_mm=r0_mm, r1_mm=r1_mm, r2_mm=r2_mm, fig=fig)
    
    fig.savefig(out_path, dpi=SAVE_DPI, format=SAVE_FORMAT)
    plt.close(fig)

    Delete(pol)
    del pol

def plot_u_errors(save_dir, params, src_proxy, xlim_mm=None):
    r0, r1, r2 = params["r0"], params["r1"], params["r2"]

    # --- sample along x-axis ---
    pol = PlotOverLine(Input=src_proxy)
    pol.Point1 = [0.0, 0.0, 0.0]
    pol.Point2 = [r2, 0.0, 0.0]
    pol.Resolution = POL_RESOLUTION
    pol.UpdatePipeline()

    line_ds = servermanager.Fetch(pol)

    arc = vtk_to_numpy(line_ds.GetPointData().GetArray("arc_length"))
    r_num = arc

    r_min, r_max = _get_x_range_for_mask(xlim_mm, r0, r2)
    mask_volume = (r_num >= r_min) & (r_num <= r_max)

    r_mm = 1000.0 * r_num[mask_volume]
    r0_mm, r1_mm, r2_mm = 1000.0*r0, 1000.0*r1, 1000.0*r2

    # read errors
    err_abs = vtk_to_numpy(line_ds.GetPointData().GetArray("err_abs"))[mask_volume]
    err_rel = vtk_to_numpy(line_ds.GetPointData().GetArray("err_rel"))[mask_volume] * 100.0  # %

    os.makedirs(save_dir, exist_ok=True)

    # ----------------- ABS ERROR PLOT -----------------
    if xlim_mm is not None:
        xlim_str = f"_xlim_{r_min * 1000.0}_{r_max * 1000.0}"
        out_abs = os.path.join(save_dir, f"err_abs_vs_r{xlim_str}.{SAVE_FORMAT}")
    else:
        out_abs = os.path.join(save_dir, f"err_abs_vs_r.{SAVE_FORMAT}")
    
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(r_mm, err_abs, linestyle=":", linewidth=0.8, marker="o", markersize=1, color=COLOR_ERROR, label="Absolute error")


    ax.set_xlabel("r [mm]")
    ax.set_ylabel("|error| [V]")
    if xlim_mm is not None:
        ax.set_title(f"u(r) absolute error [{xlim_mm[0]} < r < {xlim_mm[1]} mm]")
    else:
        ax.set_title("u(r) absolute error")
    
    finish_axes(ax, r_min_mm=r_min * 1000.0, r_max_mm=r_max * 1000.0, r0_mm=r0_mm, r1_mm=r1_mm, r2_mm=r2_mm, fig=fig)
        
    fig.savefig(out_abs, dpi=SAVE_DPI, format=SAVE_FORMAT)
    plt.close(fig)

    # ----------------- REL ERROR PLOT -----------------
    if xlim_mm is not None:
        xlim_str = f"_xlim_{xlim_mm[0]}_{xlim_mm[1]}"
        out_rel = os.path.join(save_dir, f"err_rel_vs_r{xlim_str}.{SAVE_FORMAT}")
    else:
        out_rel = os.path.join(save_dir, f"err_rel_vs_r.{SAVE_FORMAT}")
    
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(r_mm, err_rel, linestyle=":", linewidth=0.8, marker="o", markersize=1, color=COLOR_ERROR, label="Relative error")


    ax.set_xlabel("r [mm]")
    ax.set_ylabel("relative error [%]")

    if xlim_mm is not None:
        ax.set_title(f"Relative error of potential u(r) [{xlim_mm[0]} < r < {xlim_mm[1]} mm]")
    else:
        ax.set_title("Relative error of potential u(r)")
    
    finish_axes(ax, r_min_mm=r_min * 1000.0, r_max_mm=r_max * 1000.0, r0_mm=r0_mm, r1_mm=r1_mm, r2_mm=r2_mm, fig=fig)
    fig.savefig(out_rel, dpi=SAVE_DPI, format=SAVE_FORMAT)
    plt.close(fig)

    Delete(pol)
    del pol

def plot_err_abs_histogram(
    save_dir,
    src_proxy,
    params,
    bins=50,
    xlim_mm=None,
):
    os.makedirs(save_dir, exist_ok=True)

    ds = servermanager.Fetch(src_proxy)

    # --- arrays ---
    err = vtk_to_numpy(ds.GetPointData().GetArray("err_abs"))
    if err is None:
        raise RuntimeError('Point array "err_abs" not found on src_proxy.')

    pts = vtk_to_numpy(ds.GetPoints().GetData())  # (N,3)
    r = np.linalg.norm(pts, axis=1)

    r0 = params["r0"]
    r1 = params["r1"]
    r2 = params["r2"]
    
    # Convert to mm
    r_mm = r * 1000.0
    r0_mm = r0 * 1000.0
    r1_mm = r1 * 1000.0
    r2_mm = r2 * 1000.0

    # --- Determine sampling range ---

    r_min, r_max = _get_x_range_for_mask(xlim_mm, r0, r2)
    if xlim_mm is not None:
        # Use xlim to filter data
        mask_all = (r_mm >= r_min * 1000.0) & (r_mm <= r_max * 1000.0)
        
        # Determine which volumes are present in the xlim range
        has_v1 = r_min * 1000.0 < r1_mm  # xlim overlaps with v1
        has_v2 = r_max * 1000.0 > r1_mm  # xlim overlaps with v2
        
        if has_v1 and has_v2:
            # Both volumes present
            mask_v1 = (r_mm >= r_min * 1000.0) & (r_mm <= r1_mm)
            mask_v2 = (r_mm > r1_mm) & (r_mm <= r_max * 1000.0)
            title_suffix = f"({r_min*1000:.1f} ≤ r ≤ {r_max*1000:.1f} mm)"
            labels = ["Inner sphere (r ≤ r₁)", "Outer sphere (r > r₁)"]
        elif has_v1:
            # Only v1 present
            mask_v1 = mask_all
            mask_v2 = np.zeros_like(mask_all, dtype=bool)
            title_suffix = f"({r_min*1000:.1f} ≤ r ≤ {r_max*1000:.1f} mm, inner sphere)"
            labels = ["Inner sphere"]
        else:
            # Only v2 present
            mask_v1 = np.zeros_like(mask_all, dtype=bool)
            mask_v2 = mask_all
            title_suffix = f"({r_min*1000:.1f} ≤ r ≤ {r_max*1000:.1f} mm, outer sphere)"
            labels = ["Outer sphere"]
        
        fname_suffix = f"_xlim_{r_min*1000}_{r_max*1000}"
    else:
        # No xlim: use full range
        mask_all = (r_mm >= r0_mm) & (r_mm <= r2_mm)
        mask_v1 = (r_mm >= r0_mm) & (r_mm < r1_mm)
        mask_v2 = (r_mm >= r1_mm) & (r_mm <= r2_mm)
        title_suffix = "(r₀ ≤ r ≤ r₂)"
        fname_suffix = ""
        has_v1 = True
        has_v2 = True
        labels = ["Inner sphere (r ≤ r₁)", "Outer sphere (r < r₁)"]

    err_all = err[mask_all]
    err_v1 = err[mask_v1]
    err_v2 = err[mask_v2]

    N = int(len(err_all))
    if N == 0:
        raise RuntimeError("No points available to create histogram.")

    # --- bins: 0 .. max(error) over selected set ---
    emax = float(np.max(err_all))
    if emax <= 0.0:
        # All errors are zero -> still create something meaningful
        emax = 1.0e-16

    edges = np.linspace(0.0, emax, int(bins) + 1)

    # --- weights so Y is fraction of total points in selected set ---
    w_v1 = np.ones_like(err_v1) / N if len(err_v1) > 0 else np.array([])
    w_v2 = np.ones_like(err_v2) / N if len(err_v2) > 0 else np.array([])

    out_path = os.path.join(save_dir, f"hist_err_bins{bins}{fname_suffix}.{SAVE_FORMAT}")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    min_err = float(np.min(err_all))
    mean_err = float(np.mean(err_all))
    max_err  = float(np.max(err_all))
    p25, p50, p75, p99 = np.percentile(err_all, [25, 50, 75, 99])
    base, _ = os.path.splitext(out_path)
    stats_path = base + "_stats.csv"

    # Prepare data for histogram based on which volumes are present
    hist_data = []
    hist_weights = []
    hist_labels = []
    
    if has_v1 and len(err_v1) > 0:
        hist_data.append(err_v1)
        hist_weights.append(w_v1)
        hist_labels.append(labels[0])
    
    if has_v2 and len(err_v2) > 0:
        hist_data.append(err_v2)
        hist_weights.append(w_v2)
        hist_labels.append(labels[-1])

    # Stacked histogram by volume (fraction)
    ax.hist(
        hist_data,
        bins=edges,
        stacked=True,
        weights=hist_weights,
        alpha=0.9,
        label=hist_labels,
        color=[COLOR_VOLUME1, COLOR_VOLUME2][:len(hist_data)],
        edgecolor="black"
    )

    for x in (p50, p99):
        ax.axvline(x, linestyle=REFLINE_STYLE, linewidth=REFLINE_WIDTH, color=COLOR_REFLINES, alpha=0.6)
        ax.text(x, ax.get_ylim()[1]*0.9, f"p{50 if x == p50 else 99}", rotation=90, va="top", ha="right", fontsize=8, color="0.5")
    ax.set_title(f"Absolute error distribution - N={N} {title_suffix} ")
    ax.set_xlabel("|error| [V]")
    ax.set_ylabel("point fraction")
    
    # Histogram uses more y-axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=20))

    # Minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Grid: major + minor
    ax.grid(True, which="major", alpha=GRID_ALPHA)
    ax.grid(True, which="minor", alpha=0.3 * GRID_ALPHA)

    # Legend below the plot with automatic column count
    handles, labels = ax.get_legend_handles_labels()
    ncol = len(handles) if len(handles) > 0 else 1
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=True, ncol=ncol)
    fig.subplots_adjust(bottom=0.18)

    # Scientific notation
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    style_axis_offset_text(ax, color="0.5", fontsize=10)

    fig.savefig(out_path, dpi=SAVE_DPI, format=SAVE_FORMAT )
    plt.close(fig)
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "units"])
        w.writerow(["N_points_plotted", N, "count"])
        w.writerow(["min_err_abs",  min_err, "V"])
        w.writerow(["mean_err_abs", mean_err, "V"])
        w.writerow(["max_err_abs",  max_err, "V"])
        w.writerow(["p25_err_abs",  (p25), "V"])
        w.writerow(["p50_err_abs",  (p50), "V"])
        w.writerow(["p75_err_abs",  (p75), "V"])
        w.writerow(["p99_err_abs",  (p99), "V"])
        w.writerow(["N_points_r_lt_r1", int(len(err_v1)), "count"])
        w.writerow(["N_points_r_ge_r1", int(len(err_v2)), "count"])

def plot_du_dr_vs_r(save_dir, params, coeffs, src_proxy, xlim_mm=None):

    r0, r1, r2 = params["r0"], params["r1"], params["r2"]
    A1, A2 = coeffs["A1"], coeffs["A2"]

    # --- sample numerical u along x-axis ---
    pol = PlotOverLine(Input=src_proxy)
    pol.Point1 = [0.0, 0.0, 0.0]
    pol.Point2 = [r2, 0.0, 0.0]
    pol.Resolution = POL_RESOLUTION
    pol.UpdatePipeline()

    line_ds = servermanager.Fetch(pol)

    arc = vtk_to_numpy(line_ds.GetPointData().GetArray("arc_length"))
    du_dr_num = vtk_to_numpy(line_ds.GetPointData().GetArray("du_dr_num"))

    # arc_length == radius because Point1 is origin; crop to (r0, r2)
    r_num = arc

    r_min, r_max = _get_x_range_for_mask(xlim_mm, r0, r2)
    mask_volume = (r_num > r_min) & (r_num < r_max)
    r_num = r_num[mask_volume]
    du_dr_num = du_dr_num[mask_volume]

    du_dr_ana_on_num_points = np.empty_like(r_num)
    mask_v1_num = r_num <= r1
    du_dr_ana_on_num_points[mask_v1_num] = -A1 / (r_num[mask_v1_num]**2)
    du_dr_ana_on_num_points[~mask_v1_num] = -A2 / (r_num[~mask_v1_num]**2)

    err_abs_du_dr = np.abs(du_dr_num - du_dr_ana_on_num_points)


    # --- analytic continuous curve on a dense grid ---
    r_grid = np.linspace(r_min, r_max, 2000)
    du_dr_ana = np.empty_like(r_grid)
    mask_v1 = r_grid <= r1
    du_dr_ana[mask_v1] = - A1 / r_grid[mask_v1]**2
    du_dr_ana[~mask_v1] = - A2 / r_grid[~mask_v1]**2

    # --- convert x-axis to mm for plotting ---
    r_num_mm  = 1000.0 * r_num
    r_grid_mm = 1000.0 * r_grid
    r1_mm     = 1000.0 * r1
    r0_mm = 1000.0 * r0
    r2_mm = 1000.0 * r2

    # --- plot the du/dr ---
    os.makedirs(save_dir, exist_ok=True)
    # Modify the save path to include xlim if provided
    if xlim_mm is not None:
        xlim_str = f"_xlim_{xlim_mm[0]}_{xlim_mm[1]}"
        out_path = os.path.join(save_dir, f"du_dr_vs_r{xlim_str}.{SAVE_FORMAT}")
    else:
        out_path = os.path.join(save_dir, f"du_dr_vs_r.{SAVE_FORMAT}")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(r_grid_mm, du_dr_ana, color=COLOR_ANALYTICAL, label="Analytical")
    ax.plot(r_num_mm, du_dr_num, linestyle="None", marker="o", markersize=1, color=COLOR_NUMERICAL, label="Numerical")


    ax.set_xlabel("r [mm]")
    ax.set_ylabel("du/dr [V/m]")
    if xlim_mm is not None:
        ax.set_title(f"Potential derivative, du/dr [{xlim_mm[0]} < r < {xlim_mm[1]} mm]")
    else:
        ax.set_title("Potential derivative, du/dr")
    
    finish_axes(ax, r_min_mm=r_min * 1000.0, r_max_mm=r_max * 1000.0, r0_mm=r0_mm, r1_mm=r1_mm, r2_mm=r2_mm, fig=fig)
    
    fig.savefig(out_path, dpi=SAVE_DPI, format=SAVE_FORMAT )
    plt.close(fig)

    # --- plot the du/dr error ---
    if xlim_mm is not None:
        xlim_str = f"_xlim_{xlim_mm[0]}_{xlim_mm[1]}"
        out_path_err = os.path.join(save_dir, f"du_dr_err_abs_vs_r{xlim_str}.{SAVE_FORMAT}")
    else:
        out_path_err = os.path.join(save_dir, f"du_dr_err_abs_vs_r.{SAVE_FORMAT}")

    fig2, ax2 = plt.subplots(figsize=FIGSIZE)

    ax2.plot(
        r_num_mm, err_abs_du_dr,
        linestyle=":", linewidth=0.8,
        marker="o", markersize=1,
        color=COLOR_ERROR,
        label="Absolute error"
    )

    ax2.set_xlabel("r [mm]")
    ax2.set_ylabel("|error| [V/m]")  
    if xlim_mm is not None:
        ax2.set_title(f"du/dr error [{xlim_mm[0]} < r < {xlim_mm[1]} mm]")
    else:
        ax2.set_title("du/dr error")
    finish_axes(ax2, r_min_mm=r_min * 1000.0, r_max_mm=r_max * 1000.0, r0_mm=r0_mm, r1_mm=r1_mm, r2_mm=r2_mm, fig=fig2)
    fig2.savefig(out_path_err, dpi=SAVE_DPI, format=SAVE_FORMAT )
    plt.close(fig2)


    Delete(pol)
    del pol

def plot_flux_vs_r(save_dir, params, coeffs, src_proxy, xlim_mm=None):
    r0, r1, r2 = params["r0"], params["r1"], params["r2"]
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    A1, A2 = coeffs["A1"], coeffs["A2"]

    # --- sample numerical flux along x-axis ---
    pol = PlotOverLine(Input=src_proxy)
    pol.Point1 = [0.0, 0.0, 0.0]
    pol.Point2 = [r2, 0.0, 0.0]
    pol.Resolution = POL_RESOLUTION
    pol.UpdatePipeline()

    line_ds = servermanager.Fetch(pol)

    arc = vtk_to_numpy(line_ds.GetPointData().GetArray("arc_length"))
    flux_num = vtk_to_numpy(line_ds.GetPointData().GetArray("flux_num"))

    # crop to (r0, r2)
    r_num = arc
    r_min , r_max = _get_x_range_for_mask(xlim_mm, r0, r2)
    mask_volume = (r_num > r_min) & (r_num < r_max)
    r_num = r_num[mask_volume]
    flux_num = flux_num[mask_volume]


    # --- analytic continuous curve on dense grid (for smooth line) ---
    r_grid = np.linspace(r_min, r_max, 2000)
    flux_ana = np.empty_like(r_grid)
    mask_v1 = r_grid <= r1
    flux_ana[mask_v1]  = sigma1 * A1 / (r_grid[mask_v1]**2)
    flux_ana[~mask_v1] = sigma2 * A2 / (r_grid[~mask_v1]**2)

    # --- convert x-axis to mm for plotting ---
    r_num_mm  = 1000.0 * r_num
    r_grid_mm = 1000.0 * r_grid
    r0_mm, r1_mm, r2_mm = 1000.0 * r0, 1000.0 * r1, 1000.0 * r2

    # --- plot ---
    os.makedirs(save_dir, exist_ok=True)
    # Modify the save path to include xlim if provided
    if xlim_mm is not None:
        xlim_str = f"_xlim_{xlim_mm[0]}_{xlim_mm[1]}"
        out_path = os.path.join(save_dir, f"flux_vs_r{xlim_str}.{SAVE_FORMAT}")
    else:
        out_path = os.path.join(save_dir, f"flux_vs_r.{SAVE_FORMAT}")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(r_grid_mm, flux_ana, color=COLOR_ANALYTICAL, label="Analytical")
    ax.plot(r_num_mm, flux_num, linestyle="None", marker="o", markersize=1, color=COLOR_NUMERICAL, label="Numerical")

    ax.set_xlabel("r [mm]")
    ax.set_ylabel("J_r [A/m²]")
    if xlim_mm is not None:
        ax.set_title(f"Radial current density, J_r [{xlim_mm[0]} < r < {xlim_mm[1]} mm]")
    else:
        ax.set_title("Radial current density, J_r")
    
    finish_axes(ax, r_min_mm=r_min * 1000.0, r_max_mm=r_max * 1000.0, r0_mm=r0_mm, r1_mm=r1_mm, r2_mm=r2_mm, fig=fig)
    fig.savefig(out_path, dpi=SAVE_DPI, format=SAVE_FORMAT )
    plt.close(fig)

    Delete(pol)
    del pol

def plot_d2u_dr2_vs_r(save_dir, params, coeffs, src_proxy, xlim_mm=None):
    r0, r1, r2 = params["r0"], params["r1"], params["r2"]
    A1, A2 = coeffs["A1"], coeffs["A2"]

    # --- sample numerical along x-axis ---
    pol = PlotOverLine(Input=src_proxy)
    pol.Point1 = [0.0, 0.0, 0.0]
    pol.Point2 = [r2, 0.0, 0.0]
    pol.Resolution = POL_RESOLUTION
    pol.UpdatePipeline()

    line_ds = servermanager.Fetch(pol)

    arc = vtk_to_numpy(line_ds.GetPointData().GetArray("arc_length"))
    d2u_dr2_num = vtk_to_numpy(line_ds.GetPointData().GetArray("d2u_dr2_num"))

    # strict crop to (r0, r2)
    r_num = arc
    r_min, r_max = _get_x_range_for_mask(xlim_mm, r0, r2)
    mask = (r_num > r_min) & (r_num < r_max)
    r_num = r_num[mask]
    d2u_dr2_num = d2u_dr2_num[mask]

    # --- analytic on dense grid (piecewise) ---
    r_grid = np.linspace(r_min, r_max, 2000)
    d2u_dr2_ana = np.empty_like(r_grid)
    mask_v1 = r_grid <= r1
    # u=A/r+B => du/dr=-A/r^2 => d2u/dr2 = 2A/r^3
    d2u_dr2_ana[mask_v1] = 2.0 * A1 / (r_grid[mask_v1] ** 3)
    d2u_dr2_ana[~mask_v1] = 2.0 * A2 / (r_grid[~mask_v1] ** 3)

    # --- convert x-axis to mm for plotting only ---
    r_num_mm  = 1000.0 * r_num
    r_grid_mm = 1000.0 * r_grid
    r0_mm, r1_mm, r2_mm = 1000.0*r0, 1000.0*r1, 1000.0*r2

    # --- plot ---
    os.makedirs(save_dir, exist_ok=True)
    # Modify the save path to include xlim if provided
    if xlim_mm is not None:
        xlim_str = f"_xlim_{xlim_mm[0]}_{xlim_mm[1]}"
        out_path = os.path.join(save_dir, f"d2u_dr2_vs_r{xlim_str}.{SAVE_FORMAT}")
    else:
        out_path = os.path.join(save_dir, f"d2u_dr2_vs_r.{SAVE_FORMAT}")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(r_grid_mm, d2u_dr2_ana, color=COLOR_ANALYTICAL, label="Analytical")
    ax.plot(r_num_mm, d2u_dr2_num, linestyle="None", marker="o", markersize=1, color=COLOR_NUMERICAL, label="Numerical")

    ax.set_xlabel("r [mm]")
    ax.set_ylabel("d²u/dr² [V/m²]")  # x is mm only; field is still in SI
    if xlim_mm is not None:
        ax.set_title(f"Second derivative, d²u/dr² [{xlim_mm[0]} < r < {xlim_mm[1]} mm]")
    else:
        ax.set_title("Second derivative, d²u/dr²") 

    finish_axes(ax, r_min_mm=r_min * 1000.0, r_max_mm=r_max * 1000.0, r0_mm=r0_mm, r1_mm=r1_mm, r2_mm=r2_mm, fig=fig)
    
    
    fig.savefig(out_path, dpi=SAVE_DPI, format=SAVE_FORMAT)
    plt.close(fig)

    Delete(pol)
    del pol

def plot_laplacian_vs_r(save_dir, params, src_proxy, xlim_mm=None):
    r0, r1, r2 = params["r0"], params["r1"], params["r2"]

    # --- sample numerical along x-axis ---
    pol = PlotOverLine(Input=src_proxy)
    pol.Point1 = [0.0, 0.0, 0.0]
    pol.Point2 = [r2, 0.0, 0.0]
    pol.Resolution = POL_RESOLUTION
    pol.UpdatePipeline()

    line_ds = servermanager.Fetch(pol)

    arc = vtk_to_numpy(line_ds.GetPointData().GetArray("arc_length"))
    lap_num = vtk_to_numpy(line_ds.GetPointData().GetArray("laplacian_num"))

    # strict crop to (r0, r2)
    r_num = arc
    r_min , r_max = _get_x_range_for_mask(xlim_mm, r0, r2)
    mask = (r_num > r_min) & (r_num < r_max)
    r_num = r_num[mask]
    lap_num = lap_num[mask]

    # --- x-axis in mm ---
    r_num_mm = 1000.0 * r_num
    r0_mm, r1_mm, r2_mm = 1000.0*r0, 1000.0*r1, 1000.0*r2

    # --- plot ---
    os.makedirs(save_dir, exist_ok=True)
    # Modify the save path to include xlim if provided
    if xlim_mm is not None:
        xlim_str = f"_xlim_{xlim_mm[0]}_{xlim_mm[1]}"
        out_path = os.path.join(save_dir, f"laplacian_vs_r{xlim_str}.{SAVE_FORMAT}")
    else:
        out_path = os.path.join(save_dir, f"laplacian_vs_r.{SAVE_FORMAT}")

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Analytical solution: Laplacian should be zero everywhere
    ax.axhline(0, linestyle="-", linewidth=1.5, color=COLOR_ANALYTICAL, label="Analytical", zorder=1)
    
    ax.plot(r_num_mm, lap_num, linestyle="-", marker="o", markersize=1, color=COLOR_NUMERICAL, label="Numerical", zorder=2)

    ax.set_xlabel("r [mm]")
    ax.set_ylabel("∇²u [V/m²]")  # assuming your PV expression yields V/m²
    if xlim_mm is not None:
        ax.set_title(f"Laplacian, ∇²u [{xlim_mm[0]} < r < {xlim_mm[1]} mm]")
    else:
        ax.set_title("Laplacian, ∇²u")

    finish_axes(ax, r_min_mm=r_min * 1000.0, r_max_mm=r_max * 1000.0, r0_mm=r0_mm, r1_mm=r1_mm, r2_mm=r2_mm, fig=fig)
    
    fig.savefig(out_path, dpi=SAVE_DPI, format=SAVE_FORMAT )
    plt.close(fig)

    Delete(pol)
    del pol

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Setup
    setup_plot_style()
    
    # Get data
    print("Extracting data from ParaView...")
    try:
        params, ds_src = load_paraview_data()
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Save directory
    save_dir = SAVE_PATH_BASE + _get_case_name_from_source("BASE") + "/"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Output directory: {save_dir}")

    # Analytical solution coefficients
    r0, r1, r2 = params["r0"], params["r1"], params["r2"]
    sigma1, sigma2 = params["sigma1"], params["sigma2"]
    V0, V2 = params["V0"], params["V2"]
    denominator = r0 * r1 * sigma1 - r1 * r2 * sigma2 + r0 * r2 * (sigma2 - sigma1)
    B1 = (- r1 * r2 * V2 * sigma2  + r0 * V0 * ((r1 * sigma1 )+ r2 * (sigma2 - sigma1))) / denominator
    A1 =   r0 * r1 * r2 * sigma2 * (V2 - V0) / denominator

    B2 = (r0 * r1 * V0 * sigma1 - r1 * r2 * V2 * sigma2 + r0 * r2 * V2 * (sigma2 - sigma1)) / denominator
    A2 = r0 * r1 * r2 * sigma1 * (V2 - V0) / denominator

    coeffs = {"A1": A1, "B1": B1, "A2": A2, "B2": B2}
    params = {"r0": r0, "r1": r1, "r2": r2, "sigma1": sigma1, "sigma2": sigma2, "V0": V0, "V2": V2}
    
    # Convert to mm for xlim
    r1_mm = r1 * 1000.0
    r2_mm = r2 * 1000.0
    
    # Define regions of interest
    lateral_pad_mm = 1.0  # padding around interfaces
    xlim_outer_region = (r1_mm - lateral_pad_mm, r2_mm + lateral_pad_mm )  # r1-r2 region with left padding
    xlim_r1_interface = (r1_mm - lateral_pad_mm, r1_mm + lateral_pad_mm)  # Around r1 interface
    
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    # ========================================================================
    # 1. POTENTIAL (u) PLOTS
    # ========================================================================
    print("\n1. Potential plots...")
    
    # Full volume
    plot_u_vs_r(
        save_dir=save_dir,
        params=params,
        coeffs=coeffs,
        src_proxy=ds_src,
        show_error=True,
    )
    
    # Outer region (r1-r2 with padding)
    plot_u_vs_r(
        save_dir=save_dir,
        params=params,
        coeffs=coeffs,
        src_proxy=ds_src,
        show_error=False,
        xlim_mm=xlim_outer_region
    )
    
    # ========================================================================
    # 2. ERROR PLOTS
    # ========================================================================
    print("2. Error plots...")
    
    # Full volume errors
    plot_u_errors(
        save_dir=save_dir,
        params=params,
        src_proxy=ds_src
    )
    
    # Outer region errors (r1-r2 with padding)
    plot_u_errors(
        save_dir=save_dir,
        params=params,
        src_proxy=ds_src,
        xlim_mm=xlim_outer_region
    )
    
    # ========================================================================
    # 3. ERROR HISTOGRAMS
    # ========================================================================
    print("3. Error histograms...")
    
    # Full volume histogram
    plot_err_abs_histogram(
        save_dir=save_dir,
        src_proxy=ds_src,
        params=params,
        bins=15,
    )
    
    # Outer region histogram (r1-r2 with padding)
    plot_err_abs_histogram(
        save_dir=save_dir,
        src_proxy=ds_src,
        params=params,
        bins=15,
        xlim_mm=xlim_outer_region
    )
    
    # ========================================================================
    # 4. DERIVATIVE PLOTS (du/dr) - Check continuity at r1
    # ========================================================================
    print("4. Derivative plots (du/dr)...")
    
    # Full volume
    plot_du_dr_vs_r(
        save_dir=save_dir,
        params=params,
        coeffs=coeffs,
        src_proxy=ds_src,
    )
    
    # Around r1 interface (to check continuity)
    plot_du_dr_vs_r(
        save_dir=save_dir,
        params=params,
        coeffs=coeffs,
        src_proxy=ds_src,
        xlim_mm=xlim_r1_interface
    )
    
    # ========================================================================
    # 5. FLUX PLOTS (J_r) - Check continuity at r1
    # ========================================================================
    print("5. Flux plots (J_r)...")
    
    # Full volume
    plot_flux_vs_r(
        save_dir=save_dir,
        params=params,
        coeffs=coeffs,
        src_proxy=ds_src,
    )
    
    # Around r1 interface (to check continuity)
    plot_flux_vs_r(
        save_dir=save_dir,
        params=params,
        coeffs=coeffs,
        src_proxy=ds_src,
        xlim_mm=xlim_r1_interface
    )
    
    # ========================================================================
    # 6. SECOND DERIVATIVE PLOTS (d²u/dr²)
    # ========================================================================
    print("6. Second derivative plots (d²u/dr²)...")
    
    # Full volume
    plot_d2u_dr2_vs_r(
        save_dir=save_dir,
        params=params,
        coeffs=coeffs,
        src_proxy=ds_src,
    )
    
    # Around r1 interface
    plot_d2u_dr2_vs_r(
        save_dir=save_dir,
        params=params,
        coeffs=coeffs,
        src_proxy=ds_src,
        xlim_mm=xlim_r1_interface
    )
    
    # ========================================================================
    # 7. LAPLACIAN PLOTS
    # ========================================================================
    print("7. Laplacian plots...")
    
    # Full volume
    plot_laplacian_vs_r(
        save_dir=save_dir,
        params=params,
        src_proxy=ds_src,
    )
    
try:
    main()
except Exception as e:
    print("❌ Error:", e)
