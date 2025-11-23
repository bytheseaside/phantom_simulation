#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solver.py — Head-phantom conduction solver (dolfinx 0.7.x) with single-file VTU export

Model
-----
Solve the steady conduction equation on a two-material head phantom:

    - div( σ ∇u ) = 0  in Ω = Ω_gel ∪ Ω_head

Materials (piecewise constant, isotropic):
- gel  : σ_gel (default 0.33 S/m; override with env SIGMA_GEL)
- head : σ_head (default 0.02 S/m; override with env SIGMA_HEAD)

Boundary conditions
-------------------
- Internal electrode surfaces carved into the gel are Physical Surfaces named
  'v_1'..'v_9'. On each such surface we impose Dirichlet:  u = V_i  (Volts).
  Values come from simulation_cases.csv.
- Gel–head interface: continuity of u and σ∂_n u (automatic with a conformal mesh).
- Outer head surface: natural Neumann (zero normal current), i.e., "do nothing".

Notes
--------------------
- If **no Dirichlet patches are applied** (no electrode facets found): pure Neumann → gauge-free → skip.
- If **all applied Dirichlet values are equal**, this is still physically meaningful
  (Dirichlet islands inside a Neumann domain). We **keep** such cases.

Inputs (in run directory)
-------------------------
- mesh.msh
    Gmsh v4.1 mesh with Physical Volumes: "head", "gel", and Physical Surfaces "v_1".. "v_9".
- simulation_cases.csv
    Header: case,v_1,...,v_9   (Volt values, may be ±; empty cells interpreted as 0.0)

Outputs (per case → ./solutions/)
---------------------------------
- solution_<case>.vtu
    Single VTU with:
      * tetra10 volume cells, point_data['u'], cell_data['region_id']
          region_id: 1 = gel, 2 = head
      * triangle cells for internal electrode surfaces, cell_data['facet_id']
          facet_id: 1..9 for v_1..v_9
- probes_<case>.csv
    If probes.csv exists: sampled potentials at requested points.

Environment overrides
---------------------
  SIGMA_GEL=0.5 SIGMA_HEAD=0.02 python solver.py
"""
from __future__ import annotations
import os
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import meshio
import h5py

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.io import gmshio, XDMFFile
from dolfinx.fem.petsc import LinearProblem
import ufl
from dolfinx import geometry

# --------------------------------------------------------------------------------------
# Environment, paths, and constants
# --------------------------------------------------------------------------------------

COMM = MPI.COMM_WORLD
RANK = COMM.rank

RUN_DIR = Path.cwd()

RUN_DIR_PARENT = RUN_DIR.parent # Some input files are in the parent folder

MSH_PATH = RUN_DIR_PARENT / "mesh.msh"
CASES_CSV = RUN_DIR_PARENT / "simulation_cases.csv"
OUT_DIR = RUN_DIR_PARENT / "cases"
OUT_DIR.mkdir(exist_ok=True)


# Electrode names we accept (strict): v_1..v_9 (case-insensitive in the mesh)
ELECTRODE_NAMES_V = [f"v_{i}" for i in range(1, 10)]

# Default conductivities (S/m) + env overrides
SIGMA_GEL_DEFAULT = 0.33
SIGMA_HEAD_DEFAULT = 4.1 # Effective value for conductive head phantom

def _fenv(name: str, default: float) -> float:
    """
    Retrieve an environment variable as a float.

    Args:
        name (str): The name of the environment variable to retrieve.
        default (float): The default value to return if the environment variable is not set or cannot be converted to float.

    Returns:
        float: The value of the environment variable converted to float, or the default value if not set or conversion fails.
    """
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

SIGMA_GEL = _fenv("SIGMA_GEL", SIGMA_GEL_DEFAULT)
SIGMA_HEAD = _fenv("SIGMA_HEAD", SIGMA_HEAD_DEFAULT)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def read_field_data_ids(msh_path: Path) -> Dict[str, int]:
    """
    Reads the field data from a Gmsh mesh file and returns a dictionary mapping
    physical group names to their corresponding integer IDs.

    Parameters:
        msh_path (Path): The path to the Gmsh mesh file (.msh).

    Returns:
        Dict[str, int]: A dictionary where keys are physical group names (str)
        and values are their associated integer IDs (int) as used for tag lookups
        in dolfinx.
    """
    fd = meshio.read(str(msh_path)).field_data  # {name: [id, dim]}
    return {name: int(v[0]) for name, v in fd.items()}

def load_cases(csv_path: Path) -> List[Tuple[str, Dict[str, float]]]:
    """
    Parse a CSV file containing simulation cases and electrode potentials.

    Args:
        csv_path (Path): Path to the simulation_cases.csv file.
            Expected CSV format:
            ```
                case,v_1,v_2,v_3,v_4,v_5,v_6,v_7,v_8,v_9
                case1,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
                case2,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
                ...
            ```

    Returns:
        List[Tuple[str, Dict[str, float]]]: 
            A list of tuples, each containing:
                - case_name (str): The name of the simulation case.
                - potentials (Dict[str, float]): A dictionary mapping electrode names ('v_1'..'v_9') to their potential values.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the specified path.
        RuntimeError: If the CSV file is empty, malformed, or missing required columns ('v_1'..'v_9').

    Notes:
        - The CSV file must have a header row with columns: 'case', 'v_1', ..., 'v_9'.
        - Electrode potentials can be empty, which are interpreted as 0.0.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing simulation_cases.csv at: {csv_path}")
    rows: List[Tuple[str, Dict[str, float]]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            raise RuntimeError("simulation_cases.csv is empty or missing header.")
        headers = [h.strip() for h in r.fieldnames]
        have_v = all(f"v_{i}" in headers for i in range(1, 10))
        if not have_v:
            raise RuntimeError("CSV must contain columns: case,v_1,...,v_9")
        for row in r:
            case = row["case"].strip()
            pots: Dict[str, float] = {}
            for i in range(1, 10):
                key = f"v_{i}"
                val = float(row[key]) if row[key] != "" else 0.0
                pots[key] = val
            rows.append((case, pots))
    return rows

def build_sigma(mesh, cell_tags, id_head: int | None, sig_gel: float, sig_head: float):
    """
    Constructs a piecewise constant conductivity function `σ` over the given mesh.

    Parameters:
        mesh: The mesh object over which the conductivity function is defined.
        cell_tags: A mesh tag object used to identify specific regions (cells) in the mesh.
        id_head (int | None): The tag identifier for the 'head' region. If None, no override is performed.
        sig_gel (float): The conductivity value to assign to all cells by default.
        sig_head (float): The conductivity value to assign to cells tagged as 'head'.

    Returns:
        sigma: A DG-0 (piecewise constant) function representing the conductivity distribution,
               where `σ = sig_gel` in all cells, except in cells tagged with id_head, where `σ = sig_head`.
    """
    V0 = fem.functionspace(mesh, ("DG", 0))
    sigma = fem.Function(V0)
    sigma.x.array[:] = sig_gel
    if id_head is not None:
        cells = cell_tags.find(id_head)
        if cells.size > 0:
            sigma.x.array[cells] = sig_head
    return sigma

def make_region_id(mesh, cell_tags, id_head: int | None, id_gel: int | None):
    """
    Assigns integer region flags to mesh cells based on provided cell tag IDs.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh on which to define the region IDs.
    cell_tags : dolfinx.mesh.meshtags
        Mesh tags object containing cell tag information.
    id_head : int or None
        The tag ID corresponding to the "head" region. If None, no cells are assigned as head.
    id_gel : int or None
        The tag ID corresponding to the "gel" region. If None, no cells are assigned as gel.

    Returns
    -------
    region : dolfinx.fem.Function
        A DG-0 function over the mesh with integer values:
            1 for gel region cells,
            2 for head region cells,
            0 for all other cells (should not occur; default start value).

    Notes
    -----
    - The function assumes that cell_tags contains the IDs for gel and head regions.
    - The returned function can be used to identify regions in further computations.
    """
    V0 = fem.functionspace(mesh, ("DG", 0))
    region = fem.Function(V0); region.name = "region_id"
    region.x.array[:] = 0.0
    if id_gel is not None:
        gel_cells = cell_tags.find(id_gel)
        if gel_cells.size > 0:
            region.x.array[gel_cells] = 1.0
    if id_head is not None:
        head_cells = cell_tags.find(id_head)
        if head_cells.size > 0:
            region.x.array[head_cells] = 2.0
    return region

def dirichlet_from_electrodes(voltage_space, facet_tags, id_by_name: Dict[str, int], pots: Dict[str, float]):
    """
    Build Dirichlet boundary conditions (BCs) for electrode surfaces.

    For each electrode surface named 'v_1' to 'v_9' in the mesh, this function creates a Dirichlet BC
    that sets the potential u = V_i, where V_i is provided in the 'pots' dictionary. The function
    expects the mesh to have physical groups named 'v_1'...'v_9' (case-insensitive), and will print
    warnings if any are missing or have no associated facets.

    Args:
        voltage_space: The function space (dolfinx.fem.FunctionSpace) for the solution.
        facet_tags: Mesh tags for facets (dolfinx.mesh.meshtags).
        id_by_name: Dictionary mapping physical group names to their integer IDs.
        pots: Dictionary mapping electrode names ('v_1'..'v_9') to their potential values.

    Returns:
        List of Dirichlet BC objects for all found electrodes.

    Notes:
        - Only electrodes with both a valid tag and at least one facet in the mesh will receive a BC.
        - If an electrode is missing or has no facets, a warning is printed (on rank 0).
        - The function expects all electrode names to be in lower case in the mesh.
    """
    bcs = []
    mesh = voltage_space.mesh
    fdim = mesh.topology.dim - 1

    name_to_id = {name.lower(): pid for name, pid in id_by_name.items() if name.lower().startswith("v_")}

    for i in range(1, 10):
        ename = f"v_{i}"
        val = float(pots.get(ename, 0.0))
        tag = name_to_id.get(ename)
        if tag is None:
            if RANK == 0:
                print(f"[warn] Missing Physical tag for electrode {ename}")
            continue
        entities = facet_tags.find(tag)
        if entities.size == 0:
            if RANK == 0:
                print(f"[warn] Electrode {ename} (id={tag}) has no facets")
            continue
        dofs = fem.locate_dofs_topological(voltage_space, fdim, entities)
        u_d = fem.Function(voltage_space)
        u_d.x.array[:] = val
        bcs.append(fem.dirichletbc(u_d, dofs))
    return bcs

def _write_vtu_from_h5(h5_path: Path, vtu_path: Path, electrodes: List[Tuple[np.ndarray, np.ndarray]]):
    """
    Convert the dolfinx H5 to a single VTU with:
      - tetra10 cells and point_data['u'], cell_data['region_id']
      - one triangle cell block (tri3) for internal electrodes with cell_data['facet_id']
    The 'electrodes' argument is a list of (triangles, facet_ids), where:
      triangles: (N, 3) array of vertex indices
      facet_ids: (N,) array of ints in 1..9
    """
    with h5py.File(h5_path, "r") as f:
        points = np.array(f["/Mesh/mesh/geometry"], dtype=float)
        topo   = np.array(f["/Mesh/mesh/topology"], dtype=np.int64)

        def _read_func(name: str):
            grp = f.get(f"/Function/{name}")
            if grp is None:
                return None
            keys = list(grp.keys())
            if not keys:
                return None
            arr = np.array(grp[keys[0]]).reshape(-1)
            return arr

        u = _read_func("u")
        region = _read_func("region_id")

    # Build cell blocks
    cell_blocks = [meshio.CellBlock("tetra10", topo)]
    cell_data: Dict[str, List[np.ndarray]] = {}
    if region is not None:
        cell_data["region_id"] = [region]
    else:
        cell_data["region_id"] = [np.zeros(topo.shape[0], dtype=np.int32)]

    # Merge all electrode triangles into one block (tri3)
    if electrodes:
        tris_all = np.concatenate([t for (t, _) in electrodes], axis=0) if electrodes else np.empty((0, 3), dtype=np.int64)
        ids_all  = np.concatenate([i for (_, i) in electrodes], axis=0) if electrodes else np.empty((0,), dtype=np.int32)
        if tris_all.size > 0:
            cell_blocks.append(meshio.CellBlock("triangle", tris_all))

            # ---- build aligned cell_data lists ----
            # region_id: one array for tetra block + one dummy for triangles
            if "region_id" not in cell_data:
                cell_data["region_id"] = [np.zeros(topo.shape[0], dtype=np.int32)]
            cell_data["region_id"].append(np.zeros(tris_all.shape[0], dtype=np.int32))

            # facet_id: dummy for volume block + real ids for triangles
            cell_data["facet_id"] = [
                np.zeros(topo.shape[0], dtype=np.int32),  # dummy for tets
                ids_all,
            ]

    point_data = {}
    if u is not None:
        point_data["u"] = u

    meshio.write(vtu_path, meshio.Mesh(points, cell_blocks, point_data=point_data, cell_data=cell_data))

def _collect_electrode_triangles(mesh, facet_tags, id_by_name):
    """
    Build triangle connectivity for electrode facets using the cell geometry dofmap
    (works on dolfinx 0.7.x where geometry.dofmap is a NumPy array).
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Required connectivities
    mesh.topology.create_connectivity(fdim, 0)     # facet -> vertices (topology ids)
    mesh.topology.create_connectivity(fdim, tdim)  # facet -> adjacent cells
    mesh.topology.create_connectivity(tdim, 0)     # cell  -> vertices (topology ids)

    f_to_v = mesh.topology.connectivity(fdim, 0)
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    c_to_v = mesh.topology.connectivity(tdim, 0)

    # Map name -> tag id
    name_to_pid = {}
    for name, pid in id_by_name.items():
        ln = name.lower()
        if ln.startswith("v_"):
            try:
                k = int(ln.split("_")[1])
                if 1 <= k <= 9:
                    name_to_pid[f"v_{k}"] = pid
            except Exception:
                pass

    tris = []
    ids = []

    for i in range(1, 10):
        pid = name_to_pid.get(f"v_{i}")
        if pid is None:
            continue
        facets = facet_tags.find(pid)
        for f in facets:
            verts_topo = f_to_v.links(f)              # 3 topology-vertex ids
            if verts_topo.size != 3:
                continue
            cells = f_to_c.links(f)
            if cells.size == 0:
                continue
            c = int(cells[0])

            # (A) cell's 4 vertex topology ids (for a tetra)
            cell_verts_topo = c_to_v.links(c)

            # (B) cell's geometry point ids (indices into the VTU "points" array)
            #     On tetra10, first 4 entries correspond to the 4 vertices.
            cell_geom_points = np.asarray(mesh.geometry.dofmap)[c]  # <-- 0.7.x access
            # Build topo-vertex -> point-index map for this cell
            vmap = {int(tv): int(cell_geom_points[j]) for j, tv in enumerate(cell_verts_topo[:4])}

            try:
                tri = [vmap[int(v)] for v in verts_topo]
            except KeyError:
                continue

            tris.append(np.array(tri, dtype=np.int64))
            ids.append(i)

    if not tris:
        return []
    return [(np.vstack(tris), np.asarray(ids, dtype=np.int32))]

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()

    # Read Gmsh mesh and tags
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(str(MSH_PATH), COMM, 0)

    id_by_name = read_field_data_ids(MSH_PATH)
    if RANK == 0:
        print("Available Physical names:", sorted(id_by_name.keys()))
        print(f"[info] SIGMA_GEL={SIGMA_GEL} (default {SIGMA_GEL_DEFAULT}), "
              f"SIGMA_HEAD={SIGMA_HEAD} (default {SIGMA_HEAD_DEFAULT})")

    id_head = id_by_name.get("head")
    id_gel  = id_by_name.get("gel")

    # Mesh region stats
    if RANK == 0:
        n_head = cell_tags.find(id_head).size if id_head is not None else 0
        n_gel  = cell_tags.find(id_gel).size  if id_gel  is not None else 0
        print(f"[mesh] cells: gel={n_gel}, head={n_head}")

    cases = load_cases(CASES_CSV)
    if RANK == 0:
        print(f"[info] Simulation cases: {len(cases)}")

    for case_name, pots in cases:
        if RANK == 0:
            pretty = ", ".join(f"{k}={pots[k]:.6g}" for k in sorted(pots.keys()))
            print(f"\n[case] {case_name}: {pretty}")

        t_case = time.perf_counter()

        V = fem.functionspace(mesh, ("Lagrange", 2))
        sigma = build_sigma(mesh, cell_tags, id_head, SIGMA_GEL, SIGMA_HEAD)
        region_id = make_region_id(mesh, cell_tags, id_head, id_gel)

        bcs = dirichlet_from_electrodes(V, facet_tags, id_by_name, pots)

        if RANK == 0:
            print("[bc] Applied Dirichlet patches:")
            name_to_id = {name.lower(): pid for name, pid in id_by_name.items() if name.lower().startswith("v_")}
            for i in range(1, 10):
                en = f"v_{i}"
                pid = name_to_id.get(en)
                nfac = facet_tags.find(pid).size if pid is not None else 0
                val = float(pots.get(en, 0.0))
                print(f"  - {en:<3}: {val:.6g} V  ({nfac} facets)")

        # Skip only pure Neumann (no Dirichlet BCs)
        if len(bcs) == 0:
            if RANK == 0:
                print("[skip] No electrode facets found → pure Neumann (no gauge). Skipping case.")
            continue

        # Weak form and solver
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(sigma * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx

        problem = LinearProblem(a, L, bcs=bcs,
                                petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-10})
        uh = problem.solve()
        uh.name = "u"

        # Write XDMF/H5 + convert to VTU
        xdmf_path = OUT_DIR / f"solution_{case_name}.xdmf"
        if RANK == 0:
            print(f"[write] {xdmf_path.name} (+ .h5)")
        with XDMFFile(COMM, str(xdmf_path), "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(uh)
            xdmf.write_function(region_id)

        if RANK == 0:
            h5_path = xdmf_path.with_suffix(".h5")
            vtu_path = xdmf_path.with_suffix(".vtu")
            electrodes = _collect_electrode_triangles(mesh, facet_tags, id_by_name)
            print(f"[write] {vtu_path.name} (tetra10 + {'electrodes' if electrodes else 'no electrodes'})")
            _write_vtu_from_h5(h5_path, vtu_path, electrodes)

        if RANK == 0:
            print(f"[time] {case_name}: {time.perf_counter() - t_case:.2f} s")

    if RANK == 0:
        print(f"\n[time] total: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    os.environ.setdefault("DOLFINX_JIT_CACHE_DIR", str(RUN_DIR / "__dolfinx_cache__"))
    os.environ.setdefault("FFCX_CACHE_DIR",        str(RUN_DIR / "__ffcx_cache__"))
    os.environ.setdefault("XDG_CACHE_HOME",        str(RUN_DIR / "__cache__"))
    os.environ.setdefault("FFCX_REGENERATE",       "1")
    Path(os.environ["DOLFINX_JIT_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["FFCX_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    main()
