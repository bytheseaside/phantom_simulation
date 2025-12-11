#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import meshio

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
import ufl

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

COMM = MPI.COMM_WORLD
RANK = COMM.rank

# Exit codes
EXIT_SUCCESS = 0
EXIT_VALIDATION_ERROR = 1
EXIT_SOLVER_ERROR = 2

# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FEM solver for electrostatics driven by manifest configuration."
    )
    parser.add_argument(
        "manifest",
        type=str,
        help="Path to the manifest JSON file"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifest and mesh compatibility without solving"
    )
    return parser.parse_args()

# --------------------------------------------------------------------------------------
# Manifest handling
# --------------------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load and parse the manifest JSON file."""
    if not manifest_path.exists():
        if RANK == 0:
            print(f"ERROR: Manifest file not found: {manifest_path}")
        sys.exit(EXIT_VALIDATION_ERROR)
    
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        if RANK == 0:
            print(f"ERROR: Invalid JSON in manifest: {e}")
        sys.exit(EXIT_VALIDATION_ERROR)
    
    return manifest

def validate_manifest_schema(manifest: Dict[str, Any]) -> None:
    """Validate that the manifest has the required structure."""
    required_keys = ["mesh", "outputs", "volumes", "cases"]
    missing = [k for k in required_keys if k not in manifest]
    if missing:
        if RANK == 0:
            print(f"ERROR: Manifest missing required keys: {missing}")
        sys.exit(EXIT_VALIDATION_ERROR)
    
    # Validate volumes structure
    if not isinstance(manifest["volumes"], dict):
        if RANK == 0:
            print("ERROR: 'volumes' must be a dictionary")
        sys.exit(EXIT_VALIDATION_ERROR)
    
    for vol_id, vol_config in manifest["volumes"].items():
        try:
            int(vol_id)
        except ValueError:
            if RANK == 0:
                print(f"ERROR: Volume ID '{vol_id}' must be an integer")
            sys.exit(EXIT_VALIDATION_ERROR)
        
        if "sigma" not in vol_config:
            if RANK == 0:
                vol_name = vol_config.get("name", vol_id)
                print(f"ERROR: Volume '{vol_name}' (ID: {vol_id}) missing 'sigma'")
            sys.exit(EXIT_VALIDATION_ERROR)
    
    # Validate cases structure
    if not isinstance(manifest["cases"], list) or len(manifest["cases"]) == 0:
        if RANK == 0:
            print("ERROR: 'cases' must be a non-empty list")
        sys.exit(EXIT_VALIDATION_ERROR)
    
    for i, case in enumerate(manifest["cases"]):
        if "case_name" not in case:
            if RANK == 0:
                print(f"ERROR: Case {i} missing 'case_name'")
            sys.exit(EXIT_VALIDATION_ERROR)
        
        if "dirichlet" not in case:
            if RANK == 0:
                print(f"ERROR: Case '{case['case_name']}' missing 'dirichlet'")
            sys.exit(EXIT_VALIDATION_ERROR)
        
        if not isinstance(case["dirichlet"], list):
            if RANK == 0:
                print(f"ERROR: Case '{case['case_name']}' 'dirichlet' must be a list")
            sys.exit(EXIT_VALIDATION_ERROR)
        
        # Check that each case has at least one Dirichlet BC
        if len(case["dirichlet"]) == 0:
            if RANK == 0:
                print(f"ERROR: Case '{case['case_name']}' has no Dirichlet conditions.")
                print("       At least one Dirichlet BC is required to define the problem.")
            sys.exit(EXIT_VALIDATION_ERROR)
        
        # Validate Dirichlet BC structure
        for j, bc in enumerate(case["dirichlet"]):
            if "target" not in bc or "value" not in bc:
                if RANK == 0:
                    print(f"ERROR: Case '{case['case_name']}' BC {j} missing 'target' or 'value'")
                sys.exit(EXIT_VALIDATION_ERROR)

# --------------------------------------------------------------------------------------
# Mesh utilities
# --------------------------------------------------------------------------------------

def get_physical_ids_from_mesh(msh_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Read physical group information from a Gmsh mesh file.
    
    Returns:
        Tuple of (name_to_id, id_to_name) dictionaries.
    """
    mesh_data = meshio.read(str(msh_path))
    field_data = mesh_data.field_data  # {name: [id, dim]}
    
    name_to_id = {name: int(v[0]) for name, v in field_data.items()}
    id_to_name = {int(v[0]): name for name, v in field_data.items()}
    
    return name_to_id, id_to_name

def validate_mesh_against_manifest(
    mesh, cell_tags, facet_tags, manifest: Dict[str, Any], 
    id_to_name: Dict[int, str]
) -> None:
    """
    Validate that the mesh and manifest are compatible.
    
    Checks:
    - All mesh volumes have sigma defined in manifest
    - Warns about manifest volumes not in mesh
    - Warns about manifest surfaces not in mesh
    - Logs physical surfaces without Dirichlet BCs
    """
    # Get all physical volume IDs from mesh
    mesh_volume_ids = set(cell_tags.values)
    
    # Get all physical volume IDs from manifest
    manifest_volume_ids = {int(vid) for vid in manifest["volumes"].keys()}
    
    # Check: volumes in mesh but not in manifest (ERROR)
    missing_in_manifest = mesh_volume_ids - manifest_volume_ids
    if missing_in_manifest:
        if RANK == 0:
            print("ERROR: The following physical volumes exist in the mesh but have no conductivity defined:")
            for vid in sorted(missing_in_manifest):
                vname = id_to_name.get(vid, "<unnamed>")
                print(f"  - Volume '{vname}' (ID: {vid})")
            print("All volumes must have a conductivity (sigma) defined in the manifest.")
        sys.exit(EXIT_VALIDATION_ERROR)
    
    # Check: volumes in manifest but not in mesh (WARN)
    extra_in_manifest = manifest_volume_ids - mesh_volume_ids
    if extra_in_manifest and RANK == 0:
        print("WARNING: The following volumes are defined in manifest but not found in mesh:")
        for vid in sorted(extra_in_manifest):
            vol_config = manifest["volumes"][str(vid)]
            vname = vol_config.get("name", "<unnamed>")
            print(f"  - Volume '{vname}' (ID: {vid})")
    
    # Get all physical surface IDs from mesh
    mesh_surface_ids = set(facet_tags.values)
    
    # Collect all surfaces mentioned in Dirichlet BCs
    dirichlet_surface_ids = set()
    for case in manifest["cases"]:
        for bc in case["dirichlet"]:
            dirichlet_surface_ids.add(int(bc["target"]))
    
    # Check: Dirichlet surfaces in manifest but not in mesh (WARN)
    missing_surfaces = dirichlet_surface_ids - mesh_surface_ids
    if missing_surfaces and RANK == 0:
        print("WARNING: The following surfaces are referenced in Dirichlet BCs but not found in mesh:")
        for sid in sorted(missing_surfaces):
            sname = id_to_name.get(sid, "<unnamed>")
            print(f"  - Surface '{sname}' (ID: {sid})")
    
    # Info: surfaces in mesh without Dirichlet BCs
    unused_surfaces = mesh_surface_ids - dirichlet_surface_ids
    if unused_surfaces and RANK == 0:
        print(f"INFO: {len(unused_surfaces)} physical surface(s) in mesh have no Dirichlet BC (natural Neumann):")
        for sid in sorted(unused_surfaces):
            sname = id_to_name.get(sid, "<unnamed>")
            print(f"  - Surface '{sname}' (ID: {sid})")

def build_sigma_from_manifest(mesh, cell_tags, volumes_config: Dict[str, Dict[str, Any]]):
    """
    Build piecewise constant conductivity function from manifest volumes configuration.
    
    Parameters:
        mesh: The mesh object.
        cell_tags: Mesh tags for cells (volumes).
        volumes_config: Dictionary mapping volume ID (as string) to config dict with 'sigma'.
    
    Returns:
        sigma: DG-0 function with conductivity values assigned per volume.
    """
    V0 = fem.functionspace(mesh, ("DG", 0))
    sigma = fem.Function(V0)
    sigma.name = "sigma"
    
    # Initialize to zero (will be overwritten, but good practice)
    sigma.x.array[:] = 0.0
    
    # Assign conductivity for each volume
    for vol_id_str, vol_config in volumes_config.items():
        vol_id = int(vol_id_str)
        sig_value = float(vol_config["sigma"])
        
        cells = cell_tags.find(vol_id)
        if cells.size > 0:
            sigma.x.array[cells] = sig_value
    
    return sigma

def build_region_id_from_manifest(mesh, cell_tags, volumes_config: Dict[str, Dict[str, Any]]):
    """
    Build region_id function using physical volume IDs directly.
    
    Parameters:
        mesh: The mesh object.
        cell_tags: Mesh tags for cells (volumes).
        volumes_config: Dictionary mapping volume ID to config dict.
    
    Returns:
        region: DG-0 function where region_id[cell] = physical_volume_id.
    """
    V0 = fem.functionspace(mesh, ("DG", 0))
    region = fem.Function(V0)
    region.name = "region_id"
    region.x.array[:] = 0
    
    # Assign physical volume ID to each cell
    for vol_id_str in volumes_config.keys():
        vol_id = int(vol_id_str)
        cells = cell_tags.find(vol_id)
        if cells.size > 0:
            region.x.array[cells] = vol_id
    
    return region

def build_dirichlet_bcs_from_manifest(
    voltage_space, facet_tags, dirichlet_list: List[Dict[str, Any]]
):
    """
    Build Dirichlet boundary conditions from manifest configuration.
    
    Parameters:
        voltage_space: Function space for the solution.
        facet_tags: Mesh tags for facets (surfaces).
        dirichlet_list: List of dicts with 'target' (surface ID) and 'value' (potential).
    
    Returns:
        List of Dirichlet BC objects.
    """
    bcs = []
    mesh = voltage_space.mesh
    fdim = mesh.topology.dim - 1
    
    for bc_config in dirichlet_list:
        surf_id = int(bc_config["target"])
        value = float(bc_config["value"])
        
        # Find facets with this physical ID
        facets = facet_tags.find(surf_id)
        
        if facets.size == 0:
            # Already warned in validation, skip
            continue
        
        # Locate DOFs on these facets
        dofs = fem.locate_dofs_topological(voltage_space, fdim, facets)
        
        # Create constant function with the boundary value
        u_bc = fem.Function(voltage_space)
        u_bc.x.array[:] = value
        
        bcs.append(fem.dirichletbc(u_bc, dofs))
    
    return bcs

# --------------------------------------------------------------------------------------
# VTU output
# --------------------------------------------------------------------------------------

def extract_dirichlet_surface_triangles(
    mesh, facet_tags, dirichlet_list: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract triangle connectivity for surfaces with Dirichlet BCs.
    
    Returns:
        Tuple of (triangles, surface_ids, dirichlet_values) where:
        - triangles: (N, 3) array of vertex indices
        - surface_ids: (N,) array of physical surface IDs
        - dirichlet_values: (N,) array of applied potential values
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    
    # Create required connectivity
    mesh.topology.create_connectivity(fdim, 0)     # facet -> vertices
    mesh.topology.create_connectivity(fdim, tdim)  # facet -> cells
    mesh.topology.create_connectivity(tdim, 0)     # cell -> vertices
    
    f_to_v = mesh.topology.connectivity(fdim, 0)
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    c_to_v = mesh.topology.connectivity(tdim, 0)
    
    all_tris = []
    all_surf_ids = []
    all_values = []
    
    for bc_config in dirichlet_list:
        surf_id = int(bc_config["target"])
        value = float(bc_config["value"])
        
        facets = facet_tags.find(surf_id)
        
        for f in facets:
            verts_topo = f_to_v.links(f)
            if verts_topo.size != 3:
                continue  # Skip non-triangular facets
            
            # Find adjacent cell
            cells = f_to_c.links(f)
            if cells.size == 0:
                continue
            c = int(cells[0])
            
            # Get cell vertices (topology IDs)
            cell_verts_topo = c_to_v.links(c)
            
            # Get geometry point IDs for this cell
            cell_geom_points = np.asarray(mesh.geometry.dofmap)[c]
            
            # Build mapping: topology vertex ID -> geometry point index
            vmap = {int(tv): int(cell_geom_points[j]) 
                    for j, tv in enumerate(cell_verts_topo[:4])}
            
            try:
                tri = [vmap[int(v)] for v in verts_topo]
            except KeyError:
                continue
            
            all_tris.append(tri)
            all_surf_ids.append(surf_id)
            all_values.append(value)
    
    if not all_tris:
        return (np.empty((0, 3), dtype=np.int64), 
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=float))
    
    return (np.array(all_tris, dtype=np.int64),
            np.array(all_surf_ids, dtype=np.int32),
            np.array(all_values, dtype=float))

def write_vtu(
    mesh, solution, region_id, 
    dirichlet_triangles: Tuple[np.ndarray, np.ndarray, np.ndarray],
    output_path: Path
) -> None:
    """
    Write solution to VTU format with volume cells and Dirichlet surface triangles.
    
    Parameters:
        mesh: The mesh object.
        solution: The solution function (u).
        region_id: The region_id function.
        dirichlet_triangles: Tuple of (triangles, surface_ids, values) from extract function.
        output_path: Path to output VTU file.
    """
    # Get mesh geometry
    points = mesh.geometry.x
    
    # Get cell topology (assuming tetrahedra)
    cell_type = mesh.topology.cell_name()
    if cell_type == "tetrahedron":
        # For P2 (quadratic) tetrahedral elements, we have 10 nodes per cell
        topology = np.asarray(mesh.geometry.dofmap)
        meshio_cell_type = "tetra10" if topology.shape[1] == 10 else "tetra"
    else:
        # Fallback for other cell types
        topology = np.asarray(mesh.geometry.dofmap)
        meshio_cell_type = cell_type
    
    # Build cell blocks
    cell_blocks = [meshio.CellBlock(meshio_cell_type, topology)]
    
    # Cell data (region_id for volume cells)
    cell_data = {
        "region_id": [region_id.x.array.astype(np.int32)]
    }
    
    # Add Dirichlet surface triangles if any
    tris, surf_ids, values = dirichlet_triangles
    if tris.shape[0] > 0:
        cell_blocks.append(meshio.CellBlock("triangle", tris))
        
        # Align cell_data arrays
        cell_data["region_id"].append(np.zeros(tris.shape[0], dtype=np.int32))
        cell_data["surface_id"] = [
            np.zeros(topology.shape[0], dtype=np.int32),  # dummy for volumes
            surf_ids
        ]
        cell_data["dirichlet_value"] = [
            np.zeros(topology.shape[0], dtype=float),  # dummy for volumes
            values
        ]
    
    # Point data (solution values)
    point_data = {
        "u": solution.x.array
    }
    
    # Write VTU
    meshio.write(
        output_path,
        meshio.Mesh(points, cell_blocks, point_data=point_data, cell_data=cell_data)
    )

# --------------------------------------------------------------------------------------
# Main solver routine
# --------------------------------------------------------------------------------------

def main():
    """Main solver routine driven by manifest configuration."""
    t_start = time.perf_counter()
    
    # Parse arguments
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    validate_only = args.validate_only
    
    if RANK == 0:
        print("=" * 80)
        print("FEM Electrostatics Solver")
        print("=" * 80)
        print(f"Manifest: {manifest_path}")
        print(f"Mode: {'VALIDATE ONLY' if validate_only else 'SOLVE'}")
        print()
    
    # Load and validate manifest
    manifest = load_manifest(manifest_path)
    validate_manifest_schema(manifest)
    
    if RANK == 0:
        print(f"Loaded manifest with {len(manifest['cases'])} case(s)")
        print()
    
    # Resolve paths relative to manifest directory
    manifest_dir = manifest_path.parent
    mesh_path = (manifest_dir / manifest["mesh"]).resolve()
    
    # Handle outputs - support both string and dict formats
    if isinstance(manifest["outputs"], dict):
        output_dir = (manifest_dir / manifest["outputs"]["out_dir"]).resolve()
    else:
        output_dir = (manifest_dir / manifest["outputs"]).resolve()
    
    # Setup cache directories EARLY (before any FEM operations)
    cache_base = output_dir / ".cache"
    if RANK == 0:
        cache_base.mkdir(parents=True, exist_ok=True)
    os.environ["DOLFINX_JIT_CACHE_DIR"] = str(cache_base / "dolfinx")
    os.environ["FFCX_CACHE_DIR"] = str(cache_base / "ffcx")
    os.environ["XDG_CACHE_HOME"] = str(cache_base / "xdg")
    Path(os.environ["DOLFINX_JIT_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["FFCX_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    
    if RANK == 0:
        print(f"Mesh path: {mesh_path}")
        print(f"Output directory: {output_dir}")
        print()
    
    # Check mesh file exists
    if not mesh_path.exists():
        if RANK == 0:
            print(f"ERROR: Mesh file not found: {mesh_path}")
        sys.exit(EXIT_VALIDATION_ERROR)
    
    # Create output directory
    if RANK == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read mesh
    if RANK == 0:
        print("Reading mesh...")
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(str(mesh_path), COMM, 0)
    
    # Get physical ID mappings
    name_to_id, id_to_name = get_physical_ids_from_mesh(mesh_path)
    
    if RANK == 0:
        print(f"Mesh loaded: {mesh.topology.index_map(mesh.topology.dim).size_global} cells")
        print(f"Physical groups: {len(name_to_id)}")
        print()
    
    # Validate mesh against manifest
    if RANK == 0:
        print("Validating mesh against manifest...")
        print("-" * 80)
    validate_mesh_against_manifest(mesh, cell_tags, facet_tags, manifest, id_to_name)
    if RANK == 0:
        print("-" * 80)
        print()
    
    # If validate-only mode, exit here
    if validate_only:
        if RANK == 0:
            print("Validation complete. Exiting (--validate-only mode).")
            print(f"Total time: {time.perf_counter() - t_start:.2f} s")
        sys.exit(EXIT_SUCCESS)
    
    # Build conductivity and region_id functions (shared across cases)
    if RANK == 0:
        print("Building conductivity distribution...")
    sigma = build_sigma_from_manifest(mesh, cell_tags, manifest["volumes"])
    region_id = build_region_id_from_manifest(mesh, cell_tags, manifest["volumes"])
    
    if RANK == 0:
        print("Conductivity values:")
        for vol_id_str, vol_config in manifest["volumes"].items():
            vol_name = vol_config.get("name", "<unnamed>")
            sigma_val = vol_config["sigma"]
            print(f"  - Volume '{vol_name}' (ID: {vol_id_str}): σ = {sigma_val}")
        print()
    
    # Create function space (shared across cases)
    V = fem.functionspace(mesh, ("Lagrange", 2))
    if RANK == 0:
        print(f"Function space: Lagrange P2, {V.dofmap.index_map.size_global} DOFs")
        print()
    
    # Solve each case
    for case_idx, case_config in enumerate(manifest["cases"], 1):
        case_name = case_config["case_name"]
        dirichlet_list = case_config["dirichlet"]
        
        if RANK == 0:
            print("=" * 80)
            print(f"Case {case_idx}/{len(manifest['cases'])}: {case_name}")
            print("=" * 80)
        
        t_case = time.perf_counter()
        
        # Build Dirichlet BCs
        bcs = build_dirichlet_bcs_from_manifest(V, facet_tags, dirichlet_list)
        
        if RANK == 0:
            print(f"Dirichlet boundary conditions ({len(dirichlet_list)}):")
            for bc_config in dirichlet_list:
                surf_id = bc_config["target"]
                value = bc_config["value"]
                surf_name = id_to_name.get(surf_id, "<unnamed>")
                n_facets = facet_tags.find(surf_id).size
                print(f"  - Surface '{surf_name}' (ID: {surf_id}): u = {value} V ({n_facets} facets)")
            print()
        
        # Check that we have at least one BC (already checked in validation, but double-check)
        if len(bcs) == 0:
            if RANK == 0:
                print("ERROR: No valid Dirichlet BCs applied (all surfaces missing from mesh).")
            sys.exit(EXIT_VALIDATION_ERROR)
        
        # Define weak form
        if RANK == 0:
            print("Assembling and solving linear system...")
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(sigma * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx
        
        # Solve
        try:
            problem = LinearProblem(
                a, L, bcs=bcs,
                petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-10}
            )
            uh = problem.solve()
            uh.name = "u"
        except Exception as e:
            if RANK == 0:
                print(f"ERROR: Solver failed: {e}")
            sys.exit(EXIT_SOLVER_ERROR)
        
        if RANK == 0:
            print("Solve complete.")
            print()
        
        # Extract Dirichlet surface triangles for visualization
        dirichlet_surfaces = extract_dirichlet_surface_triangles(mesh, facet_tags, dirichlet_list)
        
        # Write VTU output (only on rank 0)
        if RANK == 0:
            output_file = output_dir / f"{case_name}.vtu"
            print(f"Writing output: {output_file.name}")
            write_vtu(mesh, uh, region_id, dirichlet_surfaces, output_file)
            print(f"Case time: {time.perf_counter() - t_case:.2f} s")
            print()
    
    if RANK == 0:
        print("=" * 80)
        print(f"All cases complete. Total time: {time.perf_counter() - t_start:.2f} s")
        print("=" * 80)
    
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
