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
from manifest_schema import load_manifest, mesh_info_from_field_data

import numpy as np
import meshio
import h5py
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem
from dolfinx.io import gmshio, XDMFFile
from dolfinx.fem.petsc import LinearProblem
import ufl

# --------------------------------------------------------------------------------------
# Exit codes
# --------------------------------------------------------------------------------------

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
# Mesh utilities
# --------------------------------------------------------------------------------------
def build_sigma_function(mesh, cell_tags, volumes):
    V0 = fem.functionspace(mesh, ("DG", 0))
    sigma = fem.Function(V0)
    sigma.name = "sigma"
    sigma.x.array[:] = 0.0

    for v in volumes:
        cells = cell_tags.find(v.id)
        if cells.size > 0:
            sigma.x.array[cells] = v.sigma
    return sigma

def build_dirichlet_bcs(
    voltage_space, facet_tags, dirichlets
) -> List[fem.DirichletBC]:
    """
    Build Dirichlet boundary conditions from resolved Dirichlet objects.

    Parameters:
        voltage_space: Function space for the solution.
        facet_tags: Mesh tags for facets (surfaces).
        dirichlets: List of Dirichlet objects (each has .id, .name, .value)

    Returns:
        List of Dirichlet BC objects.
    """
    bcs = []
    mesh = voltage_space.mesh
    fdim = mesh.topology.dim - 1

    for i, item in enumerate(dirichlets):
        facets = facet_tags.find(item.id)
        # --- Locate DOFs ---
        dofs = fem.locate_dofs_topological(voltage_space, fdim, facets)
        # --- Create BC function ---
        u_bc = fem.Function(voltage_space)
        u_bc.x.array[:] = item.value

        bcs.append(fem.dirichletbc(u_bc, dofs))
    return bcs

# --------------------------------------------------------------------------------------
# XDMF output
# --------------------------------------------------------------------------------------
def write_xdmf_output(
    *,
    mesh,
    cell_tags,
    facet_tags,
    u: fem.Function,
    output_path: Path,
) -> None:
    """
    Write solution and mesh data to XDMF format.
    
    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh
    cell_tags : dolfinx.mesh.MeshTags
        Physical volume tags (region IDs)
    facet_tags : dolfinx.mesh.MeshTags
        Physical surface tags (including Dirichlet surfaces)
    u : fem.Function
        Solution function
    output_path : Path
        Path to output XDMF file (e.g., "case_name.xdmf")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with XDMFFile(MPI.COMM_SELF, str(output_path), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)
        xdmf.write_meshtags(cell_tags, mesh.geometry)
        xdmf.write_meshtags(facet_tags, mesh.geometry)


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
    
    print("=" * 80)
    print("FEM Electrostatics Solver")
    print("=" * 80)
    print(f"Manifest: {manifest_path}")
    print(f"Mode: {'VALIDATE ONLY' if validate_only else 'SOLVE'}")
    print()
    
    # Load and validate manifest
    manifest = load_manifest(manifest_path)

    print(f"Loaded manifest with {len(manifest.cases)} case(s)")
    print()
    
    # Resolve paths relative to manifest directory
    manifest_dir = manifest_path.parent
    mesh_path = (manifest_dir / manifest.mesh).resolve()
        
    # Check mesh file exists
    if not mesh_path.exists():
        print(f"ERROR: Mesh file not found: {mesh_path}")
        sys.exit(EXIT_VALIDATION_ERROR)


    output_dir = (manifest_dir / manifest.output).resolve()
    
    print(f"Mesh path: {mesh_path}")
    print(f"Output directory: {output_dir}")
    print()
        
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
                
    # VALIDATION:
    print("Validating mesh against manifest...")
    mesh_data = meshio.read(str(mesh_path)) 

    report = manifest.validate_against_mesh(mesh_data.field_data)

    for w in report.warnings:
        print("⚠️", w)

    report.raise_if_errors()

    print("✅ Validation OK\n")

    if validate_only:
        print("Validation complete. Exiting (--validate-only mode).")
        sys.exit(EXIT_SUCCESS)

    manifest.resolve_with_mesh(mesh_data.field_data)
    # Read mesh
    print("Reading mesh...")
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(str(mesh_path), MPI.COMM_SELF, 0)

    print("Building conductivity distribution...")
    
    sigma = build_sigma_function(mesh, cell_tags, manifest.volumes)
    
    print("Conductivity values:")
    for vol in manifest.volumes:
        print(f"  - Volume {vol.name} (ID: {vol.id}): σ = {vol.sigma}") 
    print()
    
    print("Mesh contains:")
    print(f"  - {len(set(cell_tags.values))} physical volume(s)")
    print(f"  - {len(set(facet_tags.values))} physical surface(s)")
    print()
    
    # Create function space (shared across cases)
    V = fem.functionspace(mesh, ("Lagrange", 2))
    print(f"Function space: Lagrange P2, {V.dofmap.index_map.size_global} DOFs")
    print()
    
    # Solve each case
    for case_idx, case in enumerate(manifest.cases, 1): 
        print("=" * 80)
        print(f"Case {case_idx}/{len(manifest.cases)}: {case.name}")
        print("=" * 80)
        
        t_case = time.perf_counter()
        
        # Build Dirichlet BCs for the case
        bcs = build_dirichlet_bcs(V, facet_tags, case.dirichlet)
        
        print(f"Dirichlet boundary conditions ({len(case.dirichlet)}):")
        for bc in case.dirichlet: 
            n_facets = facet_tags.find(bc.id).size
            print(f"  - Surface '{bc.name}' (ID: {bc.id}): u = {bc.value} V ({n_facets} facets)")
        print()
                
        # Define weak form
        print("Assembling and solving linear system...")
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(sigma * ufl.grad(u), ufl.grad(v)) * ufl.dx
        L = fem.Constant(mesh, PETSc.ScalarType(0.0)) * v * ufl.dx
        
        # Solve
        try:
            problem = LinearProblem(
                a, L, bcs=bcs,
                petsc_options={"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-12}
            )
            uh = problem.solve()
            uh.name = "u"
        except Exception as e:
            print(f"ERROR: Solver failed: {e}")
            sys.exit(EXIT_SOLVER_ERROR)
        
        print("Solve complete.")
        print()
        
        # Write XDMF output
        output_file = output_dir / f"{case.name}.xdmf"
        print(f"Writing output: {output_file.name}")
        write_xdmf_output(
            mesh=mesh,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            u=uh,
            output_path=output_file,
        )
        print(f"Case time: {time.perf_counter() - t_case:.2f} s")
        print()

    print("=" * 80)
    print(f"All cases complete. Total time: {time.perf_counter() - t_start:.2f} s")
    print("=" * 80)
    
    sys.exit(EXIT_SUCCESS)


if __name__ == "__main__":
    main()
