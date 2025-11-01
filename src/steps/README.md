# Simulation Workflow Steps

The `steps` folder contains scripts, templates, and documentation for each stage of the simulation workflow. Each step is designed to be run sequentially, with outputs from one step serving as inputs for the next.

## Overview of Steps

### STEP 1: Geometry Preparation (prep_freeze_ids)

- **Objective**: Prepare the simulation geometry by cleaning, scaling, and splitting it into distinct regions with stable identifiers.

### STEP 2: Mesh Generation (label_and_mesh)

- **Objective**: Apply physical labels to the geometry and generate a mesh for the simulation.

### STEP 3: Solver

- **Objective**: Solve the simulation using the generated mesh and configured parameters.

## Additional Tools

- **View Case**: A viewer for inspecting simulation results and visualizing cases.
- **Probe VTU**: A tool for generating results and probing specific data points from the simulation output.

## Output Organization

All outputs are stored in a timestamped run directory (e.g., `run_YYYYMMDD_HHMMSS`). Within each run directory, outputs for individual steps are organized in a `run_steps` subfolder.
