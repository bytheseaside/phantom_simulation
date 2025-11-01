# STEP 1: Geometry Preparation (prep_freeze_ids)

STEP 1 prepares the simulation geometry for downstream meshing and analysis by cleaning, scaling, and splitting it into distinct regions with stable identifiers.

## Overview

STEP 1 begins with the original geometry file (`simulation_assembly.step`) located in the parent directory. The geometry is processed using the Gmsh script (`01_prep_freeze_ids.geo`), which performs operations such as scaling from millimeters to meters to ensure compatibility with simulation tools like DolfinX and FEniCS.

The process is executed in a new run directory (e.g., `run_YYYYMMDD_HHMMSS`), created automatically by the shell script. All outputs are organized in the `run_steps` subfolder inside this run directory.

**Execution:**

```bash
bash src/steps/step_1/run_step_01.sh
```

**Outputs:**

- `RUN_DIR/run_steps/prep.geo_unrolled`: The processed geometry file.
- `RUN_DIR/run_steps/prep.geo_unrolled.xao`: The frozen geometry file with stable IDs.
- `RUN_DIR/run_steps/step1.log`: Log file capturing the output of the Gmsh run.

## Notes

After completing STEP 1, the console output will provide the exact command to open the frozen geometry file in Gmsh for inspection. Use this command to verify the geometry and ensure it is ready for STEP 2.
