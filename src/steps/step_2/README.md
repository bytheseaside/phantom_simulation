# STEP 2: Labeling and Meshing

STEP 2 processes the prepared geometry from STEP 1, applies physical labels, and generates a simulation mesh.

## Overview

STEP 2 begins by using the static geometry generated in STEP 1. After completing STEP 1, the console output will indicate the exact command to open the frozen geometry file (e.g., `gmsh run_YYYYMMDD_HHMMSS/run_steps/prep.geo_unrolled.xao`). Use this command to open the file in Gmsh GUI and inspect the IDs of the geometric entities. These IDs must then be recorded in the provided template file (`template_ids.inc.geo`) located in the `templates` folder.

Once the IDs are verified and recorded, move the template file to the `run_steps` folder of the corresponding run directory, renaming it to `ids.inc.geo`. This ensures that STEP 2 can correctly reference the physical groups and entities during the meshing process.

Finally, execute STEP 2 from the `run_steps` folder of the run directory. This will generate the labeled simulation mesh and log files for downstream use.

**Execution:**

```bash
bash src/steps/step_2/run_step_02.sh RUN_DIR/run_steps
```

**Outputs:**

- `RUN_DIR/mesh.msh`: The labeled simulation mesh saved in the parent directory of `run_steps`.
- `RUN_DIR/run_steps/step2.log`: Log file capturing the output of the Gmsh run.
