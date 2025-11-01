# STEP 3: Solver Execution

STEP 3 uses the labeled mesh and simulation cases to solve the simulation and generate results.

## Overview

STEP 3 begins by utilizing the labeled mesh (`mesh.msh`) and simulation cases (`simulation_cases.csv`) from the specified run directory (`RUN_DIR`). These files, which must be placed directly in the `RUN_DIR`, serve the following purposes:

- `mesh.msh`: Defines the simulation domain.
- `simulation_cases.csv`: Specifies the simulation cases to process.

The solver uses these inputs to generate results stored in the `RUN_DIR/cases` directory. Outputs include files in various formats suitable for visualization and analysis. Logs and configuration files are saved in the `RUN_DIR/run_steps` folder to ensure traceability and support debugging.

**Execution:**

```bash
bash src/steps/step_3/run_step_03.sh RUN_DIR
```

**Outputs:**

- `RUN_DIR/cases/`: Contains the simulation results.
- `RUN_DIR/run_steps/`: Contains solver logs and configuration files.
- `RUN_DIR/simulation_cases.csv`: Defines the simulation cases used in the solver.
