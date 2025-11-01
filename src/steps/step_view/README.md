# View Case

The `View Case` tool is designed to help inspect and visualize simulation results.

## Overview

This tool processes a single `.vtu` file produced by the solver step (step #3).

## Usage

To use the `View Case` tool, ensure that the `.vtu` file is available. Then, execute the viewing script with the required parameters.

**Execution:**

```bash
python src/steps/step_view/view_case.py <path_to_vtu_file> [OPTIONS]
```

### Options

- `--head {surface,volume,off}`: Head display mode (default: `surface`).
- `--gel {off,surface,volume}`: Gel display mode (default: `off`).
- `--no-elec`: Hide electrode surfaces.
- `--pins <csv>`: Load probe positions from a CSV file (columns: name,x,y,z).
- `--global-range`: Use a single color bar across all actors (default: enabled).
- `--static-bar`: Lock color bar ranges to the full dataset (default: enabled).
- `--center-zero`: Center colormap at zero with symmetric ranges (default: enabled).
- `--cmap-head <name>`: Colormap for the head actor (default: `PRGn` if centered, else `viridis`).

## Outputs

- Interactive 3D visualization with options for plane clipping, probe validation, and more.
