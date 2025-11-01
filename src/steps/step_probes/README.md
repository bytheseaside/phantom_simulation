# Step 4: Probing VTU Files

This step performs post-processing on simulation results by probing VTU files using PyVista. It extracts data points based on a provided `probes.csv` file and generates CSV outputs for each VTU file.

## Overview

The script supports probing either a single VTU file or all VTU files in a specified folder. The results are saved in the `cases` directory of the run folder.

## Usage

```bash
bash run_probing.sh --run-dir /path/to/run_dir --probes /path/to/probes.csv
bash run_probing.sh --single-file /path/to/vtu_file --probes /path/to/probes.csv
```

### Optional Arguments

- `--var`: (Optional) Specify the scalar array to sample (default: `u`).

## Outputs

- `cases/probes_<case>.csv`: Probed data for each VTU file.
- `step4.log`: Log file capturing the execution details.

## Notes

Ensure that the `probes.csv` file is correctly formatted and located at the root of the run directory. The script now supports specifying a single VTU file or a folder containing VTU files. If `--var` is not provided, the default scalar array `u` is used.
