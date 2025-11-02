# Build F-Matrix

## Overview

This step constructs the forward transfer matrix (F-matrix) for a set of VTU files. The F-matrix represents the voltage values at probe locations for each simulation case. The column names are derived from the filenames of the VTU files (e.g., `solution_CASE-NAME.vtu` → `CASE-NAME`).

## Features

- Supports flexible input: any directory of VTU files.
- Automatically derives column names from VTU filenames.
- Samples voltage values at probe locations.

## Usage

### Build the F-Matrix

```bash
python build_f_matrix.py --vtu-dir <path-to-vtu-directory> --probes <path-to-probes.csv> --out <output-directory>
```

### Options

- `--vtu-dir <path>`: Directory containing VTU files.
- `--probes <path>`: Path to the probe CSV file (format: `name,x,y,z`).
- `--out <directory>`: Directory to save outputs (default: `f_matrix_output`).

## Outputs

- `F_matrix.npy`: Raw F-matrix.
- `F_matrix_heatmap.png`: Heatmap visualization of the F-matrix.
- `metadata.json`: Metadata with matrix statistics (e.g., max value, mean value).
