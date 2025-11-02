# Analyze F-Matrix

## Overview

This step analyzes the F-matrix generated from previous simulation steps. It computes various metrics such as correlations, norms, and matrix properties, such as rank, singular values, and condition number.

## Features

- Compute column correlations to roughly assess unit stimuli pattern similarity.
- Analyze matrix rank, condition number, and singular values.
- Generate heatmaps for visualizing correlations.

## Usage

```bash
python analyze_f_matrix.py --matrix <path-to-F_matrix.npy> --out <output-directory>
```

### Options

- `--matrix <path>`: Path to the F-matrix file in `.npy` format.
- `--out <directory>`: Directory to save analysis results (default: `analysis_results`).
- `--abs-corr`: Use absolute values for correlation heatmaps (default: True).
- `--no-abs-corr`: Use signed correlations for heatmaps.
- `--annotate`: Annotate heatmap cells with numeric values (default: True).
- `--no-annotate`: Do not annotate heatmap cells.
- `--fmt <format>`: Format string for numeric annotations (default: `"{:.2f}"`).
