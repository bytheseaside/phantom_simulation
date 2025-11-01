# Probes template for the phantom simulation

This directory contains the probe definitions and helper scripts used in the phantom head simulation.
The probe coordinates were derived from the Fusion 360 CAD model of the phantom assembly (`simulation_assembly.step`).

## Overview

- The probe positions were **manually placed in Fusion 360** following the standard **10–20 international EEG system**, ensuring the relative spacing between points matched the system’s geometric proportions over the head surface.
- Each probe corresponds to a **construction point** created on the scalp surface of the CAD model.
- After placement, the coordinates of these points were exported directly from the CAD assembly.
- The exported coordinates are in **millimetres (mm)**.
  When required, use `coord_to_meters.py` to convert them to **metres (m)** for simulation consistency.

## Contents

- **`coord_to_meters.py`** — Utility script to convert a CSV of coordinates from millimetres to metres.
- **`10_20_int_system_probes_mm.csv`** — Probe coordinates as exported from the CAD model, in millimetres. Represents the 21 standard electrode positions of the 10–20 international EEG system.
- **`10_20_int_system_probes.csv`** — Same coordinates converted to metres using the utility above.

## Usage Example

To convert a CSV file with probe coordinates in millimetres to metres, run:

```bash
python3 coord_to_meters.py 10_20_int_system_probes_mm.csv > 10_20_int_system_probes.csv
```
