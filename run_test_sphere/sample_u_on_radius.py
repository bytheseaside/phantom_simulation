#!/usr/bin/env pvpython
# -*- coding: utf-8 -*-

from pathlib import Path
from paraview.simple import (
    CSVReader,
    TableToPoints,
    Xdmf3ReaderS,
    ResampleWithDataset,
    CreateView,
    Show,
    Render,
    SaveData,
)

def main(case: str, probes: str, out: str) -> None:
    case = Path(case)
    probes = Path(probes)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- Probes CSV -> points
    probes_csv = CSVReader(registrationName="sphere_probes.csv", FileName=[str(probes)])

    probe_points = TableToPoints(registrationName="TableToPoints1", Input=probes_csv)
    probe_points.XColumn = "x"
    probe_points.YColumn = "y"
    probe_points.ZColumn = "z"

    # --- XDMF case
    mesh = Xdmf3ReaderS(registrationName=case.name, FileName=[str(case)])
    mesh.UpdatePipeline()

    # --- Resample
    sampled = ResampleWithDataset(
        registrationName="ResampleWithDataset1",
        SourceDataArrays=mesh,
        DestinationMesh=probe_points,
    )
    sampled.PassPointArrays = 1
    sampled.PassCellArrays = 0
    sampled.PassFieldArrays = 0

    sampled.UpdatePipeline()

    # --- ACTUALLY RENDER A VIEW (what you asked for)
    render_view = CreateView("RenderView")
    Show(sampled, render_view)     # show resampled data
    Render(render_view)            # force a full render pass

    # --- Save CSV (same settings as your working GUI trace)
    SaveData(
        str(out),
        proxy=sampled,
        ChooseArraysToWrite=1,
        PointDataArrays=["u"],
        Precision=6,
        UseScientificNotation=1,
    )

    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Sample scalar u on probe points (with Render before SaveData)"
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--out", required=True)  # must end in .csv
    a = parser.parse_args()
    main(a.case, a.probes, a.out)
