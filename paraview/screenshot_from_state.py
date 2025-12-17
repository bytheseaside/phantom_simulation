#!/usr/bin/env pvpython
import argparse
from paraview.simple import (
    _DisableFirstRenderCameraReset,
    LoadState,
    FindViewOrCreate,
    SetActiveView,
    Render,
    SaveScreenshot,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--xdmf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=int, default=2000)
    ap.add_argument("--h", type=int, default=2000)
    args = ap.parse_args()

    _DisableFirstRenderCameraReset()

    LoadState(
        args.state,
        filenames=[{
            "FileName": args.xdmf,
            # "id": 13816,
            "name": "BaseCase",
        }],
    )

    renderView1 = FindViewOrCreate("RenderView1", viewtype="RenderView")
    SetActiveView(renderView1)

    Render(renderView1)

    SaveScreenshot(
        filename=args.out,
        viewOrLayout=renderView1,
        ImageResolution=[args.w, args.h],
        TransparentBackground=1,
    )

if __name__ == "__main__":
    main()
