#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# CONSTANTS
# -------------------------
R_INNER = 0.001  # m (1 mm)
R_OUTER = 0.005  # m (5 mm)
# -------------------------


@dataclass(frozen=True)
class CaseMeta:
    sigma: float
    u_int: float
    ext_is_neumann: bool
    u_ext: Optional[float]  # None if NEUMANN
    filename: str


def _parse_decimal_with_comma(s: str) -> float:
    return float(s.replace(",", "."))


def parse_case_filename(path: Path) -> CaseMeta:
    """
    Expected:
      sigma=3,5-int=1,23-ext=2,22.csv
      sigma=3,5-int=1,23-ext=NEUMANN.csv
    """
    name = path.name
    m = re.search(r"sigma=([^-\s]+)-int=([^-\s]+)-ext=([^.\s]+)\.csv$", name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {name}")

    sigma_s, u_int_s, ext_s = m.group(1), m.group(2), m.group(3)
    sigma = _parse_decimal_with_comma(sigma_s)
    u_int = _parse_decimal_with_comma(u_int_s)

    if ext_s.upper() == "NEUMANN":
        return CaseMeta(sigma=sigma, u_int=u_int, ext_is_neumann=True, u_ext=None, filename=name)

    u_ext = _parse_decimal_with_comma(ext_s)
    return CaseMeta(sigma=sigma, u_int=u_int, ext_is_neumann=False, u_ext=u_ext, filename=name)


# -------------------------
# ANALYTIC (YOU FILLED)
# -------------------------
def u_analytic_dir_neumann(
    r: np.ndarray, *, u_int: float, r_inner: float, r_outer: float
) -> np.ndarray:
    """
    Dirichlet at inner surface, Neumann at outer surface.
    Defined only for r_inner <= r <= r_outer.
    """
    u = np.full_like(r, np.nan, dtype=float)
    mask = (r >= r_inner) & (r <= r_outer)
    u[mask] = u_int
    return u


def u_analytic_dir_dir(
    r: np.ndarray, *, u_int: float, u_ext: float, r_inner: float, r_outer: float
) -> np.ndarray:
    """
    Dirichlet at inner and outer surfaces.
    Defined only for r_inner <= r <= r_outer.
    """
    u = np.full_like(r, np.nan, dtype=float)
    mask = (r >= r_inner) & (r <= r_outer)
    u[mask] = (((u_ext - u_int) / (1.0 / r_outer - 1.0 / r_inner)) * (1.0 / r[mask] - 1.0 / r_inner)) + u_int
    return u
# -------------------------


def load_sampled_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    ParaView SaveData CSV.
    Given the symmetric nature of the problem, we probe along a radius aligned with the x-axis,
    hence r = |x| and y=z=0.
    """
    df = pd.read_csv(csv_path)

    if "Points:0" not in df.columns:
        raise ValueError(f"Missing Points:0 in {csv_path.name}. Columns: {list(df.columns)}")
    if "u" not in df.columns:
        raise ValueError(f"Missing u in {csv_path.name}. Columns: {list(df.columns)}")

    r = df["Points:0"].to_numpy(dtype=float)
    u = df["u"].to_numpy(dtype=float)

    order = np.argsort(r)
    return r[order], u[order]


def case_title(meta: CaseMeta) -> str:
    if meta.ext_is_neumann:
        return (
            rf"$\sigma={meta.sigma:g}\,\mathrm{{S/m}},\ "
            rf"u(r_{{\mathrm{{int}}}})={meta.u_int:g}\,\mathrm{{V}},\ "
            rf"\partial u/\partial n(r_{{\mathrm{{ext}}}})=0$"
        )
    else:
        return (
            rf"$\sigma={meta.sigma:g}\,\mathrm{{S/m}},\ "
            rf"u(r_{{\mathrm{{int}}}})={meta.u_int:g}\,\mathrm{{V}},\ "
            rf"u(r_{{\mathrm{{ext}}}})={meta.u_ext:g}\,\mathrm{{V}}$"
        )


def main(cases: List[str], out: str) -> None:
    case_paths = [Path(c) for c in cases]
    for p in case_paths:
        if not p.exists():
            raise FileNotFoundError(str(p))
        if p.suffix.lower() != ".csv":
            raise ValueError(f"Not a .csv file: {p}")

    metas = [parse_case_filename(p) for p in case_paths]

    n = len(case_paths)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(14, 8), 
        squeeze=False
    )
    fig.suptitle("Spherical mesh: FEM samples vs analytic solution", fontsize=14)

    for i, (p, meta) in enumerate(zip(case_paths, metas)):
        ax = axes[i // ncols][i % ncols]

        r, u_meas = load_sampled_csv(p)

        # Analytic evaluated at the same sample radii (no interpolation needed)
        if meta.ext_is_neumann:
            u_th = u_analytic_dir_neumann(r, u_int=meta.u_int, r_inner=R_INNER, r_outer=R_OUTER)
        else:
            assert meta.u_ext is not None
            u_th = u_analytic_dir_dir(r, u_int=meta.u_int, u_ext=meta.u_ext, r_inner=R_INNER, r_outer=R_OUTER)

        # Measured: dots only (B1)
        ax.plot(r, u_meas, linestyle="None", marker="o", markersize=3, label="Measured (FEM)")

        # Analytic: thicker line behind
        ax.plot(r, u_th, linewidth=3, alpha=0.6, label="Analytic")

        ax.set_title(case_title(meta), fontsize=11)
        ax.set_xlabel("r [m]")
        ax.set_ylabel("u(r) [V]")
        ax.grid(True)
        # --- Y padding for readability ---
        y_all = np.concatenate([
            u_meas[np.isfinite(u_meas)],
            u_th[np.isfinite(u_th)]
        ])

        if y_all.size > 0:
            y_min = y_all.min()
            y_max = y_all.max()

            # Relative padding (10% of range, fallback if flat)
            span = y_max - y_min
            if span == 0:
                pad = 0.05 * max(abs(y_max), 1.0)
            else:
                pad = 0.10 * span

            ax.set_ylim(y_min - pad, y_max + pad)


    # Hide any unused subplots if odd number of cases
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.subplots_adjust(wspace=0.15)
    handles, labels = axes[0][0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.93)
    )

    fig.savefig(out_path, dpi=300)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot sphere validation (one subplot per case).")
    parser.add_argument("--out", required=True, help="Output image path (.png or .pdf)")
    parser.add_argument("--cases", required=True, nargs="+", help="CSV case files.")
    args = parser.parse_args()

    main(cases=args.cases, out=args.out)
