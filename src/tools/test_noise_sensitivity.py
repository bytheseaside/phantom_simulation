#!/usr/bin/env python3
"""
test_noise_sensitivity.py

Personal utility script:
- Load A (.npy), compute B = pinv(A)
- Generate dense + sparse x_true
- Apply additive Gaussian noise in measurement space:
    y_noisy = A x + (mu + sigma * Z)
- Reconstruct x_hat = B y_noisy
- Compute relative error per case, print summary
- Optional sweep: mean relative error vs mu (sigma fixed)
- NEW: general wide-range mu sweep (linear OR log), with labeled reference markers
"""

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json


# ----------------------------
# Core utilities
# ----------------------------
def rel_error(x_true, x_hat, eps=1e-15):
    num = np.linalg.norm(x_true - x_hat, axis=0)
    den = np.linalg.norm(x_true, axis=0)
    return num / (den + eps)


def generate_cases(Ne, n_dense, n_sparse, seed=0, signed_sparse=False):
    rng = np.random.default_rng(seed)

    x_dense = (
        rng.normal(0.0, 1.0, size=(Ne, n_dense))
        if n_dense > 0
        else np.array([]).reshape(Ne, 0)
    )

    x_sparse = np.zeros((Ne, n_sparse))
    for j in range(n_sparse):
        k = rng.choice([1, 2])
        idx = rng.choice(Ne, size=k, replace=False)
        vals = np.ones(k)
        if signed_sparse:
            vals *= rng.choice([-1.0, 1.0], size=k)
        x_sparse[idx, j] = vals

    return x_dense, x_sparse


def summarize(err: np.ndarray):
    if err.size == 0:
        return {"N": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "N": int(err.size),
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "std": float(np.std(err, ddof=1)) if err.size > 1 else 0.0,
        "p95": float(np.percentile(err, 95)),
        "max": float(np.max(err)),
    }


def _fmt(x: float) -> str:
    return f"{x: .3e}" if (abs(x) < 1e-3 or abs(x) >= 1e3) else f"{x: .6f}"


def print_summary_block(title: str, A_shape, mu, sigma, stats_dense, stats_sparse):
    Nm, Ne = A_shape
    print("\n" + "=" * 90)
    print(f"{title} | A: {Nm}×{Ne}")
    print(f"Noise model in y:  noise = μ + σ Z  with μ={_fmt(mu)}, σ={_fmt(sigma)}")
    print("-" * 90)

    headers = ["Family", "N", "Mean", "Median", "Std", "P95", "Max"]
    colw = [10, 8, 14, 14, 14, 14, 14]
    print("".join(f"{h:<{w}}" for h, w in zip(headers, colw)))
    print("-" * 90)

    def row(name, st):
        vals = [
            name,
            str(st["N"]),
            _fmt(st["mean"]),
            _fmt(st["median"]),
            _fmt(st["std"]),
            _fmt(st["p95"]),
            _fmt(st["max"]),
        ]
        return "".join(f"{v:<{w}}" for v, w in zip(vals, colw))

    if stats_dense["N"] > 0:
        print(row("Dense", stats_dense))
    if stats_sparse["N"] > 0:
        print(row("Sparse", stats_sparse))
    print("=" * 90 + "\n")


def solve_with_fixed_Z(A, B, X, mu, sigma, Z):
    if X.size == 0:
        return np.array([]), 0.0
    y = A @ X
    noise = mu + sigma * Z
    y_noisy = y + noise
    x_hat = B @ y_noisy
    err = rel_error(X, x_hat)
    nsr = float(
        np.mean(np.linalg.norm(noise, axis=0))
        / (np.mean(np.linalg.norm(y, axis=0)) + 1e-18)
    )
    return err, nsr


def make_fixed_Z(Nm, n_cases, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=(Nm, n_cases))


# ----------------------------
# Plot helpers
# ----------------------------
def save_case_error_plot(
    err_dense,
    err_sparse,
    mean_dense,
    mean_sparse,
    nsr_d,
    nsr_s,
    out_path: Path,
    Nm,
    Ne,
    mu,
    sigma,
    logy=False,
):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    if err_dense is not None and err_dense.size > 0:
        (l1,) = ax.plot(
            err_dense,
            label=rf"$\epsilon_{{\mathrm{{dense}}}}$ (NSR≈{_fmt(nsr_d)})",
            alpha=0.85,
            lw=1.5,
        )
        ax.axhline(
            mean_dense,
            color=l1.get_color(),
            ls="--",
            alpha=0.45,
            label=r"$\overline{\epsilon}_{\mathrm{dense}}$",
        )

    if err_sparse is not None and err_sparse.size > 0:
        (l2,) = ax.plot(
            err_sparse,
            label=rf"$\epsilon_{{\mathrm{{sparse}}}}$ (NSR≈{_fmt(nsr_s)})",
            alpha=0.85,
            lw=1.5,
        )
        ax.axhline(
            mean_sparse,
            color=l2.get_color(),
            ls="--",
            alpha=0.45,
            label=r"$\overline{\epsilon}_{\mathrm{sparse}}$",
        )

    ax.set_xlabel("Test case index")
    ax.set_ylabel(r"Relative error  $\|x-\hat{x}\|_2 / \|x\|_2$")
    ax.set_title(
        rf"Noise-only reconstruction error (A: {Nm}×{Ne})" + "\n" + rf"$\mu$={_fmt(mu)}, $\sigma$={_fmt(sigma)}"
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    if logy:
        ax.set_yscale("log")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=True,
            bbox_to_anchor=(0.5, 0.02),
            fontsize=9,
        )
        fig.tight_layout(rect=[0, 0.12, 1, 1])
    else:
        fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_mu_sweep_plot(mus, mean_ed, mean_es, out_path: Path, sigma_fixed: float, sigma_min: float | None, logx: bool):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)

    if mean_ed.size > 0:
        ax.plot(mus, mean_ed, label=r"$\epsilon_{\mathrm{dense}}$ (mean)", alpha=0.9, lw=1.8)
    if mean_es.size > 0:
        ax.plot(mus, mean_es, label=r"$\epsilon_{\mathrm{sparse}}$ (mean)", alpha=0.9, lw=1.8)

    if sigma_min is not None and mus.min() <= sigma_min <= mus.max():
        ax.axvline(sigma_min, ls="--", lw=1.2, color="black", alpha=0.35)
        y_max = ax.get_ylim()[1]
        ax.text(
            sigma_min, y_max * 0.9, r"$\mu = \sigma_{\min}$",
            rotation=90, va="top", ha="right", alpha=0.55, fontsize=10
        )

    ax.set_xlabel(r"Noise mean ($\mu$)")
    ax.set_ylabel("Mean relative error")
    ax.set_title("Noise Sensitivity Sweep\n" + rf"$\sigma$ = {_fmt(sigma_fixed)}")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    if logx:
        ax.set_xscale("log")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", ncol=2, frameon=True,
            bbox_to_anchor=(0.5, 0.02), fontsize=9
        )
        fig.tight_layout(rect=[0, 0.12, 1, 1])
    else:
        fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def save_general_mu_sweep_plot(
    mus,
    mean_ed,
    mean_es,
    out_path: Path,
    sigma_fixed: float,
    markers: list[float],
    logx: bool,
):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    dense_line = None
    sparse_line = None

    if mean_ed.size > 0:
        (dense_line,) = ax.plot(
            mus, mean_ed,
            label=r"$\epsilon_{\mathrm{dense}}$ (mean)",
            lw=2.0, alpha=0.9
        )
    if mean_es.size > 0:
        (sparse_line,) = ax.plot(
            mus, mean_es,
            label=r"$\epsilon_{\mathrm{sparse}}$ (mean)",
            lw=2.0, alpha=0.9
        )

    # --- Marker points + staggered labels (avoid overlap) ---
    # Alternate label placement and increase offset as we move through markers
    bbox_kw = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75)
    arrow_kw = dict(arrowstyle="-", lw=0.8, alpha=0.35)

    # keep only markers within range
    markers_in = [m for m in markers if (mus.min() <= m <= mus.max())]

    for i, m in enumerate(markers_in):
        # base offsets: alternate up/down, and increase with i to separate crowded zones
        base_dx = 2
        base_dy =  12

        # Dense
        if dense_line is not None:
            yd = float(np.interp(m, mus, mean_ed))
            ax.plot([m], [yd], marker="x", ms=7, mew=1.6,
                    color=dense_line.get_color(), ls="None")

            # alternate: above for even, below for odd
            ax.annotate(
                f"{yd:.2e}",
                xy=(m, yd),
                xytext=(base_dx, -base_dy),
                textcoords="offset points",
                color=dense_line.get_color(),
                fontsize=9,
                bbox=bbox_kw,
                arrowprops=arrow_kw,
                ha="left",
                va="bottom",
            )

        # Sparse (shift opposite direction to avoid stacking with dense label)
        if sparse_line is not None:
            ys = float(np.interp(m, mus, mean_es))
            ax.plot([m], [ys], marker="x", ms=7, mew=1.6,
                    color=sparse_line.get_color(), ls="None")

            ax.annotate(
                f"{ys:.2e}",
                xy=(m, ys),
                xytext=(-base_dx, base_dy),
                textcoords="offset points",
                color=sparse_line.get_color(),
                fontsize=9,
                bbox=bbox_kw,
                arrowprops=arrow_kw,
                ha="left",
                va="top",
            )

    ax.set_xlabel(r"Noise mean ($\mu$)")
    ax.set_ylabel("Mean relative error")
    ax.set_title("General μ Sweep (wide range)\n" + rf"$\sigma$ = {_fmt(sigma_fixed)}")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)

    if logx:
        ax.set_xscale("log")

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="lower center", ncol=2, frameon=True,
            bbox_to_anchor=(0.5, 0.02), fontsize=9
        )
        fig.tight_layout(rect=[0, 0.12, 1, 1])
    else:
        fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def embed_reduced_vec_into_full(x_reduced: np.ndarray, full_ne: int, dropped_col_full: int) -> np.ndarray:
    """
    x_reduced: shape (full_ne-1,)
    Returns x_full: shape (full_ne,), inserting 0.0 at dropped_col_full (0-based).
    """
    if x_reduced.ndim != 1:
        raise ValueError(f"x_reduced must be 1D, got shape {x_reduced.shape}")
    if full_ne <= 0:
        raise ValueError("full_ne must be positive")
    if not (0 <= dropped_col_full < full_ne):
        raise ValueError(f"dropped_col_full out of range: {dropped_col_full} for full_ne={full_ne}")
    if x_reduced.shape[0] != full_ne - 1:
        raise ValueError(
            f"x_reduced length must be full_ne-1 ({full_ne-1}), got {x_reduced.shape[0]}"
        )

    x_full = np.zeros((full_ne,), dtype=float)
    keep = [i for i in range(full_ne) if i != dropped_col_full]
    x_full[np.array(keep)] = x_reduced
    return x_full


def build_dirichlet_from_x_full(x_full: np.ndarray) -> list[dict]:
    """
    x_full: shape (full_ne,)
    Creates 3 dicts per electrode i=1..full_ne:
      e{i}_T = x_full[i-1]
      e{i}_R = -x_full[i-1]
      e{i}_S = 0.0
    """
    if x_full.ndim != 1:
        raise ValueError(f"x_full must be 1D, got shape {x_full.shape}")

    full_ne = int(x_full.shape[0])
    out: list[dict] = []
    for i0 in range(full_ne):  # 0-based index
        ei = i0 + 1            # 1-based name
        vT = float(x_full[i0])
        out.append({"name": f"e{ei}_T", "value": vT})
        out.append({"name": f"e{ei}_R", "value": -vT})
        out.append({"name": f"e{ei}_S", "value": 0.0})
    return out


def export_manifest_cases_json(
    out_json_path: Path,
    x_dense: np.ndarray,
    x_sparse: np.ndarray,
    full_ne: int,
    dropped_col_full: int,
):
    """
    x_dense/x_sparse are in reduced space: shape (full_ne-1, n_cases)
    Writes:
      {"cases":[{"name":"test-0001-dense","dirichlet":[...]} , ...]}
    """
    cases: list[dict] = []
    case_id = 1

    def add_family(family: str, X: np.ndarray):
        nonlocal case_id
        if X.size == 0:
            return
        if X.shape[0] != full_ne - 1:
            raise ValueError(
                f"{family}: expected X.shape[0] == full_ne-1 ({full_ne-1}), got {X.shape[0]}"
            )
        for j in range(X.shape[1]):
            x_red = X[:, j]
            x_full = embed_reduced_vec_into_full(
                x_reduced=x_red,
                full_ne=full_ne,
                dropped_col_full=dropped_col_full,
            )
            name = f"test-{case_id:04d}-{family}"  # file-safe
            cases.append({"name": name, "dirichlet": build_dirichlet_from_x_full(x_full)})
            case_id += 1

    add_family("dense", x_dense)
    add_family("sparse", x_sparse)

    payload = {"cases": cases}
    out_json_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved manifest cases JSON to: {out_json_path}  (N={len(cases)})")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A_path", type=Path, help="Path to A matrix (.npy)")
    args = ap.parse_args()

    outPath = args.A_path.with_name(args.A_path.stem + "_noise_case.png")

    # ====== USER CONSTANTS (edit here) ======
    N_DENSE = 250
    N_SPARSE = 250
    X_SEED = 3243
    SIGNED_SPARSE = True

    # Single-run noise config
    MU = 5e-4
    SIGMA = 1e-6
    LOGY_CASE_PLOT = False

    # Sweep config (local window around sigma_min)
    DO_MU_SWEEP = True
    SIGMA_MIN_FOR_VLINE = 0.01441493471008068  # NOTE: singular value (not noise sigma)
    MU_SWEEP_A = SIGMA_MIN_FOR_VLINE - 0.01
    MU_SWEEP_B = SIGMA_MIN_FOR_VLINE + 0.01
    MU_SWEEP_N = 100
    MU_SWEEP_LOG = False
    SWEEP_OUT = outPath.with_name(args.A_path.stem + "_mu_sweep.png")

    # General sweep (wide range)
    DO_GENERAL_MU_SWEEP = True
    GENERAL_MU_A = 1e-5
    GENERAL_MU_B = 1e-2
    GENERAL_MU_N = 120
    GENERAL_MU_LOG = True
    GENERAL_SWEEP_OUT = outPath.with_name(args.A_path.stem + "_mu_sweep_general.png")

    # Marker mus to annotate (must be within [GENERAL_MU_A, GENERAL_MU_B] to show)
    GENERAL_MARKERS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    # ====== EXPORT CONFIG ======
    # Full simulator expects 9 electrodes e1..e9:
    FULL_NE = 9

    # You removed one column from the full operator to make the reduced A.
    # 0-based (numpy) column index in the FULL operator:
    DROPPED_COL_FULL = 5  # -> corresponds to electrode e6_* in simulator naming
    # =======================================

    # Load + pinv
    A = np.load(args.A_path)
    B = np.linalg.pinv(A)
    Nm, Ne = A.shape

    # Generate cases
    x_dense, x_sparse = generate_cases(Ne, N_DENSE, N_SPARSE, seed=X_SEED, signed_sparse=SIGNED_SPARSE)

    manifest_path = args.A_path.with_name("manifest_cases.json")

    print("\n[EXPORT] manifest JSON export ENABLED")
    print(f"[EXPORT] target path: {manifest_path.resolve()}")

    if Ne != FULL_NE - 1:
        raise ValueError(
            f"[EXPORT] A has Ne={Ne} columns, but FULL_NE={FULL_NE} implies reduced should have {FULL_NE-1} columns."
        )

    export_manifest_cases_json(
        out_json_path=manifest_path,
        x_dense=x_dense,
        x_sparse=x_sparse,
        full_ne=FULL_NE,
        dropped_col_full=DROPPED_COL_FULL,
    )

    # sanity check
    if not manifest_path.exists():
        raise RuntimeError(f"[EXPORT] Write reported success but file does not exist: {manifest_path.resolve()}")
    print(f"[EXPORT] file size: {manifest_path.stat().st_size} bytes\n")


    # Fixed Z for reproducibility
    Zd = make_fixed_Z(Nm, x_dense.shape[1], seed=101)
    Zs = make_fixed_Z(Nm, x_sparse.shape[1], seed=202)

    # ---- Single run ----
    err_dense, nsr_d = solve_with_fixed_Z(A, B, x_dense, MU, SIGMA, Zd)
    err_sparse, nsr_s = solve_with_fixed_Z(A, B, x_sparse, MU, SIGMA, Zs)

    st_dense = summarize(err_dense)
    st_sparse = summarize(err_sparse)

    print_summary_block("MATRIX RECONSTRUCTION REPORT", (Nm, Ne), MU, SIGMA, st_dense, st_sparse)

    save_case_error_plot(
        err_dense=err_dense, err_sparse=err_sparse,
        mean_dense=st_dense["mean"], mean_sparse=st_sparse["mean"],
        nsr_d=nsr_d, nsr_s=nsr_s,
        out_path=outPath, Nm=Nm, Ne=Ne,
        mu=MU, sigma=SIGMA, logy=LOGY_CASE_PLOT
    )
    print(f"Saved case-error plot to: {outPath}")

    # ---- Optional local MU sweep ----
    if DO_MU_SWEEP:
        mus = (
            np.logspace(np.log10(MU_SWEEP_A), np.log10(MU_SWEEP_B), MU_SWEEP_N)
            if MU_SWEEP_LOG
            else np.linspace(MU_SWEEP_A, MU_SWEEP_B, MU_SWEEP_N)
        )

        m_ed, m_es = [], []
        for mu_i in mus:
            ed, _ = solve_with_fixed_Z(A, B, x_dense, float(mu_i), SIGMA, Zd)
            es, _ = solve_with_fixed_Z(A, B, x_sparse, float(mu_i), SIGMA, Zs)
            m_ed.append(np.mean(ed) if ed.size > 0 else None)
            m_es.append(np.mean(es) if es.size > 0 else None)

        save_mu_sweep_plot(
            mus=mus,
            mean_ed=np.array([x for x in m_ed if x is not None]),
            mean_es=np.array([x for x in m_es if x is not None]),
            out_path=SWEEP_OUT,
            sigma_fixed=SIGMA,
            sigma_min=SIGMA_MIN_FOR_VLINE,
            logx=MU_SWEEP_LOG,
        )
        print(f"Saved μ-sweep plot to: {SWEEP_OUT}")

    # ---- General wide-range MU sweep ----
    if DO_GENERAL_MU_SWEEP:
        if GENERAL_MU_LOG:
            if GENERAL_MU_A <= 0 or GENERAL_MU_B <= 0:
                raise ValueError("For GENERAL_MU_LOG=True, GENERAL_MU_A and GENERAL_MU_B must be > 0.")
            mus_g = np.logspace(np.log10(GENERAL_MU_A), np.log10(GENERAL_MU_B), GENERAL_MU_N)
        else:
            mus_g = np.linspace(GENERAL_MU_A, GENERAL_MU_B, GENERAL_MU_N)

        mean_ed_g, mean_es_g = [], []
        for mu_i in mus_g:
            ed, _ = solve_with_fixed_Z(A, B, x_dense, float(mu_i), SIGMA, Zd)
            es, _ = solve_with_fixed_Z(A, B, x_sparse, float(mu_i), SIGMA, Zs)
            mean_ed_g.append(float(np.mean(ed)) if ed.size > 0 else np.nan)
            mean_es_g.append(float(np.mean(es)) if es.size > 0 else np.nan)

        mean_ed_g = np.array(mean_ed_g)
        mean_es_g = np.array(mean_es_g)

        save_general_mu_sweep_plot(
            mus=mus_g,
            mean_ed=mean_ed_g,
            mean_es=mean_es_g,
            out_path=GENERAL_SWEEP_OUT,
            sigma_fixed=SIGMA,
            markers=GENERAL_MARKERS,
            logx=GENERAL_MU_LOG,
        )
        print(f"Saved GENERAL μ-sweep plot to: {GENERAL_SWEEP_OUT}")
        print(f"General μ range: [{_fmt(float(mus_g.min()))}, {_fmt(float(mus_g.max()))}] "
              f"({GENERAL_MU_N} points, log={GENERAL_MU_LOG})")

    # Optional: reprint baseline summary at end (nice for long sweeps)
    print_summary_block("FINAL SUMMARY — SINGLE RUN (REFERENCE)", (Nm, Ne), MU, SIGMA, st_dense, st_sparse)


if __name__ == "__main__":
    main()
