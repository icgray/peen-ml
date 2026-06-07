#!/usr/bin/env python3
"""
analytical_mode.py
==================
Analytical shot peening surface-displacement predictor.

Given a sequence of impact events (position, optional timing) and material
parameters, computes the total surface displacement field using two models:

  A. Shen & Atluri (2006)  — full elastic-plastic loading + unloading cycle.
     Gives the *post-impact residual* (permanent) dent. This is the same
     physics that underpins the multi-shot simulator and the training datasets.

  B. Sherafatnia / Poozesh — Hertzian *elastic* loading stresses.
     No unloading is modelled, so the permanent dent is estimated from the
     elastic contact radius using a yield-scaled indentation depth.
     Sherafatnia typically over-predicts the dent area and under-predicts
     the dent depth compared to the elastic-plastic result.

Both models use linear superposition: each shot's contribution is computed
independently and added. Shot order affects timing annotations only; it does
not change the final displacement field unless sequential plasticity tracking
is explicitly enabled (--sequential flag).

Usage
-----
    # Compare both models against simulation 0 in a dataset:
    python analytical_mode.py --dataset LargeScaleRun1/Dataset_Ti_6Al_4V__steel_200 --sim 0

    # Average over N random simulations in a dataset:
    python analytical_mode.py --dataset LargeScaleRun1/Dataset_Ti_6Al_4V__steel_200 --n 20

    # Provide shot positions directly (CSV with columns: x_m, y_m, t_s):
    python analytical_mode.py --shots my_shots.csv --V 40 --D 0.6 --material Ti-6Al-4V --shot-material steel

    # Save comparison figure:
    python analytical_mode.py --dataset ... --sim 0 --out comparison.png
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src" / "peen-ml"
for _p in [str(_SRC), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from impact_sim import (  # noqa: E402
    ShotPeenParams,
    compute_contact_params,
    compute_plastic_zone,
    map_displacements,
)


# ---------------------------------------------------------------------------
# 1.  Sherafatnia displacement model
# ---------------------------------------------------------------------------


def _sherafatnia_single_impact(
    params: ShotPeenParams,
    node_xy: np.ndarray,
    impact_center_xy: np.ndarray,
) -> np.ndarray:
    """Permanent surface displacement for ONE Sherafatnia impact.

    Sherafatnia / Poozesh computes elastic Hertzian contact stresses (during
    loading) but does not model unloading.  We estimate the permanent dent as
    the elastic indentation scaled by a yield factor:

        delta_e  = a_e² / (2 R)            [elastic Hertz indentation depth]
        f_yield  = clip((p0 - sigma_y)/p0, 0, 1)   [plasticity fraction]
        delta_p* = delta_e * f_yield               [estimated permanent depth]

    The dent profile uses the same Hertzian paraboloid shape as Shen-Atluri
    but with the *elastic* contact radius a_e rather than the *plastic* radius
    a_p.  Because a_e > a_p for ductile materials, Sherafatnia predicts a
    shallower, wider dent than Shen-Atluri.

    Returns
    -------
    disp : (N, 3) float32  [ux, uy, uz]  for N nodes
    """
    E_s, nu_s = params.E_s, params.nu_s
    E_b, nu_b = params.E_b, params.nu_b
    sigma_y = params.sigma_yield
    R = params.R
    V = params.V
    rho_s = params.rho_s

    # Equivalent modulus (Hertz)
    E_eq = 1.0 / ((1 - nu_s**2) / E_s + (1 - nu_b**2) / E_b)

    # Sherafatnia peak contact pressure (energy-based, normal impact assumed)
    k = 0.8  # energy restitution coefficient
    p0 = (1.0 / math.pi) * (40.0 * math.pi * rho_s * V**2 * k * E_eq**4) ** (1.0 / 5.0)

    # Elastic contact radius (Hertz: a_e = π p0 R / (2 E_eq))
    a_e = (math.pi * p0 * R) / (2.0 * E_eq)
    if a_e <= 0:
        return np.zeros((len(node_xy), 3), dtype=np.float32)

    # Elastic indentation depth
    delta_e = a_e**2 / (2.0 * R)

    # Yield scaling: permanent fraction based on how far p0 exceeds sigma_y
    f_yield = float(np.clip((p0 - sigma_y) / max(p0, 1e-12), 0.0, 1.0))
    delta_perm = delta_e * f_yield  # estimated permanent dent depth

    # Radial distance from impact centre
    dx = node_xy[:, 0] - impact_center_xy[0]
    dy = node_xy[:, 1] - impact_center_xy[1]
    r = np.sqrt(dx**2 + dy**2)

    # Dent profile: paraboloid inside a_e, zero outside
    uz = np.zeros(len(node_xy), dtype=np.float64)
    mask = r <= a_e
    uz[mask] = -delta_perm * (1.0 - (r[mask] / a_e) ** 2)

    # Radial bulge (use a_e as the bulge radius)
    ur = (2.0 / 3.0) * delta_perm * (r / a_e) * np.exp(-((r / a_e) ** 2))
    safe_r = np.where(r > 0, r, 1.0)
    ux = ur * np.where(r > 0, dx / safe_r, 0.0)
    uy = ur * np.where(r > 0, dy / safe_r, 0.0)

    return np.stack([ux, uy, uz], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 2.  Shen-Atluri displacement model (wraps existing map_displacements)
# ---------------------------------------------------------------------------


def _shen_atluri_single_impact(
    params: ShotPeenParams,
    node_coords: np.ndarray,
    impact_center_xy: np.ndarray,
) -> np.ndarray:
    """Permanent surface displacement for ONE Shen-Atluri impact.

    Uses the full elastic-plastic loading+unloading cycle from impact_sim.py.

    Returns
    -------
    disp : (N, 3) float32  [ux, uy, uz]
    """
    contact = compute_contact_params(params)
    plastic = compute_plastic_zone(params)

    N = len(node_coords)
    # Build a minimal mesh dict
    mesh = {
        "node_coords": node_coords,
        "node_labels": np.arange(N, dtype=np.int32),
        "impact_center": np.array([impact_center_xy[0], impact_center_xy[1], 0.0]),
    }
    ic = np.array([impact_center_xy[0], impact_center_xy[1], 0.0])
    _, disp = map_displacements(mesh, contact, plastic, params, impact_center=ic)
    return disp


# ---------------------------------------------------------------------------
# 3.  Multi-shot superposition
# ---------------------------------------------------------------------------


def predict_analytical(
    node_coords: np.ndarray,
    shot_positions: np.ndarray,
    params: ShotPeenParams,
    model: str = "shen_atluri",
    shot_times: Optional[np.ndarray] = None,
    sequential: bool = False,
) -> np.ndarray:
    """Compute total surface displacement for a multi-shot sequence.

    Parameters
    ----------
    node_coords   : (N, 3) node positions in metres.
    shot_positions: (S, 2) impact centres (x, y) in metres.
    params        : ShotPeenParams with material + shot properties.
    model         : 'shen_atluri' or 'sherafatnia'.
    shot_times    : (S,) optional timing array (seconds).  Currently only used
                    for ordering and annotation — sequential effects are
                    approximated via the ``sequential`` flag.
    sequential    : If True (experimental), reduce the effective V for later
                    shots that land inside an already-worked plastic zone,
                    mimicking work-hardening (coverage > 100% saturation).
                    This is a first-order approximation only.

    Returns
    -------
    disp : (N, 3) float32  cumulative [ux, uy, uz] (metres).
    """
    if shot_times is not None:
        order = np.argsort(shot_times)
        shot_positions = shot_positions[order]

    node_xy = node_coords[:, :2]
    total = np.zeros((len(node_coords), 3), dtype=np.float32)

    # For sequential mode: track whether each node is in a worked zone
    worked_fraction = np.zeros(len(node_coords), dtype=np.float32) if sequential else None

    for shot_xy in shot_positions:
        p = params  # default: use original params
        if sequential and worked_fraction is not None:
            # Nodes with high prior displacement get reduced effective V
            dx = node_xy[:, 0] - shot_xy[0]
            dy = node_xy[:, 1] - shot_xy[1]
            r = np.sqrt(dx**2 + dy**2)
            contact = compute_contact_params(params)
            a_ref = contact.get("a_p", contact.get("a_e", 1e-4))
            in_zone = r <= a_ref
            prior_work = float(np.mean(worked_fraction[in_zone])) if in_zone.any() else 0.0
            # Work-hardening: raise yield strength proportionally
            import dataclasses

            p = dataclasses.replace(params, sigma_yield=params.sigma_yield * (1.0 + prior_work))
            worked_fraction[in_zone] = np.minimum(1.0, worked_fraction[in_zone] + (1 - worked_fraction[in_zone]) * 0.5)

        if model == "shen_atluri":
            disp = _shen_atluri_single_impact(p, node_coords, shot_xy)
        elif model == "sherafatnia":
            disp = _sherafatnia_single_impact(p, node_xy, shot_xy)
        else:
            raise ValueError(f"Unknown model '{model}'. Choose 'shen_atluri' or 'sherafatnia'.")

        total += disp

    return total


# ---------------------------------------------------------------------------
# 4.  Dataset comparison
# ---------------------------------------------------------------------------


def _read_simulation(sim_dir: str) -> dict:
    """Load ground-truth data from one simulation directory."""
    d = Path(sim_dir)
    node_coords = np.load(d / "node_coords.npy")
    displacements = np.load(d / "displacements.npy")
    shot_positions = np.load(d / "shot_positions.npy")

    # Parse V, D from simulation_params.txt
    params_path = d / "simulation_params.txt"
    V, D_mm, sigma_yield = 40.0, 0.6, 800e6
    E_b, nu_b, c = 110e9, 0.34, 1e9
    E_s, nu_s, rho_s = 200e9, 0.3, 7800.0

    if params_path.exists():
        with open(params_path) as fh:
            for line in fh:
                line = line.strip()
                if "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip()
                try:
                    val = float(v)
                    if k == "V_m_per_s":
                        V = val
                    elif k == "D_m":
                        D_mm = val * 1e3
                    elif k in ("sigma_yield", "sy"):
                        sigma_yield = val
                    elif k == "E_b":
                        E_b = val
                    elif k == "nu_b":
                        nu_b = val
                    elif k == "c":
                        c = val
                    elif k == "E_s":
                        E_s = val
                    elif k == "nu_s":
                        nu_s = val
                    elif k == "rho_s":
                        rho_s = val
                except ValueError:
                    pass

    params = ShotPeenParams(
        V=V,
        D=D_mm * 1e-3,
        E_b=E_b,
        nu_b=nu_b,
        sigma_yield=sigma_yield,
        c=c,
        E_s=E_s,
        nu_s=nu_s,
        rho_s=rho_s,
    )
    return {
        "node_coords": node_coords,
        "displacements": displacements,
        "shot_positions": shot_positions,
        "params": params,
        "V": V,
        "D_mm": D_mm,
    }


def _metrics(pred: np.ndarray, gt: np.ndarray, comp: int, frac: float = 0.05) -> dict:
    """Pearson r and RMSE for one displacement component."""
    p = pred[:, comp]
    g = gt[:, comp]
    thresh = max(np.abs(g).max() * frac, 1e-12)
    mask = np.abs(g) > thresh
    if mask.sum() < 2:
        return {"r": float("nan"), "rmse_um": float("nan"), "n": 0}
    r, _ = pearsonr(p[mask], g[mask])
    rmse = float(np.sqrt(np.mean((p[mask] - g[mask]) ** 2))) * 1e6
    return {"r": float(r), "rmse_um": rmse, "n": int(mask.sum())}


def compare_to_dataset(
    sim_dir: str,
    out_path: Optional[str] = None,
    sequential: bool = False,
) -> dict:
    """Run both analytical models on one simulation and compare to ground truth.

    Parameters
    ----------
    sim_dir   : Path to one Simulation_N/ directory.
    out_path  : If given, save a comparison figure here.
    sequential: Enable work-hardening sequential mode.

    Returns
    -------
    dict with keys: 'shen_atluri', 'sherafatnia', each containing metric dicts
    for 'ux', 'uy', 'uz'.
    """
    data = _read_simulation(sim_dir)
    nc = data["node_coords"]
    gt = data["displacements"]
    shots = data["shot_positions"]
    params = data["params"]

    pred_sa = predict_analytical(nc, shots, params, model="shen_atluri", sequential=sequential)
    pred_sh = predict_analytical(nc, shots, params, model="sherafatnia", sequential=sequential)

    results = {}
    for label, pred in [("shen_atluri", pred_sa), ("sherafatnia", pred_sh)]:
        results[label] = {}
        for i, comp in enumerate(("ux", "uy", "uz")):
            results[label][comp] = _metrics(pred, gt, i)

    if out_path:
        _plot_comparison(nc, gt, pred_sa, pred_sh, shots, data["V"], data["D_mm"], out_path)

    return results


def _plot_comparison(
    node_coords: np.ndarray,
    gt: np.ndarray,
    sa: np.ndarray,
    sh: np.ndarray,
    shot_positions: np.ndarray,
    V: float,
    D_mm: float,
    out_path: str,
) -> None:
    """Save a 3×3 figure: columns = GT / Shen-Atluri / Sherafatnia, rows = ux, uz, cross-section."""
    comps = [("ux", 0, "In-plane displacement $u_x$"), ("uz", 2, "Out-of-plane displacement $u_z$")]
    nx = np.unique(np.round(node_coords[:, 0], 10))
    ny = np.unique(np.round(node_coords[:, 1], 10))
    H, W = len(nx), len(ny)

    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    cmap = "RdBu_r"

    for row, (cname, ci, clabel) in enumerate(comps):
        data_sets = [
            ("Ground Truth\n(multi-shot sim)", gt[:, ci] * 1e6),
            ("Shen & Atluri\n(analytical)", sa[:, ci] * 1e6),
            ("Sherafatnia\n(elastic estimate)", sh[:, ci] * 1e6),
        ]
        vmax = max(np.abs(d).max() for _, d in data_sets)
        vmax = max(vmax, 0.01)

        for col, (title, vals) in enumerate(data_sets):
            ax = axes[row, col]
            try:
                grid = vals.reshape(H, W)
            except ValueError:
                grid = np.full((H, W), np.nan)
            im = ax.imshow(
                grid,
                origin="lower",
                aspect="equal",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            ax.scatter(
                shot_positions[:, 0] / (nx[1] - nx[0]),
                shot_positions[:, 1] / (ny[1] - ny[0]),
                s=3,
                c="k",
                alpha=0.15,
                linewidths=0,
            )
            ax.set_title(f"{title}\n{clabel}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("µm", fontsize=7)

    # Row 3: cross-section comparison along x-midline
    ax_cs = axes[2, :]
    mid_row = H // 2
    try:
        x_mm = nx * 1e3
        for ax, (cname, ci, clabel) in zip(ax_cs, comps[:2]):
            gt_line = gt[:, ci].reshape(H, W)[mid_row] * 1e6
            sa_line = sa[:, ci].reshape(H, W)[mid_row] * 1e6
            sh_line = sh[:, ci].reshape(H, W)[mid_row] * 1e6
            ax.plot(x_mm, gt_line, "k-", lw=2, label="Ground truth")
            ax.plot(x_mm, sa_line, "r--", lw=1.5, label="Shen & Atluri")
            ax.plot(x_mm, sh_line, "b:", lw=1.5, label="Sherafatnia")
            ax.set_xlabel("x (mm)", fontsize=8)
            ax.set_ylabel("µm", fontsize=8)
            ax.set_title(f"Cross-section — {clabel}", fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
        ax_cs[2].axis("off")
    except Exception:
        for ax in ax_cs:
            ax.axis("off")

    fig.suptitle(
        f"Analytical vs Ground Truth  |  V={V:.0f} m/s  D={D_mm:.1f} mm  " f"N_shots={len(shot_positions)}",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved -> {out_path}")


# ---------------------------------------------------------------------------
# 5.  Batch comparison over multiple simulations
# ---------------------------------------------------------------------------


def compare_dataset(
    dataset_dir: str,
    n_sims: int = 10,
    seed: int = 0,
    sequential: bool = False,
    out_csv: Optional[str] = None,
    verbose: bool = True,
) -> list:
    """Compare both models over N random simulations in a dataset.

    Returns a list of per-sim result dicts.
    """
    sims = sorted(
        [d for d in os.listdir(dataset_dir) if d.startswith("Simulation_")],
        key=lambda x: int(x.split("_")[1]),
    )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(sims, size=min(n_sims, len(sims)), replace=False)

    rows = []
    for sim_name in sorted(chosen):
        sim_dir = os.path.join(dataset_dir, sim_name)
        try:
            res = compare_to_dataset(sim_dir, sequential=sequential)
            row = {"sim": sim_name}
            for model in ("shen_atluri", "sherafatnia"):
                for comp in ("ux", "uy", "uz"):
                    m = res[model][comp]
                    row[f"{model}_{comp}_r"] = round(m["r"], 4) if not math.isnan(m["r"]) else float("nan")
                    row[f"{model}_{comp}_rmse"] = (
                        round(m["rmse_um"], 3) if not math.isnan(m["rmse_um"]) else float("nan")
                    )
            rows.append(row)
            if verbose:
                sa_r = res["shen_atluri"]["ux"]["r"]
                sh_r = res["sherafatnia"]["ux"]["r"]
                sa_uz = res["shen_atluri"]["uz"]["r"]
                sh_uz = res["sherafatnia"]["uz"]["r"]
                print(
                    f"  {sim_name:20s}  SA ux r={sa_r:+.3f}  Sh ux r={sh_r:+.3f}"
                    f"  |  SA uz r={sa_uz:+.3f}  Sh uz r={sh_uz:+.3f}"
                )
        except Exception as exc:
            print(f"  {sim_name}: SKIP — {exc}")

    if out_csv and rows:
        import csv

        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"CSV saved -> {out_csv}")

    # Summary statistics
    if rows and verbose:
        _print_summary(rows)

    return rows


def _print_summary(rows: list) -> None:
    """Print mean ± std for each model × component."""
    print("\n" + "=" * 70)
    print(f"{'':28s}  {'Shen-Atluri':^18s}  {'Sherafatnia':^18s}")
    print(f"{'Component':28s}  {'mean r':>8s}  {'RMSE µm':>8s}  {'mean r':>8s}  {'RMSE µm':>8s}")
    print("-" * 70)
    for comp in ("ux", "uy", "uz"):
        sa_r = [
            r[f"shen_atluri_{comp}_r"] for r in rows if not math.isnan(r.get(f"shen_atluri_{comp}_r", float("nan")))
        ]
        sa_rmse = [
            r[f"shen_atluri_{comp}_rmse"]
            for r in rows
            if not math.isnan(r.get(f"shen_atluri_{comp}_rmse", float("nan")))
        ]
        sh_r = [
            r[f"sherafatnia_{comp}_r"] for r in rows if not math.isnan(r.get(f"sherafatnia_{comp}_r", float("nan")))
        ]
        sh_rmse = [
            r[f"sherafatnia_{comp}_rmse"]
            for r in rows
            if not math.isnan(r.get(f"sherafatnia_{comp}_rmse", float("nan")))
        ]

        def fmt(vals):
            if not vals:
                return "  ---   ", "  ---  "
            return f"{np.mean(vals):+.3f}", f"{np.mean(vals):7.2f}"

        sa_r_s, sa_rm_s = fmt(sa_r)
        sh_r_s, sh_rm_s = fmt(sh_r)
        sa_rmse_s = f"{np.mean(sa_rmse):7.2f}" if sa_rmse else "  ---  "
        sh_rmse_s = f"{np.mean(sh_rmse):7.2f}" if sh_rmse else "  ---  "
        print(f"  {comp:26s}  {sa_r_s:>8s}  {sa_rmse_s:>8s}  {sh_r_s:>8s}  {sh_rmse_s:>8s}")
    print("=" * 70)
    print(
        "\nInterpretation:\n"
        "  Shen-Atluri r ~ 1.0  -> the simulator IS the Shen-Atluri model (expected).\n"
        "  Sherafatnia r < Shen-Atluri r -> elastic-only model misses plasticity.\n"
        "  RMSE difference -> magnitude error from using elastic vs. elastic-plastic.\n"
    )


# ---------------------------------------------------------------------------
# 6.  CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analytical shot peening displacement predictor — "
        "Shen-Atluri vs Sherafatnia vs dataset ground truth."
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dataset", help="Dataset directory (contains Simulation_N/ subdirs).")
    grp.add_argument("--shots", help="CSV file with columns: x_m, y_m [, t_s].")

    p.add_argument("--sim", type=int, default=None, help="Single simulation index (with --dataset).")
    p.add_argument("--n", type=int, default=10, help="Number of random sims to average (with --dataset).")
    p.add_argument("--sequential", action="store_true", help="Enable work-hardening sequential mode.")
    p.add_argument("--out", default=None, help="Output figure path (.png).")
    p.add_argument("--csv", default=None, help="Output CSV path for batch results.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for random sim selection.")

    # Parameters for --shots mode
    p.add_argument("--V", type=float, default=40.0, help="Impact velocity (m/s).")
    p.add_argument("--D", type=float, default=0.6, help="Shot diameter (mm).")
    p.add_argument("--material", default="Ti-6Al-4V", help="Workpiece material name.")
    p.add_argument("--shot-material", default="steel", help="Shot media material name.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    if args.dataset:
        dataset_dir = args.dataset
        if args.sim is not None:
            sim_dir = os.path.join(dataset_dir, f"Simulation_{args.sim}")
            print(f"\nComparing analytical models for {sim_dir}")
            out = args.out or os.path.join(dataset_dir, f"analytical_compare_sim{args.sim}.png")
            results = compare_to_dataset(sim_dir, out_path=out, sequential=args.sequential)
            print(f"\n{'Model':<14s}  {'comp':>4s}  {'r':>7s}  {'RMSE µm':>8s}  {'n nodes':>8s}")
            print("-" * 48)
            for model in ("shen_atluri", "sherafatnia"):
                for comp in ("ux", "uy", "uz"):
                    m = results[model][comp]
                    print(f"  {model:<14s}  {comp:>4s}  {m['r']:>+7.4f}  {m['rmse_um']:>8.2f}  {m['n']:>8d}")
        else:
            print(f"\nBatch comparison over {args.n} simulations in {dataset_dir}")
            out_csv = args.csv or os.path.join(dataset_dir, "analytical_compare.csv")
            compare_dataset(dataset_dir, n_sims=args.n, seed=args.seed, sequential=args.sequential, out_csv=out_csv)

    elif args.shots:
        # Load shot positions from CSV
        import csv as _csv

        with open(args.shots) as fh:
            reader = _csv.DictReader(fh)
            rows = list(reader)
        shots = np.array([[float(r["x_m"]), float(r["y_m"])] for r in rows])
        times = np.array([float(r["t_s"]) for r in rows]) if "t_s" in rows[0] else None

        # Build params from materials library if available
        try:
            import materials

            wp = materials.get_workpiece(args.material)
            sh_mat = materials.get_shot(args.shot_material)
            params = ShotPeenParams(
                V=args.V,
                D=args.D * 1e-3,
                E_b=wp["E"],
                nu_b=wp["nu"],
                sigma_yield=wp["sigma_yield"],
                c=wp["c"],
                E_s=sh_mat["E"],
                nu_s=sh_mat["nu"],
                rho_s=sh_mat["rho"],
            )
        except Exception:
            params = ShotPeenParams(V=args.V, D=args.D * 1e-3)

        # Build a simple flat mesh covering the shot area
        x_range = shots[:, 0].max() - shots[:, 0].min()
        y_range = shots[:, 1].max() - shots[:, 1].min()
        margin = max(params.R * 5, 0.002)
        Lx = x_range + 2 * margin
        Ly = y_range + 2 * margin
        N = 51
        xs = np.linspace(-Lx / 2, Lx / 2, N)
        ys = np.linspace(-Ly / 2, Ly / 2, N)
        Xg, Yg = np.meshgrid(xs, ys)
        nc = np.stack([Xg.ravel(), Yg.ravel(), np.zeros(N * N)], axis=1)

        print(f"\nRunning analytical models on {len(shots)} shots, {N}×{N} mesh ...")
        sa = predict_analytical(nc, shots, params, "shen_atluri", times, args.sequential)
        sh = predict_analytical(nc, shots, params, "sherafatnia", times, args.sequential)
        print(f"  Shen-Atluri   uz range: [{sa[:, 2].min() * 1e6:.1f}, {sa[:, 2].max() * 1e6:.1f}] um")
        print(f"  Sherafatnia   uz range: [{sh[:, 2].min() * 1e6:.1f}, {sh[:, 2].max() * 1e6:.1f}] um")

        if args.out:
            # Create a dummy GT (zeros) for the figure function
            gt_dummy = np.zeros_like(sa)
            _plot_comparison(nc, gt_dummy, sa, sh, shots, args.V, args.D, args.out)


if __name__ == "__main__":
    main()
