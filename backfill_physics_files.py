#!/usr/bin/env python3
"""
backfill_physics_files.py
=========================
Post-process an existing dataset directory to add the three new files
needed by MultiTaskPredictor:

    checkerboard_physics.npy   (6, G, G)  physics-rich sector encoding
    nodal_stresses.npy         (N_nodes, 4)  element stresses averaged to nodes
    cupping.npy                scalar (m)  Almen arc-height

Works by reading the already-saved files per simulation (shot_positions.npy,
stresses.npy, element_connectivity.npy, sR_depth_profile.npy) plus
simulation_params.txt for material properties. The per-shot V and D are
approximated as the simulation-level base values (V_scatter ±2 m/s is <5%
of the total range, so channels derived from V are accurate to within ~10%).

Usage:
    python backfill_physics_files.py --dataset LargeScaleRun1/Dataset_Ti_6Al_4V__steel_200
    python backfill_physics_files.py --dataset LargeScaleRun1/Dataset_Al_7075_T6__glass_2000 --workers 4
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src" / "peen-ml"
for _p in [str(_SRC), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _parse_params(params_path: str) -> dict:
    """Parse simulation_params.txt into a dict."""
    out = {}
    with open(params_path) as fh:
        for line in fh:
            line = line.strip()
            if ":" in line and not line.startswith("["):
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip()
            elif "=" in line and not line.startswith("["):
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
    return out


def _backfill_one(sim_dir: str, checkerboard_size: int = None) -> dict:
    """Backfill one simulation directory. Returns status dict."""
    sd = Path(sim_dir)
    t0 = time.perf_counter()

    try:
        # ---- Parse simulation params ----
        params_path = sd / "simulation_params.txt"
        if not params_path.exists():
            return {"sim": sd.name, "ok": False, "error": "no simulation_params.txt"}
        p = _parse_params(str(params_path))

        V = float(p.get("V_m_per_s", 40.0))
        D = float(p.get("D_mm", 0.6)) * 1e-3
        Lx = float(p.get("Lx_mm", 10.0)) * 1e-3
        Ly = float(p.get("Ly_mm", 10.0)) * 1e-3
        E_b = float(p.get("E_b", 1.138e11))
        nu_b = float(p.get("nu_b", 0.342))
        sy = float(p.get("sigma_yield", 8.8e8))
        c = float(p.get("c", 3.0e9))
        E_s = float(p.get("E_s", 2.1e11))
        nu_s = float(p.get("nu_s", 0.30))
        rho_s = float(p.get("rho_s", 7800.0))
        t_plate = 0.003  # 3 mm default

        # ---- Shot positions ----
        shot_path = sd / "shot_positions.npy"
        if not shot_path.exists():
            return {"sim": sd.name, "ok": False, "error": "no shot_positions.npy"}
        centres = np.load(str(shot_path))  # (M, 2)
        n_shots = len(centres)

        # ---- Compute per-shot physics (same V,D for all — V_scatter is small) ----
        from impact_sim import ShotPeenParams, compute_contact_params, compute_stress_field, compute_plastic_zone
        from multi_shot_sim import compute_influence_fields

        p_i = ShotPeenParams(
            E_s=E_s,
            nu_s=nu_s,
            D=D,
            rho_s=rho_s,
            E_b=E_b,
            nu_b=nu_b,
            sigma_yield=sy,
            c=c,
            V=V,
            n_depth=1000,
        )
        contact_i = compute_contact_params(p_i)
        plastic_i = compute_plastic_zone(p_i)
        sf_i = compute_stress_field(contact_i, p_i)

        a_p = plastic_i["a_p"]
        r_p = plastic_i["r_p"]
        R = p_i.R
        Vn = p_i.Vn
        delta_p = a_p**2 / (2.0 * R) if R > 0 else 0.0
        sigma_R_surface = float(sf_i["sR"][0]) if len(sf_i["sR"]) > 0 else 0.0
        sR_depth = np.stack([sf_i["Z"], sf_i["sR"]], axis=1)  # (K, 2)

        # Bending moment for this shot (same for all)
        if sR_depth.shape[0] > 1:
            z_arr, sR_arr = sR_depth[:, 0], sR_depth[:, 1]
            bm_per_shot = float(np.trapz(sR_arr * z_arr, z_arr))
        else:
            bm_per_shot = 0.0

        vol_shot = math.pi * D**3 / 6.0
        ke_per_shot = 0.5 * rho_s * Vn**2 * vol_shot

        # ---- Build physics checkerboard ----
        # Determine G (checkerboard grid size) from existing checkerboard.npy
        cb_path = sd / "checkerboard.npy"
        if cb_path.exists():
            G = np.load(str(cb_path)).shape[0]
        elif checkerboard_size is not None:
            G = checkerboard_size
        else:
            G = 10  # LargeScaleRun1 default

        cell_w = Lx / G
        cell_h = Ly / G
        A_cell = cell_w * cell_h

        n_shots_grid = np.zeros((G, G), dtype=np.float64)
        energy_grid = np.zeros((G, G), dtype=np.float64)
        dent_grid = np.zeros((G, G), dtype=np.float64)
        stress_grid = np.zeros((G, G), dtype=np.float64)
        rp2_grid = np.zeros((G, G), dtype=np.float64)
        bimoment_grid = np.zeros((G, G), dtype=np.float64)

        for c_xy in centres:
            col = min(G - 1, int(c_xy[0] / cell_w))
            row = min(G - 1, int(c_xy[1] / cell_h))
            n_shots_grid[row, col] += 1.0
            energy_grid[row, col] += ke_per_shot
            dent_grid[row, col] += delta_p
            stress_grid[row, col] += abs(sigma_R_surface)
            rp2_grid[row, col] += r_p**2
            bimoment_grid[row, col] += bm_per_shot

        coverage_grid = 1.0 - np.exp(-math.pi * rp2_grid / A_cell)
        energy_grid /= A_cell
        stress_grid /= A_cell
        bimoment_grid /= A_cell

        raw = np.stack([n_shots_grid, energy_grid, dent_grid, stress_grid, coverage_grid, bimoment_grid], axis=0)
        phys_cb = np.zeros_like(raw, dtype=np.float32)
        for ch in range(6):
            mn, mx = raw[ch].min(), raw[ch].max()
            if mx > mn:
                phys_cb[ch] = ((raw[ch] - mn) / (mx - mn)).astype(np.float32)

        np.save(str(sd / "checkerboard_physics.npy"), phys_cb)

        # ---- Node-resolution influence fields ----
        nc_path2 = sd / "node_coords.npy"
        if nc_path2.exists():
            nc_arr = np.load(str(nc_path2))
            Nx_inferred = len(np.unique(np.round(nc_arr[:, 0], 8))) - 1
            Ny_inferred = len(np.unique(np.round(nc_arr[:, 1], 8))) - 1
            inf_fields = compute_influence_fields(
                shot_positions=centres,
                node_coords=nc_arr,
                a_p=a_p,
                r_p=r_p,
                delta_p=delta_p,
                Nx=Nx_inferred,
                Ny=Ny_inferred,
            )
            np.save(str(sd / "influence_fields.npy"), inf_fields)

        # ---- Cupping from saved depth profile ----
        sR_path = sd / "sR_depth_profile.npy"
        if sR_path.exists():
            sR_mean = np.load(str(sR_path))  # (L, 2): [depth, sigma_R]
            if sR_mean.shape[0] > 1:
                z_arr = sR_mean[:, 0]
                sr_arr = sR_mean[:, 1]
                mask = z_arr <= t_plate
                if mask.sum() < 2:
                    mask = np.ones(len(z_arr), dtype=bool)
                M_b = float(np.trapz(sr_arr[mask] * z_arr[mask], z_arr[mask]))
                I_per_w = t_plate**3 / 12.0
                kappa = M_b / (E_b * I_per_w) if E_b > 0 else 0.0
                cupping = kappa * Lx**2 / 8.0
            else:
                cupping = 0.0
        else:
            cupping = 0.0
        np.save(str(sd / "cupping.npy"), np.float32(cupping))

        # ---- Nodal stresses from element stresses + connectivity ----
        stress_path = sd / "stresses.npy"
        conn_path = sd / "element_connectivity.npy"
        nc_path = sd / "node_coords.npy"
        if stress_path.exists() and conn_path.exists() and nc_path.exists():
            elem_stress = np.load(str(stress_path))  # (N_elems, 4)
            conn = np.load(str(conn_path))  # (N_elems, 4)
            n_nodes = len(np.load(str(nc_path)))  # use node_coords count

            n_comp = elem_stress.shape[1]
            accum = np.zeros((n_nodes, n_comp), dtype=np.float64)
            count = np.zeros(n_nodes, dtype=np.float64)
            for e_idx, nodes_row in enumerate(conn):
                for n_idx in nodes_row:
                    if 0 <= n_idx < n_nodes:
                        accum[n_idx] += elem_stress[e_idx]
                        count[n_idx] += 1.0
            nodal = np.zeros((n_nodes, n_comp), dtype=np.float32)
            safe = count > 0
            nodal[safe] = (accum[safe] / count[safe, None]).astype(np.float32)
            np.save(str(sd / "nodal_stresses.npy"), nodal)

        return {
            "sim": sd.name,
            "ok": True,
            "n_shots": n_shots,
            "G": G,
            "cupping_um": cupping * 1e6,
            "elapsed_s": time.perf_counter() - t0,
        }

    except Exception as exc:
        return {"sim": sd.name, "ok": False, "error": str(exc), "elapsed_s": time.perf_counter() - t0}


def backfill_dataset(dataset_dir: str, workers: int = 4, force: bool = False):
    ds = Path(dataset_dir)
    sim_dirs = sorted(
        [
            d
            for d in ds.iterdir()
            if d.is_dir() and d.name.startswith("Simulation_") and d.name[len("Simulation_") :].isdigit()
        ],
        key=lambda x: int(x.name.split("_")[1]),
    )
    total = len(sim_dirs)
    print(f"Backfilling {total} simulations in: {ds}")

    # Skip already-done unless forced.
    # Re-process if influence_fields.npy is missing even if other files exist.
    to_process = []
    skipped = 0
    for sd in sim_dirs:
        if not force:
            has_phys = (sd / "checkerboard_physics.npy").exists()
            has_inf = (sd / "influence_fields.npy").exists()
            if has_phys and has_inf:
                skipped += 1
                continue
        to_process.append(str(sd))

    if skipped:
        print(f"  Skipping {skipped} already backfilled (use --force to redo)")
    print(f"  Processing {len(to_process)} simulations  workers={workers}")

    if not to_process:
        print("  All done.")
        return

    t0 = time.perf_counter()
    n_ok = n_fail = 0

    if workers == 1:
        for i, sd in enumerate(to_process, 1):
            r = _backfill_one(sd)
            if r["ok"]:
                n_ok += 1
            else:
                n_fail += 1
                print(f"  FAIL {r['sim']}: {r.get('error','?')}")
            if i % max(1, len(to_process) // 10) == 0:
                print(f"  {i}/{len(to_process)}  ok={n_ok}  fail={n_fail}  " f"({time.perf_counter()-t0:.0f}s)")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_backfill_one, sd): sd for sd in to_process}
            for i, fut in enumerate(as_completed(futures), 1):
                r = fut.result()
                if r["ok"]:
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"  FAIL {r['sim']}: {r.get('error','?')}")
                if i % max(1, len(to_process) // 10) == 0:
                    print(f"  {i}/{len(to_process)}  ok={n_ok}  fail={n_fail}  " f"({time.perf_counter()-t0:.0f}s)")

    elapsed = time.perf_counter() - t0
    print(f"\nBackfill complete: {n_ok} OK  {n_fail} failed  ({elapsed:.1f}s total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset directory path")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="Re-compute even if files already exist")
    args = parser.parse_args()
    backfill_dataset(args.dataset, args.workers, args.force)
