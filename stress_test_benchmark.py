"""
stress_test_benchmark.py
========================
Comprehensive stress-test and benchmark for the peen-ml pipeline.

Generates 30 datasets (25 material combos + 5 special high-res / extreme
configs), trains DisplacementPredictor on each, and produces a CSV report.

Each dataset uses a different workpiece + shot material combination drawn from
the materials.py library.  Five additional configs exercise high-resolution
meshes and extreme peening conditions.

Usage
-----
    python stress_test_benchmark.py                            # defaults
    python stress_test_benchmark.py --n_sims 40 --epochs 30   # more data
    python stress_test_benchmark.py --output ./MyBench --quick # 5 sims/config
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup — works whether run from repo root or any other CWD
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src" / "peen-ml"
for _p in [str(_SRC), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import model as M  # noqa: E402
from native_dataset_gen import GeneratorParams, generate_single_simulation
from materials import WORKPIECE_MATERIALS, SHOT_MATERIALS


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    """One benchmark run specification."""

    label: str
    workpiece: str
    shot: str
    n_sims: int
    Nx: int
    Ny: int
    checkerboard_size: int
    V_range: tuple
    D_range: tuple
    n_shots_range: tuple
    description: str = ""


def build_configs(n_sims: int) -> List[BenchConfig]:
    """Return the full list of 30 benchmark configurations."""
    configs: List[BenchConfig] = []

    # ---- 25 material combos (standard resolution) ----
    workpieces = sorted(WORKPIECE_MATERIALS.keys())
    shots = sorted(SHOT_MATERIALS.keys())
    for wp in workpieces:
        for sp in shots:
            configs.append(
                BenchConfig(
                    label=f"{wp}__{sp}".replace("-", "_").replace(" ", "_"),
                    workpiece=wp,
                    shot=sp,
                    n_sims=n_sims,
                    Nx=50,
                    Ny=50,
                    checkerboard_size=10,
                    V_range=(25.0, 60.0),
                    D_range=(0.0003, 0.0010),
                    n_shots_range=(30, 150),
                    description=f"Standard: {wp} + {sp}",
                )
            )

    # ---- 5 special configs ----
    special = [
        BenchConfig(
            label="Ti6Al4V__steel__highres",
            workpiece="Ti-6Al-4V",
            shot="steel",
            n_sims=n_sims,
            Nx=70,
            Ny=70,
            checkerboard_size=10,
            V_range=(25.0, 60.0),
            D_range=(0.0003, 0.0010),
            n_shots_range=(30, 150),
            description="High-resolution mesh (71×71=5041 nodes)",
        ),
        BenchConfig(
            label="Al7075__tungsten__extreme_V",
            workpiece="Al-7075-T6",
            shot="tungsten",
            n_sims=n_sims,
            Nx=50,
            Ny=50,
            checkerboard_size=10,
            V_range=(10.0, 80.0),
            D_range=(0.0003, 0.0010),
            n_shots_range=(30, 150),
            description="Wide velocity range (Al + tungsten)",
        ),
        BenchConfig(
            label="Inconel718__ceramic__dense_cb",
            workpiece="Inconel-718",
            shot="ceramic",
            n_sims=n_sims,
            Nx=50,
            Ny=50,
            checkerboard_size=15,
            V_range=(25.0, 60.0),
            D_range=(0.0003, 0.0010),
            n_shots_range=(30, 150),
            description="Dense checkerboard G=15 (Inconel + ceramic)",
        ),
        BenchConfig(
            label="316LSS__glass__many_shots",
            workpiece="316L-SS",
            shot="glass",
            n_sims=n_sims,
            Nx=50,
            Ny=50,
            checkerboard_size=10,
            V_range=(25.0, 60.0),
            D_range=(0.0003, 0.0010),
            n_shots_range=(100, 400),
            description="High shot count (316L-SS + glass)",
        ),
        BenchConfig(
            label="4340Steel__cast_iron__small_D",
            workpiece="4340-Steel",
            shot="cast_iron",
            n_sims=n_sims,
            Nx=50,
            Ny=50,
            checkerboard_size=10,
            V_range=(25.0, 60.0),
            D_range=(0.0001, 0.0004),
            n_shots_range=(30, 150),
            description="Small diameter range (4340-Steel + cast_iron)",
        ),
    ]
    configs.extend(special)

    return configs


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(cfg: BenchConfig, output_root: str) -> dict:
    """Generate one dataset for *cfg* inside *output_root/cfg.label/*.

    Returns a dict with keys: success, n_sims_ok, n_sims_fail, elapsed_s, error.
    Skips already-existing simulation folders so the script is resumable.
    """
    dataset_dir = os.path.join(output_root, cfg.label)
    os.makedirs(dataset_dir, exist_ok=True)

    gp = GeneratorParams(
        output_dir=dataset_dir,
        n_simulations=cfg.n_sims,
        start_index=0,
        workers=1,  # Windows-safe sequential generation
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        checkerboard_size=cfg.checkerboard_size,
        V_range=cfg.V_range,
        D_range=cfg.D_range,
        n_shots_range=cfg.n_shots_range,
        workpiece_material=cfg.workpiece,
        shot_material=cfg.shot,
        vary_distribution=True,
        base_seed=abs(hash(cfg.label)) % (2**31),
    )

    t0 = time.perf_counter()
    n_ok = n_fail = 0
    errors = []

    for i in range(cfg.n_sims):
        sim_dir = os.path.join(dataset_dir, f"Simulation_{i}")
        # Skip if already complete (resumable runs)
        if os.path.exists(os.path.join(sim_dir, "displacements.npy")):
            n_ok += 1
            continue
        res = generate_single_simulation(i, gp)
        if res["success"]:
            n_ok += 1
        else:
            n_fail += 1
            errors.append(f"Sim_{i}: {res['error']}")

    elapsed = time.perf_counter() - t0
    success = n_fail == 0 and n_ok > 0
    return {
        "success": success,
        "n_sims_ok": n_ok,
        "n_sims_fail": n_fail,
        "elapsed_s": elapsed,
        "error": "; ".join(errors) if errors else None,
        "dataset_dir": dataset_dir,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    mse: float = float("nan")
    smape_pct: float = float("nan")
    rmse_um: float = float("nan")
    epochs_trained: int = 0
    train_time_s: float = float("nan")
    success: bool = False
    error: Optional[str] = None


def train_and_evaluate(
    dataset_dir: str,
    epochs: int = 30,
    patience: int = 8,
    batch_size: int = 12,
) -> TrainResult:
    """Load *dataset_dir*, train DisplacementPredictor, return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    try:
        num_nodes, cb_size = M.infer_dataset_shape(dataset_dir)

        train_loader, val_loader, test_loader, loaded_data = M.create_data_loaders(
            base_folder=dataset_dir,
            load_files=("checkerboard", "displacements"),
            batch_size=batch_size,
            load_material_features=False,
        )

        # Guard against empty val/test loaders (degenerate split)
        if len(val_loader) == 0:
            return TrainResult(
                success=False,
                error=f"Val split is empty (n_sims too small for batch_size={batch_size}). "
                "Increase n_sims to at least 7.",
            )

        model = M.create_model(
            input_channels=1,
            num_nodes=num_nodes,
            checkerboard_size=cb_size,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        train_losses, val_losses = M.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            patience=patience,
            device=device,
            plot_save_path=None,  # skip PNG in benchmark mode
        )

        mse = M.evaluate_model(model, test_loader, criterion, device=device)
        smape = float("nan")
        rmse = float("nan")
        if not (mse != mse):  # not NaN
            # Collect all test predictions for RMSE (physical units)
            all_pred, all_true = [], []
            model.eval()
            with torch.no_grad():
                for cb, disp in test_loader:
                    pred = model(cb.to(device))
                    all_pred.append(pred.cpu().numpy())
                    all_true.append(disp.numpy())
            pred_np = np.concatenate(all_pred)
            true_np = np.concatenate(all_true)
            rmse = float(np.sqrt(np.mean((pred_np - true_np) ** 2))) * 1e6  # µm
            # sMAPE — guard against zero denominator
            denom = (np.abs(true_np) + np.abs(pred_np)) / 2.0
            safe = denom > 1e-15
            if safe.any():
                smape = float(np.mean(np.abs(true_np[safe] - pred_np[safe]) / denom[safe])) * 100

        return TrainResult(
            mse=float(mse),
            smape_pct=smape,
            rmse_um=rmse,
            epochs_trained=len(train_losses),
            train_time_s=time.perf_counter() - t0,
            success=True,
        )

    except Exception as exc:  # noqa: BLE001
        return TrainResult(
            success=False,
            train_time_s=time.perf_counter() - t0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(rows: list, output_path: str) -> None:
    fieldnames = [
        "label",
        "workpiece",
        "shot",
        "description",
        "n_sims_ok",
        "n_sims_fail",
        "Nx",
        "Ny",
        "cb_size",
        "gen_ok",
        "gen_s",
        "train_ok",
        "mse",
        "smape_pct",
        "rmse_um",
        "epochs_trained",
        "train_s",
        "error",
    ]
    with open(output_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\nBenchmark report written to: {output_path}")


def print_summary(rows: list) -> None:
    ok = [r for r in rows if r.get("train_ok")]
    fail = [r for r in rows if not r.get("train_ok")]

    print("\n" + "=" * 70)
    print(f"BENCHMARK SUMMARY  ({len(rows)} configs)")
    print("=" * 70)
    print(f"  Successful   : {len(ok)}")
    print(f"  Failed       : {len(fail)}")

    if ok:

        def _floats(key):
            vals = []
            for r in ok:
                try:
                    v = float(r[key])
                    if v == v:  # not NaN
                        vals.append(v)
                except (ValueError, TypeError):
                    pass
            return vals

        mses = _floats("mse")
        rmses = _floats("rmse_um")
        smapes = _floats("smape_pct")
        if mses:
            print(f"\n  MSE   -- min={min(mses):.3e}  max={max(mses):.3e}  " f"mean={sum(mses)/len(mses):.3e}")
        if rmses:
            print(
                f"  RMSE  -- min={min(rmses):.2f} um  max={max(rmses):.2f} um  " f"mean={sum(rmses)/len(rmses):.2f} um"
            )
        if smapes:
            print(
                f"  sMAPE -- min={min(smapes):.1f}%   max={max(smapes):.1f}%   " f"mean={sum(smapes)/len(smapes):.1f}%"
            )

    if fail:
        print("\n  Failed configs:")
        for r in fail:
            print(f"    {r['label']}: {r.get('error', 'unknown')}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Holes / anomaly detection
# ---------------------------------------------------------------------------


def detect_holes(rows: list) -> None:
    """Print any anomalies detected across the benchmark results."""
    print("\n--- Anomaly / Hole Detection ---")
    holes_found = 0

    for r in rows:
        label = r["label"]

        if not r.get("gen_ok"):
            print(f"  [HOLE] {label}: dataset generation failed — {r.get('error')}")
            holes_found += 1
            continue

        if not r.get("train_ok"):
            err = r.get("error", "")
            if "ZeroDivisionError" in err or "empty" in err.lower():
                print(
                    f"  [HOLE] {label}: empty val/test split — too few sims for "
                    "70/15/15 ratio.  Minimum n_sims=7 needed."
                )
            else:
                print(f"  [HOLE] {label}: training failed — {err}")
            holes_found += 1
            continue

        # Accuracy anomalies — values are stored as strings in rows
        def _safe_float(key):
            try:
                return float(r.get(key, "nan"))
            except (ValueError, TypeError):
                return float("nan")

        rmse = _safe_float("rmse_um")
        smape = _safe_float("smape_pct")
        mse = _safe_float("mse")

        if rmse == rmse and rmse > 500:
            print(
                f"  [WARN] {label}: RMSE={rmse:.0f} um -- suspiciously large "
                "(physical range ~1-50 um). Check dataset or training duration."
            )
            holes_found += 1

        if smape != smape:
            print(
                f"  [HOLE] {label}: sMAPE is NaN — likely zero displacement "
                "denominator. smape() in model.py lacks a zero-guard."
            )
            holes_found += 1
        elif smape > 150:
            print(
                f"  [WARN] {label}: sMAPE={smape:.0f}% -- near-random predictions "
                "(ground truth may be near-zero, triggering sMAPE instability)"
            )
            holes_found += 1

        if mse != mse:
            print(f"  [HOLE] {label}: MSE is NaN — NaN propagating through model")
            holes_found += 1

    if holes_found == 0:
        print("  No anomalies detected.")
    else:
        print(f"\n  Total anomalies: {holes_found}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="peen-ml stress-test benchmark — 30 material / condition configs.")
    parser.add_argument("--output", default="./BenchmarkResults", help="Root output directory for datasets and report")
    parser.add_argument("--n_sims", type=int, default=30, help="Simulations per config (default 30)")
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs per config (default 25)")
    parser.add_argument("--patience", type=int, default=7, help="Early-stopping patience (default 7)")
    parser.add_argument("--batch", type=int, default=10, help="Training batch size (default 10)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 8 sims, 10 epochs")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation (datasets already exist)")
    parser.add_argument("--label", default=None, help="Run only configs whose label contains this string")
    args = parser.parse_args()

    if args.quick:
        args.n_sims = 8
        args.epochs = 10
        args.patience = 4

    os.makedirs(args.output, exist_ok=True)

    configs = build_configs(args.n_sims)
    if args.label:
        configs = [c for c in configs if args.label in c.label]
        if not configs:
            print(f"No configs match label filter '{args.label}'. Exiting.")
            return

    print("=" * 70)
    print("peen-ml Stress-Test Benchmark")
    print(f"  Configs  : {len(configs)}")
    print(f"  Sims/cfg : {args.n_sims}")
    print(f"  Epochs   : {args.epochs}  (patience={args.patience})")
    print(f"  Output   : {args.output}")
    print(f"  Device   : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    rows = []

    for ci, cfg in enumerate(configs, 1):
        print(f"\n[{ci:2d}/{len(configs)}] {cfg.label}")
        print(f"         {cfg.description}")

        row: dict = {
            "label": cfg.label,
            "workpiece": cfg.workpiece,
            "shot": cfg.shot,
            "description": cfg.description,
            "Nx": cfg.Nx,
            "Ny": cfg.Ny,
            "cb_size": cfg.checkerboard_size,
        }

        # ---- Generation ----
        if args.skip_gen:
            gen_res = {
                "success": True,
                "n_sims_ok": cfg.n_sims,
                "n_sims_fail": 0,
                "elapsed_s": 0.0,
                "error": None,
                "dataset_dir": os.path.join(args.output, cfg.label),
            }
        else:
            print(f"         Generating {cfg.n_sims} simulations...")
            gen_res = generate_dataset(cfg, args.output)

        row.update(
            {
                "n_sims_ok": gen_res["n_sims_ok"],
                "n_sims_fail": gen_res["n_sims_fail"],
                "gen_ok": gen_res["success"],
                "gen_s": f"{gen_res['elapsed_s']:.1f}",
            }
        )

        if not gen_res["success"]:
            row["train_ok"] = False
            row["error"] = gen_res.get("error", "generation failed")
            print(f"         GENERATION FAILED: {row['error']}")
            rows.append(row)
            continue

        ok_str = f"OK ({gen_res['n_sims_ok']}/{cfg.n_sims} sims)"
        print(f"         Generation {ok_str}  {gen_res['elapsed_s']:.1f}s")

        # ---- Training ----
        print(f"         Training (max {args.epochs} epochs, patience={args.patience})...")
        tr = train_and_evaluate(
            dataset_dir=gen_res["dataset_dir"],
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch,
        )

        row.update(
            {
                "train_ok": tr.success,
                "mse": f"{tr.mse:.4e}" if tr.success else "nan",
                "smape_pct": f"{tr.smape_pct:.2f}" if tr.success else "nan",
                "rmse_um": f"{tr.rmse_um:.3f}" if tr.success else "nan",
                "epochs_trained": tr.epochs_trained,
                "train_s": f"{tr.train_time_s:.1f}",
                "error": tr.error or "",
            }
        )

        if tr.success:
            print(
                f"         MSE={tr.mse:.3e}  RMSE={tr.rmse_um:.2f}um  "
                f"sMAPE={tr.smape_pct:.1f}%  epochs={tr.epochs_trained}  "
                f"time={tr.train_time_s:.0f}s"
            )
        else:
            print(f"         TRAINING FAILED: {tr.error}")

        rows.append(row)

    # ---- Write report ----
    report_path = os.path.join(args.output, "benchmark_report.csv")
    write_report(rows, report_path)
    print_summary(rows)
    detect_holes(rows)


if __name__ == "__main__":
    main()
