#!/usr/bin/env python3
"""
run_eval.py  -  Evaluate all LargeScaleRun1 models and write a summary.

Usage:
    python run_eval.py                        # evaluates every model in LargeScaleRun1/Models/
    python run_eval.py --run LargeScaleRun2   # different run directory
    python run_eval.py --models G H           # subset of model names
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE / "src" / "peen-ml"
for _p in [str(_SRC), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gc

import numpy as np
import torch
import model as M


def _flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model catalogue: (model_dir_name, pth_filename, dataset_dir, label)
# ---------------------------------------------------------------------------

DEFAULT_RUN = "LargeScaleRun1"

MODEL_CATALOGUE = [
    # name                              pth_file                                         dataset                                label
    ("A_DisplPredictor_Ti_Steel",       "trained_displacement_predictor_full_model.pth", "Dataset_Ti_6Al_4V__steel_200",        "A - Std Ti+steel 200"),
    ("B_ImprovedPredictor_Ti_Steel",    "trained_displacement_predictor_full_model.pth", "Dataset_Ti_6Al_4V__steel_200",        "B - Improved Ti+steel 200"),
    ("C_MatCond_MultiMat",              "trained_displacement_predictor_full_model.pth", "Dataset_MultiMat_Merged",             "C - MatCond MultiMat 5000"),
    ("D_Improved_Ti_6Al_4V__steel",     "trained_displacement_predictor_full_model.pth", "Dataset_Ti_6Al_4V__steel_200",        "D - Improved Ti+steel 200"),
    ("D_Improved_316L_SS__ceramic",     "trained_displacement_predictor_full_model.pth", "Dataset_316L_SS__ceramic_200",        "D - Improved 316L+ceramic 200"),
    ("D_Improved_Inconel_718__tungsten","trained_displacement_predictor_full_model.pth", "Dataset_Inconel_718__tungsten_200",   "D - Improved Inconel+tungsten 200"),
    ("D_Improved_Al_7075_T6__glass",    "trained_displacement_predictor_full_model.pth", "Dataset_Al_7075_T6__glass_200",       "D - Improved Al+glass 200"),
    ("D_Improved_4340_Steel__cast_iron","trained_displacement_predictor_full_model.pth", "Dataset_4340_Steel__cast_iron_200",   "D - Improved 4340+cast_iron 200"),
    ("E_ConvDecoder_HighRes",            "trained_conv_decoder_full_model.pth",          "Dataset_HighRes_Ti_Steel_300",        "E - ConvDecoder HighRes 300"),
    ("G_Improved_Al7075_glass_2000",    "model.pth",                                    "Dataset_Al_7075_T6__glass_2000",      "G - Improved Al+glass 2000 (own)"),
    ("H_Improved_Ti_steel_2000",        "model.pth",                                    "Dataset_Ti_6Al_4V__steel_2000",       "H - Improved Ti+steel 2000 (own)"),
    ("MT_MultiTask_Ti_Steel",           "trained_multitask_model.pth",                  "Dataset_Ti_6Al_4V__steel_200",        "MT - MultiTask Ti+steel 200"),
    ("I_InfluenceField_Ti_Steel",                "influence_field_model.pth",  "Dataset_Ti_6Al_4V__steel_200",        "I - InfluenceField Ti+steel 200"),
    ("I_InfluenceField_316L_ceramic",           "influence_field_model.pth",  "Dataset_316L_SS__ceramic_200",        "I - InfluenceField 316L+ceramic 200"),
    ("I_InfluenceField_Inconel_tungsten",        "influence_field_model.pth",  "Dataset_Inconel_718__tungsten_200",   "I - InfluenceField Inconel+tungsten 200"),
    ("I_InfluenceField_Al_glass",               "influence_field_model.pth",  "Dataset_Al_7075_T6__glass_200",       "I - InfluenceField Al+glass 200"),
    ("I_InfluenceField_4340_cast_iron",         "influence_field_model.pth",  "Dataset_4340_Steel__cast_iron_200",   "I - InfluenceField 4340+cast_iron 200"),
    # J: InfluenceField on 2000-sim (data-scaling ablation)
    ("J_InfluenceField_Ti_steel_2000",           "influence_field_model.pth",  "Dataset_Ti_6Al_4V__steel_2000",       "J - InfluenceField Ti+steel 2000"),
    # K: InfluenceField on HighRes (resolution-scaling ablation)
    ("K_InfluenceField_HighRes",                 "influence_field_model.pth",  "Dataset_HighRes_Ti_Steel_300",        "K - InfluenceField HighRes 300"),
]

# Cross-evaluations: model evaluated on a dataset it was NOT trained on
CROSS_EVALS = [
    # (model_dir, pth, eval_dataset, label)
    ("G_Improved_Al7075_glass_2000", "model.pth", "Dataset_Ti_6Al_4V__steel_2000",  "G - Improved Al+glass on Ti+steel (unseen)"),
    ("H_Improved_Ti_steel_2000",     "model.pth", "Dataset_Al_7075_T6__glass_2000",  "H - Improved Ti+steel on Al+glass (unseen)"),
]


def evaluate_multitask(
    model_path: str,
    dataset_dir: str,
    components: List[str],
    label: str,
    save_plot: Optional[str] = None,
) -> dict:
    """Evaluate a MultiTaskPredictor on displacement metrics.

    Delegates to M.evaluate_on_dataset (which now handles dict output) for
    displacement, then optionally computes cupping scatter via
    M.evaluate_cupping_on_dataset if cupping.npy files are present.

    Returns the same dict shape as eval_model so it can be appended to rows.
    """
    row = eval_model(model_path, dataset_dir, components, label, save_plot=save_plot)
    # Detect whether this dataset has cupping files for the MT bonus metric
    cup_count = sum(
        1 for d in os.listdir(dataset_dir)
        if d.startswith("Simulation_") and
        os.path.exists(os.path.join(dataset_dir, d, "cupping.npy"))
    ) if os.path.isdir(dataset_dir) else 0
    if cup_count >= 3:
        cup_plot = save_plot.replace("_ux.png", "_cupping.png") if save_plot else None
        try:
            cup_res = M.evaluate_cupping_on_dataset(model_path, dataset_dir, cup_plot)
            row["cupping_r"]    = round(cup_res["pearson_r"], 4)
            row["cupping_rmse"] = round(cup_res["rmse_um"], 3)
            row["cupping_n"]    = cup_res["n_ok"]
        except Exception as exc:
            print(f"  Cupping evaluation skipped: {exc}")
    return row


def count_sims(dataset_dir: str) -> int:
    if not os.path.isdir(dataset_dir):
        return 0
    return sum(
        1 for d in os.listdir(dataset_dir)
        if d.startswith("Simulation_") and
        os.path.exists(os.path.join(dataset_dir, d, "displacements.npy"))
    )


def eval_model(
    model_path: str,
    dataset_dir: str,
    components: List[str],
    label: str,
    max_sims: Optional[int] = None,
    save_plot: Optional[str] = None,
) -> dict:
    """Evaluate one model on one dataset for each displacement component."""
    n_sims = count_sims(dataset_dir)
    row = {"label": label, "model_path": model_path,
           "dataset": os.path.basename(dataset_dir), "n_sims": n_sims}

    for comp in components:
        plot = None
        if save_plot and comp == "ux":
            plot = save_plot

        # subsample for speed on large datasets
        # (evaluate_on_dataset always uses all sims; we cap by limiting sims available)
        try:
            res = M.evaluate_on_dataset(
                model_path     = model_path,
                data_path      = dataset_dir,
                component      = comp,
                threshold_frac = 0.05,
                plot_save_path = plot,
            )
            row[f"{comp}_r"]            = round(res["mean_r"],            4)
            row[f"{comp}_rmse_um"]      = round(res["mean_rmse_um"],      3)
            row[f"{comp}_rel_rmse_pct"] = round(res["mean_rel_rmse_pct"], 1)
            row[f"{comp}_n_ok"]         = res["n_ok"]
        except Exception as exc:
            print(f"  ERROR evaluating {comp}: {exc}")
            row[f"{comp}_r"]            = float("nan")
            row[f"{comp}_rmse_um"]      = float("nan")
            row[f"{comp}_rel_rmse_pct"] = float("nan")
            row[f"{comp}_n_ok"]         = 0
        finally:
            _flush_gpu()

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",    default=DEFAULT_RUN,
                        help="Run directory (default: LargeScaleRun1)")
    parser.add_argument("--models", nargs="*",
                        help="Subset of model name prefixes to evaluate (e.g. G H MT)")
    parser.add_argument("--components", nargs="*", default=["ux", "uy", "uz"],
                        help="Displacement components to evaluate (default: ux uy uz)")
    parser.add_argument("--no-cross", action="store_true",
                        help="Skip cross-dataset evaluations")
    parser.add_argument("--cupping", action="store_true",
                        help="Run standalone cupping scatter for MT model and save PNG")
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    models_dir = run_dir / "Models"
    components = args.components

    print(f"\n{'='*72}")
    print(f"  Evaluation run: {run_dir}")
    print(f"  Components    : {components}")
    print(f"{'='*72}\n")

    catalogue = MODEL_CATALOGUE
    if args.models:
        catalogue = [e for e in catalogue
                     if any(e[0].startswith(m) for m in args.models)]

    rows = []

    # ---- Standard evaluations ----
    for (model_dir_name, pth_file, dataset_rel, label) in catalogue:
        model_path   = str(models_dir / model_dir_name / pth_file)
        dataset_path = str(run_dir / dataset_rel)

        if not os.path.exists(model_path):
            print(f"SKIP (no model): {label}")
            print(f"  expected: {model_path}")
            continue
        if not os.path.isdir(dataset_path):
            print(f"SKIP (no dataset): {label}")
            print(f"  expected: {dataset_path}")
            continue

        print(f"\n--- {label} ---")
        plot_path = str(run_dir / f"eval_{model_dir_name}_ux.png")
        is_mt = model_dir_name.startswith("MT_")
        if is_mt:
            row = evaluate_multitask(model_path, dataset_path, components, label,
                                     save_plot=plot_path)
        else:
            row = eval_model(model_path, dataset_path, components, label,
                             save_plot=plot_path)
        rows.append(row)
        _flush_gpu()
        cup_str = f"  cupping r={row['cupping_r']:.4f}" if "cupping_r" in row else ""
        print(f"  ux r={row.get('ux_r','?'):.4f}  "
              f"uy r={row.get('uy_r','?'):.4f}  "
              f"uz r={row.get('uz_r','?'):.4f}  "
              f"(n_ok={row.get('ux_n_ok',0)}){cup_str}")

    # ---- Cross-dataset evaluations ----
    if not args.no_cross:
        print("\n\n--- Cross-dataset evaluations ---")
        cross_catalogue = CROSS_EVALS
        if args.models:
            cross_catalogue = [e for e in cross_catalogue
                               if any(e[0].startswith(m) for m in args.models)]

        for (model_dir_name, pth_file, dataset_rel, label) in cross_catalogue:
            model_path   = str(models_dir / model_dir_name / pth_file)
            dataset_path = str(run_dir / dataset_rel)

            if not os.path.exists(model_path):
                print(f"SKIP (no model): {label}")
                continue
            if not os.path.isdir(dataset_path):
                print(f"SKIP (no dataset): {label}")
                continue

            print(f"\n--- {label} ---")
            plot_path = str(run_dir / f"eval_cross_{model_dir_name}_ux.png")
            row = eval_model(model_path, dataset_path, components, label,
                             save_plot=plot_path)
            rows.append(row)
            _flush_gpu()
            print(f"  ux r={row.get('ux_r','?'):.4f}  "
                  f"uy r={row.get('uy_r','?'):.4f}  "
                  f"uz r={row.get('uz_r','?'):.4f}  "
                  f"(n_ok={row.get('ux_n_ok',0)})")

    # ---- Standalone cupping scatter (--cupping flag) ----
    if args.cupping:
        mt_model_path = str(models_dir / "MT_MultiTask_Ti_Steel" / "trained_multitask_model.pth")
        mt_dataset    = str(run_dir / "Dataset_Ti_6Al_4V__steel_200")
        cup_save      = str(run_dir / "cupping_validation_scatter.png")
        if os.path.exists(mt_model_path) and os.path.isdir(mt_dataset):
            print("\n--- Cupping (Almen arc-height) validation scatter ---")
            try:
                cup_res = M.evaluate_cupping_on_dataset(mt_model_path, mt_dataset, cup_save)
                print(f"  r={cup_res['pearson_r']:.4f}  RMSE={cup_res['rmse_um']:.3f} µm  "
                      f"n={cup_res['n_ok']}")
            except Exception as exc:
                print(f"  Cupping scatter failed: {exc}")
        else:
            print("\nSKIP cupping scatter — MT model or dataset not found")
            print(f"  expected model : {mt_model_path}")
            print(f"  expected dataset: {mt_dataset}")

    # ---- Write CSV ----
    out_csv = str(run_dir / "eval_results.csv")
    fieldnames = ["label", "dataset", "n_sims",
                  "ux_r", "ux_rmse_um", "ux_rel_rmse_pct", "ux_n_ok",
                  "uy_r", "uy_rmse_um", "uy_rel_rmse_pct", "uy_n_ok",
                  "uz_r", "uz_rmse_um", "uz_rel_rmse_pct", "uz_n_ok",
                  "cupping_r", "cupping_rmse", "cupping_n",
                  "model_path"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nResults written to: {out_csv}")

    # ---- Console summary ----
    print(f"\n{'='*90}")
    print(f"{'Label':<48} {'ux r':>7} {'ux rel%':>8} {'uy r':>7} {'uz r':>7} {'uz rel%':>8}")
    print(f"  note: r = pattern correlation; rel RMSE% = RMSE / peak_gt (scale accuracy)")
    print(f"{'-'*90}")
    for r in rows:
        print(f"{r['label']:<48} "
              f"{r.get('ux_r', float('nan')):>7.4f} "
              f"{r.get('ux_rel_rmse_pct', float('nan')):>7.1f}% "
              f"{r.get('uy_r', float('nan')):>7.4f} "
              f"{r.get('uz_r', float('nan')):>7.4f} "
              f"{r.get('uz_rel_rmse_pct', float('nan')):>7.1f}%")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
