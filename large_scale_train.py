#!/usr/bin/env python3
"""
large_scale_train.py
====================
Large-scale dataset generation and model training for peen-ml.

Improvements over the basic benchmark:
  - 200+ sims per config (vs 30 in the benchmark)
  - High-resolution meshes (Nx=Ny=100, 10201 nodes)
  - Wider physics ranges (V, D, n_shots) for better generalisation
  - Material-conditioned ImprovedDisplacementPredictor on 5000-sim combined dataset
  - ConvDecoder on high-res (101x101 grid) data
  - AdamW + CosineAnnealing + gradient clipping + AMP on RTX 4090
  - Parallel dataset generation (CPU workers)

Usage
-----
    python large_scale_train.py                              # full run (~60 min)
    python large_scale_train.py --quick                      # fast test (~10 min)
    python large_scale_train.py --phase generate             # generation only
    python large_scale_train.py --phase train --skip-gen     # training only
    python large_scale_train.py --output MyRun --n_sims 300  # custom
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import model as M  # noqa: E402
from native_dataset_gen import GeneratorParams, generate_single_simulation
from materials import WORKPIECE_MATERIALS, SHOT_MATERIALS


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    """One dataset generation spec."""

    name: str
    workpiece: str
    shot: str
    n_sims: int
    Nx: int = 50
    Ny: int = 50
    G: int = 10
    V_range: tuple = (10.0, 80.0)  # wider than benchmark
    D_range: tuple = (0.0001, 0.0015)  # wider than benchmark
    n_shots_range: tuple = (20, 400)  # wider than benchmark
    workers: int = 4  # parallel gen workers
    description: str = ""


def build_dataset_specs(
    n_sims_standard: int = 200, n_sims_hires: int = 300, workers: int = 4
) -> Dict[str, List[DatasetSpec]]:
    """Return grouped dataset specs.

    Groups
    ------
    'per_material'   : 5 key combos, large standard-res datasets
    'multi_material' : all 25 combos, used for merging into a combined set
    'high_res'       : Ti+steel at Nx=100 for ConvDecoder training
    """
    specs: Dict[str, List[DatasetSpec]] = {
        "per_material": [],
        "multi_material": [],
        "high_res": [],
    }

    # ---- 5 representative per-material large datasets ----
    key_combos = [
        ("Ti-6Al-4V", "steel", "Main aerospace alloy + standard shot"),
        ("316L-SS", "ceramic", "Stainless steel + hard ceramic shot"),
        ("Inconel-718", "tungsten", "Superalloy + heavy tungsten shot"),
        ("Al-7075-T6", "glass", "Lightweight alloy + light glass shot"),
        ("4340-Steel", "cast_iron", "High-strength steel + cast iron shot"),
    ]
    for wp, sp, desc in key_combos:
        tag = f"{wp.replace('-', '_')}__{sp}"
        specs["per_material"].append(
            DatasetSpec(
                name=f"Dataset_{tag}_{n_sims_standard}",
                workpiece=wp,
                shot=sp,
                n_sims=n_sims_standard,
                workers=workers,
                description=desc,
            )
        )

    # ---- All 25 material combos (for multi-material model) ----
    for wp in sorted(WORKPIECE_MATERIALS):
        for sp in sorted(SHOT_MATERIALS):
            tag = f"{wp.replace('-', '_')}__{sp}"
            specs["multi_material"].append(
                DatasetSpec(
                    name=f"MultiMat/{tag}_{n_sims_standard}",
                    workpiece=wp,
                    shot=sp,
                    n_sims=n_sims_standard,
                    workers=workers,
                    description=f"Multi-material: {wp} + {sp}",
                )
            )

    # ---- High-resolution dataset for ConvDecoder ----
    specs["high_res"].append(
        DatasetSpec(
            name=f"Dataset_HighRes_Ti_Steel_{n_sims_hires}",
            workpiece="Ti-6Al-4V",
            shot="steel",
            n_sims=n_sims_hires,
            Nx=100,
            Ny=100,
            G=10,
            workers=workers,
            description="High-res 101x101 mesh for ConvDecoder",
        )
    )

    return specs


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def generate_one_spec(spec: DatasetSpec, output_root: str) -> Dict:
    """Generate one dataset spec. Returns a result dict."""
    dataset_dir = os.path.join(output_root, spec.name)
    os.makedirs(dataset_dir, exist_ok=True)

    gp = GeneratorParams(
        output_dir=dataset_dir,
        n_simulations=spec.n_sims,
        start_index=0,
        workers=spec.workers,
        Nx=spec.Nx,
        Ny=spec.Ny,
        checkerboard_size=spec.G,
        V_range=spec.V_range,
        D_range=spec.D_range,
        n_shots_range=spec.n_shots_range,
        workpiece_material=spec.workpiece,
        shot_material=spec.shot,
        vary_distribution=True,
        base_seed=abs(hash(spec.name)) % (2**31),
    )

    t0 = time.perf_counter()
    n_ok = n_fail = 0
    errors = []

    for i in range(spec.n_sims):
        sim_dir = os.path.join(dataset_dir, f"Simulation_{i}")
        if os.path.exists(os.path.join(sim_dir, "displacements.npy")):
            n_ok += 1
            continue
        res = generate_single_simulation(i, gp)
        if res["success"]:
            n_ok += 1
        else:
            n_fail += 1
            errors.append(f"Sim_{i}: {res['error']}")

    return {
        "name": spec.name,
        "dataset_dir": dataset_dir,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "elapsed_s": time.perf_counter() - t0,
        "success": n_fail == 0 and n_ok > 0,
        "error": "; ".join(errors) if errors else None,
    }


def generate_all_datasets(
    specs_dict: Dict[str, List[DatasetSpec]], output_root: str, skip_existing: bool = True
) -> List[Dict]:
    """Generate all datasets sequentially (generation within each uses workers)."""
    all_specs = (
        specs_dict.get("per_material", []) + specs_dict.get("multi_material", []) + specs_dict.get("high_res", [])
    )
    results = []
    total = len(all_specs)

    for ci, spec in enumerate(all_specs, 1):
        dataset_dir = os.path.join(output_root, spec.name)
        # Check if already fully generated
        if skip_existing and os.path.isdir(dataset_dir):
            existing = sum(
                1
                for d in os.listdir(dataset_dir)
                if d.startswith("Simulation_") and os.path.exists(os.path.join(dataset_dir, d, "displacements.npy"))
            )
            if existing >= spec.n_sims:
                print(f"[{ci:3d}/{total}] SKIP (already {existing} sims): {spec.name}")
                results.append(
                    {
                        "name": spec.name,
                        "dataset_dir": dataset_dir,
                        "n_ok": existing,
                        "n_fail": 0,
                        "elapsed_s": 0.0,
                        "success": True,
                        "error": None,
                    }
                )
                continue

        print(f"\n[{ci:3d}/{total}] Generating {spec.n_sims} sims — {spec.name}")
        print(f"          {spec.description}  " f"Nx={spec.Nx} Ny={spec.Ny} G={spec.G} workers={spec.workers}")
        res = generate_one_spec(spec, output_root)
        print(f"          Done: {res['n_ok']}/{spec.n_sims} OK  " f"({res['elapsed_s']:.1f}s)")
        if res["error"]:
            print(f"          Errors: {res['error'][:120]}")
        results.append(res)

    return results


# ---------------------------------------------------------------------------
# Multi-material dataset merger
# ---------------------------------------------------------------------------


def merge_multi_material_datasets(source_dirs: List[str], merged_dir: str, skip_existing: bool = True) -> int:
    """Copy Simulation_* folders from all source dirs into merged_dir,
    renumbered consecutively from 0. Returns total number of merged sims.
    """
    os.makedirs(merged_dir, exist_ok=True)

    # Count already merged sims to support resuming
    existing_count = (
        sum(
            1
            for d in os.listdir(merged_dir)
            if d.startswith("Simulation_") and os.path.isdir(os.path.join(merged_dir, d))
        )
        if os.path.isdir(merged_dir)
        else 0
    )

    if skip_existing and existing_count > 0:
        # Count total available sims across all sources
        total_available = 0
        for src in source_dirs:
            if os.path.isdir(src):
                total_available += sum(
                    1
                    for d in os.listdir(src)
                    if d.startswith("Simulation_") and os.path.exists(os.path.join(src, d, "displacements.npy"))
                )
        if existing_count >= total_available:
            print(f"  Merge: already {existing_count} sims in {merged_dir} — skip")
            return existing_count

    print(f"  Merging {len(source_dirs)} source dirs into {merged_dir} ...")
    idx = 0
    for src in sorted(source_dirs):
        if not os.path.isdir(src):
            continue
        sims = sorted(
            [
                d
                for d in os.listdir(src)
                if d.startswith("Simulation_") and os.path.exists(os.path.join(src, d, "displacements.npy"))
            ],
            key=lambda x: int(x.split("_")[1]),
        )
        for sim_name in sims:
            dst_name = f"Simulation_{idx}"
            dst_path = os.path.join(merged_dir, dst_name)
            if not os.path.isdir(dst_path):
                shutil.copytree(os.path.join(src, sim_name), dst_path)
            idx += 1

    print(f"  Merged {idx} simulations.")
    return idx


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _make_cosine_scheduler(optimizer, epochs, warmup_frac=0.1, eta_min=1e-6):
    """LinearLR warmup → CosineAnnealingLR."""
    warmup_epochs = max(1, int(epochs * warmup_frac))
    cosine_epochs = max(1, epochs - warmup_epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


@dataclass
class TrainResult:
    mse: float = float("nan")
    rmse_um: float = float("nan")
    epochs_trained: int = 0
    train_time_s: float = float("nan")
    success: bool = False
    error: Optional[str] = None


def train_standard(
    dataset_dir: str,
    model_save_dir: str,
    epochs: int = 100,
    patience: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    use_material: bool = False,
    use_improved: bool = False,
) -> TrainResult:
    """Train DisplacementPredictor (or ImprovedDisplacementPredictor) with
    AdamW + CosineAnnealing + AMP + gradient clipping.

    Args:
        dataset_dir    : Parent folder with Simulation_N/ sub-folders.
        model_save_dir : Where to save the trained model + loss curve.
        use_improved   : If True use ImprovedDisplacementPredictor (4 blocks, dropout).
        use_material   : If True load & use material feature conditioning (mat_dim=10:
                         7 material + 3 shot-process scalars V, D, n_shots).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    try:
        num_nodes, cb_size = M.infer_dataset_shape(dataset_dir)
        train_loader, val_loader, test_loader, loaded_data = M.create_data_loaders(
            base_folder=dataset_dir,
            batch_size=batch_size,
            load_material_features=use_material,
            per_sim_normalize_displacements=True,  # per-sim normalization (HOLE 2 fix)
        )
        disp_scale = loaded_data.get("disp_scale", 1.0)

        if len(val_loader) == 0:
            return TrainResult(
                success=False, error="Val split empty — increase n_sims.", train_time_s=time.perf_counter() - t0
            )

        mat_dim = M.FULL_COND_DIM if use_material else 0

        if use_improved:
            model = M.ImprovedDisplacementPredictor(
                input_channels=1,
                num_nodes=num_nodes,
                checkerboard_size=cb_size,
                mat_dim=mat_dim,
            ).to(device)
        else:
            model = M.DisplacementPredictor(
                input_channels=1,
                num_nodes=num_nodes,
                checkerboard_size=cb_size,
                mat_dim=mat_dim,
            ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        arch_name = "ImprovedDisplacementPredictor" if use_improved else "DisplacementPredictor"
        print(
            f"    {arch_name}: nodes={num_nodes} G={cb_size} " f"mat_dim={mat_dim} params={n_params:,}  device={device}"
        )

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = _make_cosine_scheduler(optimizer, epochs)

        os.makedirs(model_save_dir, exist_ok=True)
        plot_path = os.path.join(model_save_dir, "training_loss_curve.png")

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
            plot_save_path=plot_path,
            use_amp=torch.cuda.is_available(),
            use_material=use_material,
            max_grad_norm=1.0,
        )

        # ---- Evaluation ----
        mse = M.evaluate_model(model, test_loader, criterion, device=device, use_material=use_material)
        rmse = float("nan")

        if mse == mse:  # not NaN
            all_pred, all_true = [], []
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    if use_material:
                        cb, mat_f, disp = batch
                        pred = model(cb.to(device), mat_f.to(device))
                    else:
                        cb, disp = batch
                        pred = model(cb.to(device))
                    all_pred.append(pred.cpu().numpy())
                    all_true.append(disp.numpy())
            # With per-sim normalization, both pred and true are already in normalized space;
            # disp_scale is the median per-sim scale for representative logging only.
            pred_np = np.concatenate(all_pred)
            true_np = np.concatenate(all_true)
            rmse = float(np.sqrt(np.mean((pred_np - true_np) ** 2)))

        # Save model + normalization stats (cb + disp_scale)
        save_path = os.path.join(model_save_dir, "trained_displacement_predictor_full_model.pth")
        torch.save(model, save_path)
        norm = np.array(
            [
                loaded_data.get("checkerboard_norm_min", 0.0),
                loaded_data.get("checkerboard_norm_max", 1.0),
                disp_scale,
            ],
            dtype=np.float64,
        )
        np.save(os.path.join(model_save_dir, "normalization_stats.npy"), norm)

        # Signal per-sim normalization so evaluate_on_dataset uses GT-based per-sim scale
        if loaded_data.get("per_sim_norm", False):
            np.save(os.path.join(model_save_dir, "per_sim_norm.npy"), np.array([True]))

        # Copy reference node coords
        ref_nc = next(
            (
                p / "node_coords.npy"
                for p in sorted(Path(dataset_dir).glob("Simulation_*"))
                if (p / "node_coords.npy").exists()
            ),
            None,
        )
        if ref_nc:
            shutil.copy2(str(ref_nc), os.path.join(model_save_dir, "reference_node_coords.npy"))

        print(f"    Saved to {save_path}")
        result = TrainResult(
            mse=float(mse),
            rmse_um=rmse,
            epochs_trained=len(train_losses),
            train_time_s=time.perf_counter() - t0,
            success=True,
        )
        del model, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TrainResult(
            success=False,
            train_time_s=time.perf_counter() - t0,
            error=str(exc),
        )


def train_conv_decoder(
    dataset_dir: str,
    model_save_dir: str,
    epochs: int = 60,
    patience: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    use_material: bool = False,
) -> TrainResult:
    """Train ConvDecoderPredictor with AdamW + CosineAnnealing + AMP.

    ConvDecoder predicts a (3, H, W) displacement field and is ideal for
    high-resolution meshes since it never allocates a huge FC output layer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    try:
        train_loader, val_loader, test_loader, grid_H, grid_W, _cd_disp_scale, _cd_per_sim = (
            M.create_field_data_loaders(dataset_dir, batch_size=batch_size, load_material_features=use_material)
        )
        _, G = M.infer_dataset_shape(dataset_dir)
        mat_dim = M.FULL_COND_DIM if use_material else 0

        model = M.ConvDecoderPredictor(
            input_channels=1,
            out_H=grid_H,
            out_W=grid_W,
            mat_dim=mat_dim,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"    ConvDecoderPredictor: grid={grid_H}×{grid_W}  G={G}  "
            f"mat_dim={mat_dim}  params={n_params:,}  device={device}"
        )

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = _make_cosine_scheduler(optimizer, epochs)

        # Auto AMP + accum for large grids to avoid OOM
        use_amp = torch.cuda.is_available()
        accum_steps = 4 if (grid_H > 80 or grid_W > 80) else 2

        os.makedirs(model_save_dir, exist_ok=True)
        plot_path = os.path.join(model_save_dir, "training_loss_curve.png")

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
            plot_save_path=plot_path,
            use_amp=use_amp,
            accum_steps=accum_steps,
            use_material=use_material,
            max_grad_norm=1.0,
        )

        # ---- Evaluation on test set ----
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                if use_material:
                    cb, mat_f, field = batch
                    pred = model(cb.to(device), mat_f.to(device))
                else:
                    cb, field = batch
                    pred = model(cb.to(device))
                all_pred.append(pred.cpu().numpy())
                all_true.append(field.numpy())
        pred_np = np.concatenate(all_pred)  # (N, 3, H, W) in per-sim-normalized units [0,1]
        true_np = np.concatenate(all_true)
        mse = float(np.mean((pred_np - true_np) ** 2))  # normalized MSE (unitless)
        # Convert to approximate µm using median per-sim scale for reporting only.
        # The precise per-node RMSE is computed by evaluate_on_dataset at eval time.
        rmse = float(np.sqrt(mse)) * _cd_disp_scale * 1e6

        save_path = os.path.join(model_save_dir, "trained_conv_decoder_full_model.pth")
        torch.save(model, save_path)
        np.save(os.path.join(model_save_dir, "normalization_stats.npy"), np.array([0.0, 1.0, _cd_disp_scale]))
        np.save(os.path.join(model_save_dir, "per_sim_norm.npy"), np.array([True]))

        ref_nc = next(
            (
                p / "node_coords.npy"
                for p in sorted(Path(dataset_dir).glob("Simulation_*"))
                if (p / "node_coords.npy").exists()
            ),
            None,
        )
        if ref_nc:
            shutil.copy2(str(ref_nc), os.path.join(model_save_dir, "reference_node_coords.npy"))

        print(f"    Saved to {save_path}")
        return TrainResult(
            mse=mse,
            rmse_um=rmse,
            epochs_trained=len(train_losses),
            train_time_s=time.perf_counter() - t0,
            success=True,
        )

    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TrainResult(
            success=False,
            train_time_s=time.perf_counter() - t0,
            error=str(exc),
        )


def train_siren(
    dataset_dir: str,
    model_save_dir: str,
    epochs: int = 80,
    patience: int = 15,
    batch_size: int = 8,
    k_nodes: int = 1024,
    lr: float = 5e-4,
    use_material: bool = False,
) -> TrainResult:
    """Train SIRENPredictor (INR decoder — resolution-free inference).

    SIREN trains on k_nodes random node subsamples per step so GPU memory
    is O(batch × k_nodes) regardless of total mesh size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()

    try:
        train_loader, val_loader, test_loader, N_total, disp_scale = M.create_siren_loaders(
            dataset_dir,
            k_nodes=k_nodes,
            batch_size=batch_size,
            load_material_features=use_material,
            normalize_displacements=True,
        )
        mat_dim = M.FULL_COND_DIM if use_material else 0
        print(f"    disp_scale={disp_scale:.4e} m (targets normalized to [-1,1])")

        model = M.SIRENPredictor(
            input_channels=1,
            latent_dim=256,
            hidden=256,
            n_layers=4,
            mat_dim=mat_dim,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"    SIRENPredictor: N_total={N_total} k_nodes={k_nodes}  "
            f"mat_dim={mat_dim}  params={n_params:,}  device={device}"
        )

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = _make_cosine_scheduler(optimizer, epochs)

        os.makedirs(model_save_dir, exist_ok=True)
        plot_path = os.path.join(model_save_dir, "training_loss_curve.png")

        # SIREN needs its own training loop since batches include coords
        best_val = float("inf")
        patience_ctr = 0
        train_losses, val_losses = [], []

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("SIREN Training Loss")
        (line1,) = ax.plot([], [], label="Train", color="blue")
        (line2,) = ax.plot([], [], label="Val", color="orange")
        ax.legend()

        use_amp = torch.cuda.is_available()
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                if use_material:
                    cbs, mats, coords, disps = batch
                    mats = mats.to(device)
                else:
                    cbs, coords, disps = batch
                    mats = None
                cbs = cbs.to(device)
                coords = coords.to(device)
                disps = disps.to(device)
                optimizer.zero_grad()
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        pred = model(cbs, coords, mats)
                        loss = criterion(pred, disps)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(cbs, coords, mats)
                    loss = criterion(pred, disps)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                epoch_loss += loss.item()
            train_loss = epoch_loss / len(train_loader)
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    if use_material:
                        cbs, mats, coords, disps = val_batch
                        mats = mats.to(device)
                    else:
                        cbs, coords, disps = val_batch
                        mats = None
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            val_loss += criterion(
                                model(cbs.to(device), coords.to(device), mats),
                                disps.to(device),
                            ).item()
                    else:
                        val_loss += criterion(
                            model(cbs.to(device), coords.to(device), mats),
                            disps.to(device),
                        ).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            scheduler.step()

            if val_loss < best_val:
                best_val = val_loss
                patience_ctr = 0
                torch.save(model, os.path.join(model_save_dir, "siren_best.pth"))
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"    Early stop at epoch {epoch+1}")
                    break

            line1.set_xdata(range(1, len(train_losses) + 1))
            line1.set_ydata(train_losses)
            line2.set_xdata(range(1, len(val_losses) + 1))
            line2.set_ydata(val_losses)
            ax.relim()
            ax.autoscale_view()
            print(f"    Epoch {epoch+1}/{epochs}  train={train_loss:.4e}  " f"val={val_loss:.4e}")

        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        save_path = os.path.join(model_save_dir, "trained_siren_full_model.pth")
        torch.save(model, save_path)

        # Quick test-set evaluation (subsample for speed)
        model.eval()
        mse_accum, n_batches = 0.0, 0
        with torch.no_grad():
            for batch in test_loader:
                if use_material:
                    cbs, mats, coords, disps = batch
                    mats = mats.to(device)
                else:
                    cbs, coords, disps = batch
                    mats = None
                pred = model(cbs.to(device), coords.to(device), mats)
                mse_accum += nn.MSELoss()(pred, disps.to(device)).item()
                n_batches += 1
        # MSE is in normalised space; convert back to physical µm
        mse_norm = mse_accum / max(n_batches, 1)
        mse = mse_norm * (disp_scale**2)  # physical (m²)
        rmse = float(np.sqrt(mse)) * 1e6  # µm

        # Save displacement scale for inference denormalization
        np.save(os.path.join(model_save_dir, "disp_scale.npy"), np.array([disp_scale], dtype=np.float64))

        ref_nc = next(
            (
                p / "node_coords.npy"
                for p in sorted(Path(dataset_dir).glob("Simulation_*"))
                if (p / "node_coords.npy").exists()
            ),
            None,
        )
        if ref_nc:
            shutil.copy2(str(ref_nc), os.path.join(model_save_dir, "reference_node_coords.npy"))

        print(f"    Saved to {save_path}")
        del model, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TrainResult(
            mse=mse,
            rmse_um=rmse,
            epochs_trained=len(train_losses),
            train_time_s=time.perf_counter() - t0,
            success=True,
        )

    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TrainResult(
            success=False,
            train_time_s=time.perf_counter() - t0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(rows: List[Dict], path: str) -> None:
    fieldnames = [
        "model_name",
        "dataset",
        "arch",
        "n_sims",
        "Nx",
        "Ny",
        "success",
        "mse",
        "rmse_um",
        "epochs_trained",
        "train_s",
        "error",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nReport written to: {path}")


def print_summary(rows: List[Dict]) -> None:
    ok = [r for r in rows if r.get("success")]
    fail = [r for r in rows if not r.get("success")]
    print("\n" + "=" * 72)
    print(f"LARGE-SCALE TRAINING SUMMARY  ({len(rows)} models)")
    print("=" * 72)
    print(f"  Successful : {len(ok)}")
    print(f"  Failed     : {len(fail)}")
    if ok:
        rmses = [float(r["rmse_um"]) for r in ok if r.get("rmse_um") and str(r["rmse_um"]) != "nan"]
        if rmses:
            print(f"\n  RMSE range : {min(rmses):.2f} – {max(rmses):.2f} µm")
            print(f"  RMSE mean  : {sum(rmses)/len(rmses):.2f} µm")
        for r in ok:
            print(f"\n  {r['model_name']}")
            print(f"    arch={r['arch']}  dataset={r['dataset']}")
            print(
                f"    RMSE={r.get('rmse_um', '?')} µm  "
                f"epochs={r.get('epochs_trained', '?')}  "
                f"time={r.get('train_s', '?')}s"
            )
    if fail:
        print("\n  Failed models:")
        for r in fail:
            print(f"    {r['model_name']}: {r.get('error', '?')}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def _train_multitask(
    dataset_dir: str,
    model_save_dir: str,
    epochs: int = 100,
    patience: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    use_material: bool = False,
    loss_weights: tuple = (1.0, 0.005, 0.01),
    warmup_disp_epochs: int = 20,
    stress_components: int = 2,
) -> TrainResult:
    """Train MultiTaskPredictor (physics 6-ch CB → displacement + stress + scalars)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    try:
        train_loader, val_loader, test_loader, stats = M.create_multitask_data_loaders(
            dataset_dir,
            batch_size=batch_size,
            load_material_features=use_material,
            use_physics_cb=True,
        )
        if len(val_loader) == 0:
            return TrainResult(success=False, error="Val split empty.", train_time_s=time.perf_counter() - t0)

        model = M.MultiTaskPredictor(
            input_channels=stats["input_channels"],
            num_nodes=stats["num_nodes"],
            checkerboard_size=stats["checkerboard_size"],
            mat_dim=M.MAT_DIM if use_material else 0,
            predict_stress=True,
            predict_scalars=True,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"    MultiTaskPredictor: nodes={stats['num_nodes']}  "
            f"C_in={stats['input_channels']}  params={n_params:,}  device={device}"
        )

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = _make_cosine_scheduler(optimizer, epochs)
        os.makedirs(model_save_dir, exist_ok=True)
        plot_path = os.path.join(model_save_dir, "training_loss_curve.png")

        train_losses, val_losses = M.train_model_multitask(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=epochs,
            patience=patience,
            device=device,
            plot_save_path=plot_path,
            use_amp=torch.cuda.is_available(),
            max_grad_norm=1.0,
            loss_weights=loss_weights,
            use_material=use_material,
            stats=stats,
            warmup_disp_epochs=warmup_disp_epochs,
            stress_components=stress_components,
        )

        # Evaluate displacement RMSE on test set (denormalized)
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                if use_material:
                    cb, mat_f, disp_t, stress_t, scalar_t = batch
                    out = model(cb.to(device), mat_f.to(device))
                else:
                    cb, disp_t, stress_t, scalar_t = batch
                    out = model(cb.to(device))
                all_pred.append(out["displacements"].cpu().numpy())
                all_true.append(disp_t.numpy())
        pred_np = np.concatenate(all_pred) * stats["disp_scale"]
        true_np = np.concatenate(all_true) * stats["disp_scale"]
        mse = float(np.mean((pred_np - true_np) ** 2))
        rmse = float(np.sqrt(mse)) * 1e6

        save_path = os.path.join(model_save_dir, "trained_multitask_model.pth")
        torch.save(model, save_path)
        np.save(
            os.path.join(model_save_dir, "multitask_stats.npy"),
            np.array(
                [
                    stats["disp_scale"],
                    stats["stress_scale"],
                ]
            ),
        )
        # Per-sim normalization flag (always True for MT, which uses per-sim by default)
        if stats.get("per_sim_norm", False):
            np.save(os.path.join(model_save_dir, "per_sim_norm.npy"), np.array([True]))
        np.save(os.path.join(model_save_dir, "scalar_scales.npy"), stats["scalar_scales"])

        del model, train_loader, val_loader, test_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return TrainResult(
            mse=mse,
            rmse_um=rmse,
            epochs_trained=len(train_losses),
            train_time_s=time.perf_counter() - t0,
            success=True,
        )

    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return TrainResult(
            success=False,
            train_time_s=time.perf_counter() - t0,
            error=str(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="peen-ml large-scale training pipeline.")
    parser.add_argument(
        "--output", default="./LargeScaleResults", help="Root output directory (default: ./LargeScaleResults)"
    )
    parser.add_argument("--n_sims", type=int, default=200, help="Simulations per material combo (default 200)")
    parser.add_argument(
        "--hires", type=int, default=300, help="Simulations for high-res ConvDecoder dataset (default 300)"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs (default 100)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default 20)")
    parser.add_argument("--batch", type=int, default=32, help="Training batch size (default 32)")
    parser.add_argument("--workers", type=int, default=4, help="Dataset generation workers per spec (default 4)")
    parser.add_argument(
        "--phase", default="all", choices=["all", "generate", "train"], help="Which phase to run (default: all)"
    )
    parser.add_argument("--skip-gen", action="store_true", help="Skip already-generated datasets")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 20 sims, 15 epochs (for testing)")
    parser.add_argument("--tiny", action="store_true", help="Tiny mode: 10 sims, 5 epochs (smoke test)")
    parser.add_argument(
        "--no-multimat", action="store_true", help="Skip multi-material (all 25 combos) generation/training"
    )
    parser.add_argument("--no-hires", action="store_true", help="Skip high-resolution ConvDecoder dataset/training")
    parser.add_argument("--no-siren", action="store_true", help="Skip SIREN model training")
    args = parser.parse_args()

    if args.tiny:
        args.n_sims = 10
        args.hires = 10
        args.epochs = 5
        args.patience = 3
        args.batch = 8
        args.workers = 2
    elif args.quick:
        args.n_sims = 20
        args.hires = 20
        args.epochs = 15
        args.patience = 5
        args.batch = 16
        args.workers = 4

    output_root = args.output
    os.makedirs(output_root, exist_ok=True)

    device_str = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"

    print("=" * 72)
    print("peen-ml Large-Scale Training Pipeline")
    print(f"  Output   : {output_root}")
    print(f"  Sims/cfg : {args.n_sims}  Hi-res: {args.hires}")
    print(f"  Epochs   : {args.epochs}  Patience: {args.patience}")
    print(f"  Batch    : {args.batch}   Workers: {args.workers}")
    print(f"  Device   : {device_str}")
    print(f"  Phase    : {args.phase}")
    print("=" * 72)

    # ----------------------------------------------------------------
    # Phase 1 — Dataset generation
    # ----------------------------------------------------------------
    if args.phase in ("all", "generate"):
        specs_dict = build_dataset_specs(
            n_sims_standard=args.n_sims,
            n_sims_hires=args.hires,
            workers=args.workers,
        )
        if args.no_multimat:
            specs_dict["multi_material"] = []
        if args.no_hires:
            specs_dict["high_res"] = []

        print("\n--- Phase 1: Dataset Generation ---")
        gen_results = generate_all_datasets(
            specs_dict,
            output_root,
            skip_existing=args.skip_gen or (args.phase == "train"),
        )
        n_gen_ok = sum(1 for r in gen_results if r["success"])
        print(f"\nGeneration complete: {n_gen_ok}/{len(gen_results)} datasets OK")

    # ----------------------------------------------------------------
    # Phase 1b — Merge multi-material datasets
    # ----------------------------------------------------------------
    merged_dir = os.path.join(output_root, "Dataset_MultiMat_Merged")
    if (args.phase in ("all", "generate")) and not args.no_multimat:
        print("\n--- Phase 1b: Merging multi-material datasets ---")
        multimat_source_dirs = [
            os.path.join(output_root, f"MultiMat/{wp.replace('-', '_')}__{sp}_{args.n_sims}")
            for wp in sorted(WORKPIECE_MATERIALS)
            for sp in sorted(SHOT_MATERIALS)
        ]
        n_merged = merge_multi_material_datasets(multimat_source_dirs, merged_dir)
        print(f"  Combined dataset: {n_merged} simulations at {merged_dir}")

    # ----------------------------------------------------------------
    # Phase 2 — Training
    # ----------------------------------------------------------------
    if args.phase in ("all", "train"):
        print("\n--- Phase 2: Model Training ---")
        report_rows = []

        # Helper to run + record a training job
        def run_training(model_name, train_fn, dataset_path, arch, n_sims, Nx, Ny, batch_size_override=None, **kwargs):
            if not os.path.isdir(dataset_path):
                print(f"\n  [{model_name}] SKIP — dataset not found: {dataset_path}")
                report_rows.append(
                    {
                        "model_name": model_name,
                        "dataset": dataset_path,
                        "arch": arch,
                        "n_sims": n_sims,
                        "Nx": Nx,
                        "Ny": Ny,
                        "success": False,
                        "error": "dataset not found",
                    }
                )
                return

            n_available = sum(
                1
                for d in os.listdir(dataset_path)
                if d.startswith("Simulation_") and os.path.exists(os.path.join(dataset_path, d, "displacements.npy"))
            )
            if n_available < 7:
                print(f"\n  [{model_name}] SKIP — only {n_available} sims (need ≥7)")
                report_rows.append(
                    {
                        "model_name": model_name,
                        "dataset": dataset_path,
                        "arch": arch,
                        "n_sims": n_available,
                        "Nx": Nx,
                        "Ny": Ny,
                        "success": False,
                        "error": f"only {n_available} sims",
                    }
                )
                return

            model_save_dir = os.path.join(output_root, "Models", model_name)
            print(f"\n  [{model_name}]  arch={arch}  sims={n_available}  " f"Nx={Nx} Ny={Ny}")
            # Free GPU cache from previous run before starting a new one
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()
            effective_batch = batch_size_override if batch_size_override is not None else args.batch
            tr = train_fn(
                dataset_dir=dataset_path,
                model_save_dir=model_save_dir,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=effective_batch,
                **kwargs,
            )
            status = "OK" if tr.success else "FAIL"
            if tr.success:
                print(
                    f"    [{status}] RMSE={tr.rmse_um:.2f}  "
                    f"epochs={tr.epochs_trained}  "
                    f"time={tr.train_time_s:.0f}s"
                )
            else:
                print(f"    [{status}] {tr.error}")
            report_rows.append(
                {
                    "model_name": model_name,
                    "dataset": os.path.basename(dataset_path),
                    "arch": arch,
                    "n_sims": n_available,
                    "Nx": Nx,
                    "Ny": Ny,
                    "success": tr.success,
                    "mse": f"{tr.mse:.4e}" if tr.success else "nan",
                    "rmse_um": f"{tr.rmse_um:.3f}" if tr.success else "nan",
                    "epochs_trained": tr.epochs_trained,
                    "train_s": f"{tr.train_time_s:.1f}",
                    "error": tr.error or "",
                }
            )

        # ---- Model A: Standard DisplacementPredictor on Ti+steel 200-sim ----
        ti_steel_dir = os.path.join(
            output_root,
            f"Dataset_Ti_6Al_4V__steel_{args.n_sims}",
        )
        run_training(
            model_name="A_DisplPredictor_Ti_Steel",
            train_fn=train_standard,
            dataset_path=ti_steel_dir,
            arch="DisplacementPredictor",
            n_sims=args.n_sims,
            Nx=50,
            Ny=50,
            use_improved=False,
            use_material=False,
        )

        # ---- Model B: ImprovedDisplacementPredictor on Ti+steel ----
        run_training(
            model_name="B_ImprovedPredictor_Ti_Steel",
            train_fn=train_standard,
            dataset_path=ti_steel_dir,
            arch="ImprovedDisplacementPredictor",
            n_sims=args.n_sims,
            Nx=50,
            Ny=50,
            use_improved=True,
            use_material=False,
        )

        # ---- Model C: Material-conditioned Improved on combined MultiMat ----
        if not args.no_multimat:
            run_training(
                model_name="C_MatCond_MultiMat",
                train_fn=train_standard,
                dataset_path=merged_dir,
                arch="ImprovedDisplacementPredictor(mat_dim=10)",
                n_sims=args.n_sims * 25,
                Nx=50,
                Ny=50,
                batch_size_override=max(8, args.batch // 2),  # smaller for 5000-sim dataset
                use_improved=True,
                use_material=True,
            )

        # ---- Model D: Per-material Improved for 5 key combos ----
        key_combos = [
            ("Ti-6Al-4V", "steel"),
            ("316L-SS", "ceramic"),
            ("Inconel-718", "tungsten"),
            ("Al-7075-T6", "glass"),
            ("4340-Steel", "cast_iron"),
        ]
        for wp, sp in key_combos:
            tag = f"{wp.replace('-', '_')}__{sp}"
            ds_dir = os.path.join(output_root, f"Dataset_{tag}_{args.n_sims}")
            run_training(
                model_name=f"D_Improved_{tag}",
                train_fn=train_standard,
                dataset_path=ds_dir,
                arch="ImprovedDisplacementPredictor",
                n_sims=args.n_sims,
                Nx=50,
                Ny=50,
                use_improved=True,
                use_material=False,
            )

        # ---- Model E: ConvDecoder on high-res ----
        if not args.no_hires:
            hires_dir = os.path.join(output_root, f"Dataset_HighRes_Ti_Steel_{args.hires}")
            run_training(
                model_name="E_ConvDecoder_HighRes",
                train_fn=train_conv_decoder,
                dataset_path=hires_dir,
                arch="ConvDecoderPredictor",
                n_sims=args.hires,
                Nx=100,
                Ny=100,
                use_material=False,
            )

        # ---- Model F: SIREN on Ti+steel ----
        if not args.no_siren:
            run_training(
                model_name="F_SIREN_Ti_Steel",
                train_fn=train_siren,
                dataset_path=ti_steel_dir,
                arch="SIRENPredictor",
                n_sims=args.n_sims,
                Nx=50,
                Ny=50,
                use_material=False,
                k_nodes=1024,
            )

        # ---- Model MT: MultiTaskPredictor on Ti+steel (physics checkerboard) ----
        _mt_dir = os.path.join(output_root, "Models", "MT_MultiTask_Ti_Steel")
        _mt_avail = (
            sum(
                1
                for d in os.listdir(ti_steel_dir)
                if d.startswith("Simulation_")
                and os.path.exists(os.path.join(ti_steel_dir, d, "checkerboard_physics.npy"))
            )
            if os.path.isdir(ti_steel_dir)
            else 0
        )

        if _mt_avail >= 7:
            print(f"\n  [MT_MultiTask_Ti_Steel]  arch=MultiTaskPredictor  " f"sims={_mt_avail}  (6-ch physics CB)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()
            _mt_result = _train_multitask(
                dataset_dir=ti_steel_dir,
                model_save_dir=_mt_dir,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch,
                use_material=False,
            )
            if _mt_result.success:
                print(
                    f"    [OK] disp_RMSE={_mt_result.rmse_um:.2f} µm  "
                    f"epochs={_mt_result.epochs_trained}  time={_mt_result.train_time_s:.0f}s"
                )
            else:
                print(f"    [FAIL] {_mt_result.error}")
            report_rows.append(
                {
                    "model_name": "MT_MultiTask_Ti_Steel",
                    "dataset": os.path.basename(ti_steel_dir),
                    "arch": "MultiTaskPredictor",
                    "n_sims": _mt_avail,
                    "Nx": 50,
                    "Ny": 50,
                    "success": _mt_result.success,
                    "mse": f"{_mt_result.mse:.4e}" if _mt_result.success else "nan",
                    "rmse_um": f"{_mt_result.rmse_um:.3f}" if _mt_result.success else "nan",
                    "epochs_trained": _mt_result.epochs_trained,
                    "train_s": f"{_mt_result.train_time_s:.1f}",
                    "error": _mt_result.error or "",
                }
            )
        else:
            print(
                f"\n  [MT_MultiTask_Ti_Steel] SKIP — only {_mt_avail} sims with "
                "checkerboard_physics.npy (need ≥7; regenerate dataset first)"
            )

        # ---- Models I/J/K: InfluenceField ConvDecoder on various datasets ----
        def _run_influence(model_name, dataset_dir, n_sims, Nx, Ny):
            """Train + record one InfluenceField ConvDecoder variant."""
            inf_avail = sum(
                1
                for d in (os.listdir(dataset_dir) if os.path.isdir(dataset_dir) else [])
                if d.startswith("Simulation_") and os.path.exists(os.path.join(dataset_dir, d, "influence_fields.npy"))
            )
            if inf_avail < 7:
                print(
                    f"\n  [{model_name}] SKIP — only {inf_avail} sims with "
                    "influence_fields.npy (need ≥7; run backfill_physics_files.py first)"
                )
                return
            print(f"\n  [{model_name}]  arch=InfluenceField ConvDecoder  " f"sims={inf_avail}  (4-ch influence fields)")
            save_dir = os.path.join(output_root, "Models", model_name)
            os.makedirs(save_dir, exist_ok=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc

            gc.collect()
            res = M.train_influence_field_model(
                dataset_dir=dataset_dir,
                model_save_dir=save_dir,
                epochs=args.epochs,
                patience=args.patience,
            )
            if res.get("success"):
                print(f"    [OK] RMSE={res['rmse_um']:.2f}  epochs={res['epochs_trained']}")
            else:
                print(f"    [FAIL] {res.get('error', 'unknown')}")
            report_rows.append(
                {
                    "model_name": model_name,
                    "dataset": os.path.basename(dataset_dir),
                    "arch": "InfluenceField ConvDecoder",
                    "n_sims": inf_avail,
                    "Nx": Nx,
                    "Ny": Ny,
                    "success": res.get("success", False),
                    "mse": f"{res.get('mse', float('nan')):.4e}" if res.get("success") else "nan",
                    "rmse_um": f"{res.get('rmse_um', float('nan')):.3f}" if res.get("success") else "nan",
                    "epochs_trained": res.get("epochs_trained", 0),
                    "train_s": f"{res.get('train_time_s', 0):.1f}" if "train_time_s" in res else "nan",
                    "error": res.get("error", ""),
                }
            )

        # I variants: 5 key material combos (200-sim each)
        for wp, sp in key_combos:
            tag = f"{wp.replace('-', '_')}__{sp}"
            ds_dir = os.path.join(output_root, f"Dataset_{tag}_{args.n_sims}")
            _run_influence(f"I_InfluenceField_{tag}", ds_dir, args.n_sims, 50, 50)

        # J: InfluenceField on Ti+steel 2000-sim (tests data-scaling hypothesis)
        ti_steel_2000_dir = os.path.join(output_root, "Dataset_Ti_6Al_4V__steel_2000")
        if os.path.isdir(ti_steel_2000_dir):
            _run_influence("J_InfluenceField_Ti_steel_2000", ti_steel_2000_dir, 2000, 50, 50)
        else:
            print(f"\n  [J_InfluenceField_Ti_steel_2000] SKIP — dataset not found: {ti_steel_2000_dir}")

        # K: InfluenceField on HighRes Ti+steel (tests resolution scaling)
        if not args.no_hires:
            hires_dir_k = os.path.join(output_root, f"Dataset_HighRes_Ti_Steel_{args.hires}")
            if os.path.isdir(hires_dir_k):
                _run_influence("K_InfluenceField_HighRes", hires_dir_k, args.hires, 100, 100)
            else:
                print(f"\n  [K_InfluenceField_HighRes] SKIP — dataset not found: {hires_dir_k}")

        # ---- Write report ----
        report_path = os.path.join(output_root, "large_scale_results.csv")
        write_report(report_rows, report_path)
        print_summary(report_rows)


if __name__ == "__main__":
    main()
