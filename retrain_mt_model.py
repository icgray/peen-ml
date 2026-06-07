#!/usr/bin/env python3
"""Retrain only the MT_MultiTask_Ti_Steel model with phased-warmup settings.

Uses warmup_disp_epochs=20 so the displacement head trains alone for the
first 20 epochs before stress and cupping losses are activated.  This fixes
the prior run where MT ux r=0.31 < single-task ux r=0.37 on the same data.

Usage:
    python retrain_mt_model.py [--epochs 100] [--output LargeScaleRun1]
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE / "src" / "peen-ml"
for _p in [str(_SRC), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model as M


def _cosine_scheduler(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",  default="LargeScaleRun1",
                        help="Run directory (default: LargeScaleRun1)")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--patience",type=int, default=20)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--lr",      type=float, default=3e-4)
    args = parser.parse_args()

    run_dir     = Path(args.output).resolve()
    dataset_dir = str(run_dir / "Dataset_Ti_6Al_4V__steel_200")
    model_dir   = str(run_dir / "Models" / "MT_MultiTask_Ti_Steel")

    if not os.path.isdir(dataset_dir):
        print(f"ERROR: dataset not found: {dataset_dir}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  MT_MultiTask_Ti_Steel  (phased warmup, warmup=20 epochs)")
    print(f"  dataset : {dataset_dir}")
    print(f"  save to : {model_dir}")
    print(f"  device  : {device}  epochs={args.epochs}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    train_loader, val_loader, test_loader, stats = M.create_multitask_data_loaders(
        dataset_dir,
        batch_size            = args.batch,
        load_material_features= False,
        use_physics_cb        = True,
    )

    model = M.MultiTaskPredictor(
        input_channels    = stats["input_channels"],
        num_nodes         = stats["num_nodes"],
        checkerboard_size = stats["checkerboard_size"],
        mat_dim           = 0,
        predict_stress    = True,
        predict_scalars   = True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  nodes={stats['num_nodes']}  C_in={stats['input_channels']}  "
          f"params={n_params:,}  checkerboard={stats['checkerboard_size']}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = _cosine_scheduler(optimizer, args.epochs)
    os.makedirs(model_dir, exist_ok=True)
    plot_path = os.path.join(model_dir, "training_loss_curve.png")

    train_losses, val_losses = M.train_model_multitask(
        model              = model,
        train_loader       = train_loader,
        val_loader         = val_loader,
        optimizer          = optimizer,
        scheduler          = scheduler,
        epochs             = args.epochs,
        patience           = args.patience,
        device             = device,
        plot_save_path     = plot_path,
        use_amp            = torch.cuda.is_available(),
        max_grad_norm      = 1.0,
        loss_weights       = (1.0, 0.005, 0.01),   # λ_d, λ_s, λ_c
        use_material       = False,
        stats              = stats,
        warmup_disp_epochs = 20,
        stress_components  = 2,
    )

    # Evaluate displacement RMSE on test set
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            cb, disp_t, stress_t, scalar_t = batch
            out = model(cb.to(device))
            all_pred.append(out["displacements"].cpu().numpy())
            all_true.append(disp_t.numpy())
    pred_np = np.concatenate(all_pred) * stats["disp_scale"]
    true_np = np.concatenate(all_true) * stats["disp_scale"]
    rmse_um = float(np.sqrt(np.mean((pred_np - true_np) ** 2))) * 1e6

    save_path = os.path.join(model_dir, "trained_multitask_model.pth")
    torch.save(model, save_path)
    np.save(os.path.join(model_dir, "multitask_stats.npy"),
            np.array([stats["disp_scale"], stats["stress_scale"]]))

    elapsed = time.perf_counter() - t0
    print(f"\n[OK]  RMSE={rmse_um:.2f} µm  "
          f"epochs={len(train_losses)}  time={elapsed:.0f}s")
    print(f"Model saved to: {save_path}")
    print("\nNext: run_eval.py --models MT to update eval_results.csv")


if __name__ == "__main__":
    main()
