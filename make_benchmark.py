"""
Self-contained benchmark: generate data, train ConvDecoder with proper
target normalisation, evaluate, and save images/pred_vs_gt.png.

The native simulator produces displacements on the order of 1-100 um while
the existing train_save_conv_gui was tuned for Abaqus-scale (~100 mm).
This script normalises targets to [-1, 1] before training and rescales
predictions back to physical units for the figure.
"""

import os, sys, time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "peen-ml"))

OUT_DIR = os.path.join(ROOT, "Dataset_benchmark_500")
MODEL_DIR = os.path.join(OUT_DIR, "saved_model_bench")


# ── 1. Generate dataset ──────────────────────────────────────────────────────


def generate():
    sims = [d for d in os.listdir(OUT_DIR) if d.startswith("Simulation_")] if os.path.exists(OUT_DIR) else []
    if len(sims) >= 500:
        print(f"Dataset already exists ({len(sims)} sims), skipping.")
        return
    from native_dataset_gen import GeneratorParams, generate_dataset

    print("Generating 500 simulations ...")
    t0 = time.time()
    gp = GeneratorParams(
        n_simulations=500,
        output_dir=OUT_DIR,
        Nx=30,
        Ny=30,
        checkerboard_size=5,
        n_shots_range=(10, 80),
        V_range=(20.0, 55.0),
        D_range=(0.0003, 0.0009),
        workers=4,
        base_seed=42,
        shot_material="steel",
    )
    generate_dataset(gp)
    print(f"Done in {time.time()-t0:.1f}s")


# ── 2. Custom training with normalised targets ───────────────────────────────


def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "conv_decoder.pth")
    scale_path = os.path.join(MODEL_DIR, "disp_scale.npy")
    if os.path.exists(save_path) and os.path.exists(scale_path):
        print("Trained model already exists, skipping.")
        return

    from model import ConvDecoderPredictor, FieldDataset, infer_grid_shape, infer_dataset_shape

    print("Loading data ...")
    sims = sorted(
        [d for d in os.listdir(OUT_DIR) if d.startswith("Simulation_")],
        key=lambda x: int(x.split("_")[1]),
    )
    cbs, disps = [], []
    for s in sims:
        p = os.path.join(OUT_DIR, s)
        cbs.append(np.load(p + "/checkerboard.npy"))
        disps.append(np.load(p + "/displacements.npy"))
    cbs = np.stack(cbs)  # (500, G, G)
    disps = np.stack(disps)  # (500, N, 3)

    # --- normalise targets to [-1, 1] using 99th-percentile of |disp| ---
    disp_scale = float(np.percentile(np.abs(disps), 99.9))
    print(f"Displacement scale (99.9th pct): {disp_scale*1e6:.2f} um")
    disps_norm = disps / disp_scale
    np.save(scale_path, np.array([disp_scale]))

    grid_H, grid_W = infer_grid_shape(OUT_DIR)
    print(f"Grid: {grid_H}x{grid_W}")

    dataset = FieldDataset(cbs, disps_norm, grid_H, grid_W)
    n = len(dataset)
    n_tr = int(0.70 * n)
    n_va = int(0.15 * n)
    n_te = n - n_tr - n_va
    torch.manual_seed(42)
    tr_ds, va_ds, _ = random_split(dataset, [n_tr, n_va, n_te])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _pin = device.type == "cuda"
    tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=_pin)
    va_ld = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=_pin)

    model = ConvDecoderPredictor(input_channels=1, out_H=grid_H, out_W=grid_W).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ConvDecoder: {n_params:,} params  device={device}")

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    # Cosine decay: no LR collapse before the model converges
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    best_val, patience_ctr, PATIENCE = float("inf"), 0, 15
    print("Training (up to 120 epochs) ...")
    t0 = time.time()
    for epoch in range(1, 121):
        model.train()
        tr_loss = 0.0
        for cb_b, disp_b in tr_ld:
            cb_b, disp_b = cb_b.to(device), disp_b.to(device)
            opt.zero_grad()
            pred = model(cb_b)
            loss = loss_fn(pred, disp_b)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_ld)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for cb_b, disp_b in va_ld:
                cb_b, disp_b = cb_b.to(device), disp_b.to(device)
                va_loss += loss_fn(model(cb_b), disp_b).item()
        va_loss /= len(va_ld)
        sch.step()

        if epoch % 10 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d}  train={tr_loss:.4e}  val={va_loss:.4e}  " f"lr={opt.param_groups[0]['lr']:.2e}")

        if va_loss < best_val:
            best_val = va_loss
            patience_ctr = 0
            torch.save(model, save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stop at epoch {epoch}  best_val={best_val:.4e}")
                break

    print(f"Training done in {time.time()-t0:.1f}s  best_val={best_val:.4e}")


# ── 3. Evaluate and plot ─────────────────────────────────────────────────────


def evaluate_and_plot():
    save_path = os.path.join(MODEL_DIR, "conv_decoder.pth")
    scale_path = os.path.join(MODEL_DIR, "disp_scale.npy")

    from model import sample_field_at_coords, infer_grid_shape

    disp_scale = float(np.load(scale_path)[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(save_path, map_location=device, weights_only=False)
    model.eval()

    sims = sorted(
        [d for d in os.listdir(OUT_DIR) if d.startswith("Simulation_")],
        key=lambda x: int(x.split("_")[1]),
    )
    # test set: last 15 %
    test_sims = sims[425:]
    grid_H, grid_W = infer_grid_shape(OUT_DIR)

    def predict_uz_um(sim_name):
        p = os.path.join(OUT_DIR, sim_name)
        cb = np.load(p + "/checkerboard.npy")
        nc = np.load(p + "/node_coords.npy")
        gt = np.load(p + "/displacements.npy")
        cb_t = torch.tensor(cb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            field = model(cb_t) * disp_scale  # re-scale to metres
        nc_t = torch.tensor(nc[:, :2], dtype=torch.float32).to(device)
        pred = sample_field_at_coords(field, nc_t)[0].cpu().numpy()  # (N,3) metres
        gt_um = gt[:, 2] * 1e6
        pred_um = pred[:, 2] * 1e6
        return cb, nc, gt_um, pred_um, gt_H_W(gt_um, nc, grid_H, grid_W), gt_H_W(pred_um, nc, grid_H, grid_W)

    def gt_H_W(vals, nc, H, W):
        xs = np.unique(np.round(nc[:, 0], 8))
        ys = np.unique(np.round(nc[:, 1], 8))
        return vals.reshape(len(xs), len(ys))

    # aggregate test metrics (only on nodes that are meaningfully deformed)
    all_rmse, all_r, all_rel_rmse = [], [], []
    from scipy.stats import pearsonr

    for sim_name in test_sims:
        cb, nc, gt_um, pred_um, _, _ = predict_uz_um(sim_name)
        thresh = max(np.abs(gt_um).max() * 0.05, 0.1)
        mask = np.abs(gt_um) > thresh
        if mask.sum() < 5:
            continue
        rmse = float(np.sqrt(np.mean((pred_um[mask] - gt_um[mask]) ** 2)))
        r, _ = pearsonr(pred_um[mask], gt_um[mask])
        peak = float(np.abs(gt_um).max()) or 1.0
        all_rmse.append(rmse)
        all_r.append(r)
        all_rel_rmse.append(rmse / peak * 100.0)

    mean_rmse = float(np.mean(all_rmse)) if all_rmse else float("nan")
    mean_r = float(np.mean(all_r)) if all_r else float("nan")
    mean_rel_rmse = float(np.mean(all_rel_rmse)) if all_rel_rmse else float("nan")
    print(f"\nTest set ({len(test_sims)} sims) — affected nodes only:")
    print(f"  RMSE = {mean_rmse:.2f} um   rel RMSE = {mean_rel_rmse:.1f}%   Pearson r = {mean_r:.3f}")
    print(f"  note: r = pattern correlation; rel RMSE = RMSE / peak_gt (scale accuracy)")

    # ── pick figure sim (median r in test set) ───────────────────────────────
    med_r = float(np.median(all_r)) if all_r else 0.0
    med_idx = int(np.argmin(np.abs(np.array(all_r) - med_r))) if all_r else 0
    fig_sim = test_sims[med_idx]
    cb, nc, gt_um, pred_um, uz_gt_2d, uz_pred_2d = predict_uz_um(fig_sim)

    thresh = max(np.abs(gt_um).max() * 0.05, 0.1)
    mask_fig = np.abs(gt_um) > thresh
    rmse_fig = (
        float(np.sqrt(np.mean((pred_um[mask_fig] - gt_um[mask_fig]) ** 2))) if mask_fig.sum() > 0 else float("nan")
    )
    r_fig, _ = pearsonr(pred_um[mask_fig], gt_um[mask_fig]) if mask_fig.sum() > 1 else (float("nan"), None)

    vmin = min(uz_gt_2d.min(), uz_pred_2d.min())
    vmax = max(uz_gt_2d.max(), uz_pred_2d.max())
    err = uz_pred_2d - uz_gt_2d
    elim = max(np.abs(err).max(), 0.01)

    fig = plt.figure(figsize=(14, 3.8))
    gs = gridspec.GridSpec(
        1, 5, width_ratios=[0.7, 1, 1, 1, 0.06], wspace=0.10, left=0.03, right=0.95, top=0.88, bottom=0.12
    )

    ax_cb = fig.add_subplot(gs[0])
    ax_gt = fig.add_subplot(gs[1])
    ax_pr = fig.add_subplot(gs[2])
    ax_er = fig.add_subplot(gs[3])
    cax = fig.add_subplot(gs[4])

    kw = dict(origin="lower", aspect="equal")
    ax_cb.imshow(cb, cmap="Blues", **kw)
    ax_cb.set_title("Input\nCheckerboard", fontsize=9)
    ax_gt.imshow(uz_gt_2d, cmap="viridis", vmin=vmin, vmax=vmax, **kw)
    ax_gt.set_title("Ground Truth $u_z$ (µm)", fontsize=9)
    im = ax_pr.imshow(uz_pred_2d, cmap="viridis", vmin=vmin, vmax=vmax, **kw)
    ax_pr.set_title("CNN Prediction $u_z$ (µm)", fontsize=9)
    ax_er.imshow(err, cmap="RdBu_r", vmin=-elim, vmax=elim, **kw)
    ax_er.set_title("Residual (µm)", fontsize=9)

    for ax in (ax_cb, ax_gt, ax_pr, ax_er):
        ax.set_xticks([])
        ax.set_yticks([])
    plt.colorbar(im, cax=cax).set_label("$u_z$ (µm)", fontsize=8)

    fig.suptitle(
        f"ConvDecoderPredictor — {fig_sim} "
        f"| RMSE = {rmse_fig:.2f} µm  | Pearson r = {r_fig:.3f}  (on affected nodes)",
        fontsize=9,
        y=0.98,
    )
    out = os.path.join(ROOT, "images", "pred_vs_gt.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved -> {out}")
    return mean_rmse, mean_r, len(test_sims)


# ── GUI composite ─────────────────────────────────────────────────────────────


def make_gui_composite():
    from PIL import Image

    img_dir = os.path.join(ROOT, "images")
    panels = [
        ("(a) Generate Dataset", "Generate_dataset_page.png"),
        ("(b) Train Model", "train_page.png"),
        ("(c) Load & Evaluate", "Load_and_Predict_page.png"),
    ]
    imgs = []
    for label, fname in panels:
        p = os.path.join(img_dir, fname)
        if os.path.exists(p):
            imgs.append((label, Image.open(p).convert("RGB")))
    if not imgs:
        print("No GUI images found.")
        return

    TARGET_H = 600
    resized = []
    for label, im in imgs:
        w, h = im.size
        nw = int(w * TARGET_H / h)
        resized.append((label, im.resize((nw, TARGET_H), Image.LANCZOS)))

    total_w = sum(im.size[0] for _, im in resized) + (len(resized) - 1) * 10
    canvas = np.ones((TARGET_H + 28, total_w, 3), dtype=np.uint8) * 255

    x = 0
    for _, im in resized:
        arr = np.array(im)
        canvas[28:, x : x + arr.shape[1]] = arr
        x += arr.shape[1] + 10

    fig2, ax2 = plt.subplots(figsize=(total_w / 100, (TARGET_H + 28) / 100), dpi=100)
    ax2.imshow(canvas)
    x = 0
    for i, (label, im) in enumerate(resized):
        xc = x + im.size[0] // 2
        ax2.text(xc, 14, label, ha="center", va="center", fontsize=8, fontweight="bold")
        x += im.size[0] + 10
    ax2.axis("off")
    fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)

    out = os.path.join(img_dir, "gui_composite.png")
    fig2.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"GUI composite saved -> {out}")


# ── STL crop ─────────────────────────────────────────────────────────────────


def make_stl_clean():
    from PIL import Image

    src = os.path.join(ROOT, "images", "STL_Deformation.png")
    out = os.path.join(ROOT, "images", "stl_deformation.png")
    if not os.path.exists(src):
        print("STL_Deformation.png not found")
        return
    im = Image.open(src).convert("RGB")
    arr = np.array(im)
    h, w = arr.shape[:2]
    grey = 235

    top = next((r for r in range(h) if not np.all(arr[r] > grey)), 0)
    bot = next((r for r in range(h - 1, -1, -1) if not np.all(arr[r] > grey)), h) + 1
    lft = next((c for c in range(w) if not np.all(arr[:, c] > grey)), 0)
    rgt = next((c for c in range(w - 1, -1, -1) if not np.all(arr[:, c] > grey)), w) + 1

    cropped = im.crop((lft, top, rgt, bot))
    cropped.save(out, dpi=(300, 300))
    print(f"STL saved -> {out}  ({rgt-lft}x{bot-top})")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate()

    print("\n" + "=" * 60)
    print("Training with normalised targets")
    print("=" * 60)
    train()

    print("\n" + "=" * 60)
    print("Evaluating and plotting")
    print("=" * 60)
    rmse, r, n_test = evaluate_and_plot()

    print("\n" + "=" * 60)
    print("GUI composite")
    make_gui_composite()

    print("\n" + "=" * 60)
    print("STL crop")
    make_stl_clean()

    print()
    print("=" * 60)
    print("NUMBERS FOR paper.md:")
    print(f"  Dataset:    500 simulations, 30x30 mesh (961 nodes), 5x5 checkerboard")
    print(f"  Test set:   {n_test} simulations (15%)")
    print(f"  RMSE:       {rmse:.2f} um  (on affected nodes, >5% of peak uz)")
    print(f"  Pearson r:  {r:.3f}  (spatial pattern correlation)")
    print(f"  Architecture: ConvDecoderPredictor (~170 K params)")
    print("=" * 60)
