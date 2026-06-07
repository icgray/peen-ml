"""
Generate all remaining paper figures:
  images/pred_vs_gt.png    - ground truth vs CNN prediction side-by-side
  images/gui_composite.png - three GUI panels stitched
  images/stl_deformation.png - STL plot cropped of window chrome
"""

import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "peen-ml"))

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PRED vs GT figure
# ─────────────────────────────────────────────────────────────────────────────


def make_pred_vs_gt():
    import torch

    DATA = os.path.join(ROOT, "Dataset_Gaussian")
    MODEL = os.path.join(DATA, "saved_model_conv", "trained_conv_decoder_full_model.pth")

    # pick a simulation that was likely in the test split (last ~30 of 200)
    SIM_ID = 185
    SIM = os.path.join(DATA, f"Simulation_{SIM_ID}")

    print(f"Loading simulation {SIM_ID} ...")
    cb = np.load(os.path.join(SIM, "checkerboard.npy"))  # (G, G)
    gt = np.load(os.path.join(SIM, "displacements.npy"))  # (N, 3)
    nc = np.load(os.path.join(SIM, "node_coords.npy"))  # (N, 3)

    device = torch.device("cpu")
    print("Loading model ...")
    model = torch.load(MODEL, map_location=device, weights_only=False)
    model.eval()

    cb_t = torch.tensor(cb, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        field = model(cb_t)  # (1, 3, H, W)

    # sample field at node coords
    from model import sample_field_at_coords

    nc_t = torch.tensor(nc[:, :2], dtype=torch.float32)
    pred = sample_field_at_coords(field, nc_t)[0].numpy()  # (N, 3)

    # ── reshape flat arrays to 2D grid ──────────────────────────────────────
    xs = np.unique(np.round(nc[:, 0], 8))
    ys = np.unique(np.round(nc[:, 1], 8))
    H, W = len(xs), len(ys)
    print(f"Grid: {H}x{W}, N={len(nc)}")

    uz_gt = gt[:, 2].reshape(H, W) * 1e3  # m → mm
    uz_pred = pred[:, 2].reshape(H, W) * 1e3

    # shared colour scale (clip to 2–98 percentile so outliers don't dominate)
    vmin = np.percentile(uz_gt, 2)
    vmax = np.percentile(uz_gt, 98)

    error = uz_pred - uz_gt
    mse = float(np.mean(error**2))
    smape = float(np.mean(2 * np.abs(error) / (np.abs(uz_gt) + np.abs(uz_pred) + 1e-9)) * 100)

    print(f"Test sample MSE={mse:.4e} mm^2   sMAPE={smape:.2f}%")

    # ── plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 4.2))
    gs = gridspec.GridSpec(
        1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.08, left=0.05, right=0.93, top=0.88, bottom=0.12
    )

    kw_img = dict(cmap="viridis", vmin=vmin, vmax=vmax, origin="lower", aspect="equal")
    kw_err = dict(
        cmap="RdBu_r",
        vmin=-np.percentile(np.abs(error), 98),
        vmax=np.percentile(np.abs(error), 98),
        origin="lower",
        aspect="equal",
    )

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    cax = fig.add_subplot(gs[3])

    ax0.imshow(uz_gt, **kw_img)
    ax0.set_title("Ground Truth $u_z$ (mm)", fontsize=10)
    im = ax1.imshow(uz_pred, **kw_img)
    ax1.set_title("CNN Prediction $u_z$ (mm)", fontsize=10)
    ax2.imshow(error, **kw_err)
    ax2.set_title("Error (Pred − GT, mm)", fontsize=10)

    for ax in (ax0, ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    cb_bar = plt.colorbar(im, cax=cax)
    cb_bar.set_label("$u_z$ (mm)", fontsize=9)

    fig.suptitle(
        f"ConvDecoderPredictor on Simulation {SIM_ID}  " f"|  MSE = {mse:.2e} mm²  |  sMAPE = {smape:.1f}%",
        fontsize=10,
        y=0.97,
    )

    out = os.path.join(ROOT, "images", "pred_vs_gt.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved -> {out}")
    return mse, smape


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GUI composite
# ─────────────────────────────────────────────────────────────────────────────


def make_gui_composite():
    img_dir = os.path.join(ROOT, "images")

    panels = [
        ("Generate Dataset", "Generate_dataset_page.png"),
        ("Train Model", "train_page.png"),
        ("Load & Evaluate", "Load_and_Predict_page.png"),
    ]

    imgs = []
    for label, fname in panels:
        path = os.path.join(img_dir, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {fname} not found, skipping")
            continue
        im = Image.open(path).convert("RGB")
        imgs.append((label, im))

    if not imgs:
        print("No GUI images found.")
        return

    TARGET_H = 600  # resize all to this height
    padded = []
    for label, im in imgs:
        w, h = im.size
        new_w = int(w * TARGET_H / h)
        resized = im.resize((new_w, TARGET_H), Image.LANCZOS)
        padded.append((label, resized))

    total_w = sum(im.size[0] for _, im in padded) + (len(padded) - 1) * 8
    total_h = TARGET_H + 30  # 30 px header for label
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    x = 0
    for label, im in padded:
        canvas.paste(im, (x, 30))
        x += im.size[0] + 8

    # draw labels using matplotlib (avoids ImageDraw font issues)
    fig2, ax2 = plt.subplots(figsize=(total_w / 100, total_h / 100), dpi=100)
    ax2.imshow(np.array(canvas))
    for i, (label, im) in enumerate(padded):
        xpos = sum(p.size[0] for _, p in padded[:i]) + i * 8 + im.size[0] // 2
        ax2.text(xpos, 15, f"({chr(97+i)}) {label}", ha="center", va="center", fontsize=8, fontweight="bold")
    ax2.axis("off")
    fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)

    out = os.path.join(img_dir, "gui_composite.png")
    fig2.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  STL deformation — crop window chrome
# ─────────────────────────────────────────────────────────────────────────────


def make_stl_clean():
    src = os.path.join(ROOT, "images", "STL_Deformation.png")
    out = os.path.join(ROOT, "images", "stl_deformation.png")

    if not os.path.exists(src):
        print("STL_Deformation.png not found, skipping")
        return

    im = Image.open(src).convert("RGB")
    w, h = im.size
    arr = np.array(im)

    # find rows/cols that are NOT the window-chrome grey (≈ rgb 240,240,240)
    # chrome is the top title bar and bottom toolbar — both are uniform light grey
    grey_thresh = 235

    def is_chrome_row(row):
        return np.all(row > grey_thresh)

    def is_chrome_col(col):
        return np.all(col > grey_thresh)

    # scan from top to find first non-chrome row
    top = 0
    for r in range(h):
        if not is_chrome_row(arr[r]):
            top = r
            break

    # scan from bottom
    bot = h
    for r in range(h - 1, -1, -1):
        if not is_chrome_row(arr[r]):
            bot = r + 1
            break

    # scan left / right
    left = 0
    for c in range(w):
        if not is_chrome_col(arr[:, c]):
            left = c
            break

    right = w
    for c in range(w - 1, -1, -1):
        if not is_chrome_col(arr[:, c]):
            right = c + 1
            break

    cropped = im.crop((left, top, right, bot))
    cropped.save(out, dpi=(300, 300))
    print(f"STL cropped {w}x{h} -> {right-left}x{bot-top}")
    print(f"Saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Figure 1: pred_vs_gt.png")
    print("=" * 60)
    mse, smape = make_pred_vs_gt()

    print()
    print("=" * 60)
    print("Figure 2: gui_composite.png")
    print("=" * 60)
    make_gui_composite()

    print()
    print("=" * 60)
    print("Figure 3: stl_deformation.png")
    print("=" * 60)
    make_stl_clean()

    print()
    print("=" * 60)
    print("BENCHMARK NUMBERS (fill into paper.md):")
    print(f"  ConvDecoderPredictor test sample MSE  = {mse:.4e} mm^2")
    print(f"  ConvDecoderPredictor test sample sMAPE = {smape:.2f}%")
    print("=" * 60)
