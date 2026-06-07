"""Generate final pred_vs_gt.png using the median-r test simulation."""

import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "peen-ml"))
from model import sample_field_at_coords, infer_grid_shape

DATA = os.path.join(ROOT, "Dataset_bench_final")
MDL = os.path.join(DATA, "saved_model_bench")
scale = float(np.load(MDL + "/scale.npy")[0])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MDL + "/model.pth", map_location=device, weights_only=False)
model.eval()
grid_H, grid_W = infer_grid_shape(DATA)
G = 10  # checkerboard cells

sims = sorted([d for d in os.listdir(DATA) if d.startswith("Simulation_")], key=lambda x: int(x.split("_")[1]))
test_sims = sims[425:]


def run(sn):
    p = os.path.join(DATA, sn)
    cb = np.load(p + "/checkerboard.npy")
    nc = np.load(p + "/node_coords.npy")
    gt = np.load(p + "/displacements.npy")
    cb_t = torch.tensor(cb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        field = model(cb_t) * scale
    pred = sample_field_at_coords(field, torch.tensor(nc[:, :2], dtype=torch.float32).to(device))[0].cpu().numpy()
    xs = np.unique(np.round(nc[:, 0], 8))
    ys = np.unique(np.round(nc[:, 1], 8))
    H, W = len(xs), len(ys)
    gt_um = gt[:, 2] * 1e6
    pr_um = pred[:, 2] * 1e6
    gt_2d = gt_um.reshape(H, W)
    pr_2d = pr_um.reshape(H, W)
    ch, cw = H // G, W // G
    gt_c = np.array([[gt_2d[i * ch : (i + 1) * ch, j * cw : (j + 1) * cw].mean() for j in range(G)] for i in range(G)])
    pr_c = np.array([[pr_2d[i * ch : (i + 1) * ch, j * cw : (j + 1) * cw].mean() for j in range(G)] for i in range(G)])
    r_cell, _ = pearsonr(gt_c.ravel(), pr_c.ravel())
    return cb, gt_2d, pr_2d, gt_c, pr_c, r_cell


# pick test sim with median cell-averaged correlation (honest typical-case figure)
all_r_sims = [(sn, run(sn)[-1]) for sn in test_sims]
all_r_sims.sort(key=lambda x: x[1])
med_sn, med_r = all_r_sims[len(all_r_sims) // 2]
print(f"Median sim: {med_sn}  cell-r={med_r:.3f}")

cb, gt_2d, pr_2d, gt_c, pr_c, r_cell = run(med_sn)

r_cell_v, _ = pearsonr(gt_c.ravel(), pr_c.ravel())
rmse_cell = float(np.sqrt(np.mean((gt_c - pr_c) ** 2)))

# node-level stats on affected nodes
gt_flat = gt_2d.ravel()
pr_flat = pr_2d.ravel()
thresh = max(abs(gt_flat).max() * 0.05, 0.5)
mask = abs(gt_flat) > thresh
rmse_n = float(np.sqrt(np.mean((pr_flat[mask] - gt_flat[mask]) ** 2))) if mask.sum() > 0 else float("nan")
r_n, _ = pearsonr(pr_flat[mask], gt_flat[mask]) if mask.sum() > 1 else (float("nan"), None)

vmin = min(gt_2d.min(), pr_2d.min())
vmax = max(gt_2d.max(), pr_2d.max())
vcmin = min(gt_c.min(), pr_c.min())
vcmax = max(gt_c.max(), pr_c.max())

# ── 5 panels + colorbar ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 4.0))
gs = gridspec.GridSpec(
    1, 6, width_ratios=[0.62, 1, 1, 1, 1, 0.06], wspace=0.09, left=0.02, right=0.96, top=0.87, bottom=0.10
)

ax = [fig.add_subplot(gs[i]) for i in range(6)]
kw = dict(origin="lower", aspect="equal")

ax[0].imshow(cb, cmap="Blues", **kw)
ax[0].set_title("Shot Density\nCheckerboard", fontsize=9)
ax[1].imshow(gt_2d, cmap="viridis", vmin=vmin, vmax=vmax, **kw)
ax[1].set_title("Ground Truth $u_z$ (µm)\n[node level, 31×31]", fontsize=9)
im = ax[2].imshow(pr_2d, cmap="viridis", vmin=vmin, vmax=vmax, **kw)
ax[2].set_title("CNN Prediction $u_z$ (µm)\n[node level, 31×31]", fontsize=9)
ax[3].imshow(gt_c, cmap="viridis", vmin=vcmin, vmax=vcmax, **kw)
ax[3].set_title("GT Cell-Avg $u_z$ (µm)\n[10×10 density cells]", fontsize=9)
ax[4].imshow(pr_c, cmap="viridis", vmin=vcmin, vmax=vcmax, **kw)
ax[4].set_title("Pred Cell-Avg $u_z$ (µm)\n[10×10 density cells]", fontsize=9)
plt.colorbar(im, cax=ax[5]).set_label(r"$u_z$ (µm)", fontsize=8)

for a in ax[:5]:
    a.set_xticks([])
    a.set_yticks([])

fig.suptitle(
    f"{med_sn} (median-r test sim) — "
    f"Node level: RMSE = {rmse_n:.1f} µm, r = {r_n:.3f}  |  "
    f"Cell-averaged: RMSE = {rmse_cell:.1f} µm, r = {r_cell_v:.3f}",
    fontsize=9,
    y=0.99,
)

out = os.path.join(ROOT, "images", "pred_vs_gt.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved -> {out}")

print()
print("=" * 60)
print("FINAL NUMBERS FOR paper.md:")
print(f"  Median test sim: {med_sn}")
print(f"  Node-level:      RMSE = {rmse_n:.2f} um   r = {r_n:.3f}")
print(f"  Cell-averaged:   RMSE = {rmse_cell:.2f} um   r = {r_cell_v:.3f}")
print("=" * 60)
