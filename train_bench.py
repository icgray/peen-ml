"""Train ConvDecoder on Dataset_bench_final and produce pred_vs_gt.png."""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "peen-ml"))
from model import ConvDecoderPredictor, FieldDataset, infer_grid_shape, sample_field_at_coords

DATA = os.path.join(ROOT, "Dataset_bench_final")
MDL = os.path.join(DATA, "saved_model_bench")
os.makedirs(MDL, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
sims = sorted([d for d in os.listdir(DATA) if d.startswith("Simulation_")], key=lambda x: int(x.split("_")[1]))
print(f"{len(sims)} simulations")
cbs, disps = [], []
for s in sims:
    p = os.path.join(DATA, s)
    cbs.append(np.load(p + "/checkerboard.npy"))
    disps.append(np.load(p + "/displacements.npy"))
cbs = np.stack(cbs)
disps = np.stack(disps)

scale = float(np.percentile(np.abs(disps), 99.9))
print(f"Displacement scale (99.9th pct): {scale*1e6:.1f} um")
np.save(os.path.join(MDL, "scale.npy"), np.array([scale]))
disps_n = disps / scale

grid_H, grid_W = infer_grid_shape(DATA)
print(f"Grid: {grid_H}x{grid_W}")

ds = FieldDataset(cbs, disps_n, grid_H, grid_W)
n = len(ds)
ntr, nva = int(0.70 * n), int(0.15 * n)
nte = n - ntr - nva
torch.manual_seed(0)
tr_ds, va_ds, te_ds = random_split(ds, [ntr, nva, nte])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin = device.type == "cuda"
tr_ld = DataLoader(tr_ds, 32, shuffle=True, num_workers=0, pin_memory=pin)
va_ld = DataLoader(va_ds, 32, shuffle=False, num_workers=0, pin_memory=pin)

# ── train ─────────────────────────────────────────────────────────────────────
model = ConvDecoderPredictor(input_channels=1, out_H=grid_H, out_W=grid_W).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}  device={device}")
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-6)
lf = nn.MSELoss()

best, pat, PAT = 1e9, 0, 20
t0 = time.time()
for ep in range(1, 201):
    model.train()
    tl = 0.0
    for cb_b, d_b in tr_ld:
        opt.zero_grad()
        loss = lf(model(cb_b.to(device)), d_b.to(device))
        loss.backward()
        opt.step()
        tl += loss.item()
    tl /= len(tr_ld)

    model.eval()
    vl = 0.0
    with torch.no_grad():
        for cb_b, d_b in va_ld:
            vl += lf(model(cb_b.to(device)), d_b.to(device)).item()
    vl /= len(va_ld)
    sch.step()

    if ep % 10 == 0 or ep <= 5:
        lr = opt.param_groups[0]["lr"]
        print(f"ep {ep:3d}  tr={tl:.3e}  va={vl:.3e}  lr={lr:.1e}")

    if vl < best:
        best = vl
        pat = 0
        torch.save(model, os.path.join(MDL, "model.pth"))
    else:
        pat += 1
        if pat >= PAT:
            print(f"Early stop ep {ep}  best_val={best:.3e}")
            break

print(f"Done {time.time()-t0:.0f}s  best_val={best:.3e}")

# ── evaluate ──────────────────────────────────────────────────────────────────
model = torch.load(os.path.join(MDL, "model.pth"), map_location=device, weights_only=False)
model.eval()

test_sims = sims[ntr + nva :]
all_rmse, all_r = [], []


def run_sim(sim_name):
    p = os.path.join(DATA, sim_name)
    cb = np.load(p + "/checkerboard.npy")
    nc = np.load(p + "/node_coords.npy")
    gt = np.load(p + "/displacements.npy")
    cb_t = torch.tensor(cb, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        field = model(cb_t) * scale
    nc_t = torch.tensor(nc[:, :2], dtype=torch.float32).to(device)
    pred = sample_field_at_coords(field, nc_t)[0].cpu().numpy()
    return cb, nc, gt[:, 2] * 1e6, pred[:, 2] * 1e6


for sn in test_sims:
    cb, nc, gt_um, pr_um = run_sim(sn)
    thresh = max(abs(gt_um).max() * 0.05, 0.5)
    mask = abs(gt_um) > thresh
    if mask.sum() < 10:
        continue
    rmse = float(np.sqrt(np.mean((pr_um[mask] - gt_um[mask]) ** 2)))
    r, _ = pearsonr(pr_um[mask], gt_um[mask])
    all_rmse.append(rmse)
    all_r.append(r)

mean_rmse = float(np.mean(all_rmse))
mean_r = float(np.mean(all_r))
print(f"\nTest set ({len(test_sims)} sims): RMSE={mean_rmse:.2f} um  Pearson r={mean_r:.3f}")

# ── figure: pick the median-r test sim ───────────────────────────────────────
valid_idx = [
    i
    for i, sn in enumerate(test_sims)
    if abs(np.load(os.path.join(DATA, sn, "displacements.npy"))[:, 2]).max() * 1e6 > 5
]
med_idx = valid_idx[len(valid_idx) // 2]
fig_sim = test_sims[med_idx]

cb, nc, gt_um, pr_um = run_sim(fig_sim)
xs = np.unique(np.round(nc[:, 0], 8))
ys = np.unique(np.round(nc[:, 1], 8))
H, W = len(xs), len(ys)
gt_2d = gt_um.reshape(H, W)
pr_2d = pr_um.reshape(H, W)
er_2d = pr_2d - gt_2d

thresh_fig = max(abs(gt_um).max() * 0.05, 0.5)
mask_fig = abs(gt_um) > thresh_fig
rmse_fig = float(np.sqrt(np.mean((pr_um[mask_fig] - gt_um[mask_fig]) ** 2))) if mask_fig.sum() > 0 else float("nan")
r_fig, _ = pearsonr(pr_um[mask_fig], gt_um[mask_fig]) if mask_fig.sum() > 1 else (float("nan"), None)

vmin = min(gt_2d.min(), pr_2d.min())
vmax = max(gt_2d.max(), pr_2d.max())
elim = max(abs(er_2d).max(), 0.01)

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
ax_gt.imshow(gt_2d, cmap="viridis", vmin=vmin, vmax=vmax, **kw)
ax_gt.set_title("Ground Truth $u_z$ (µm)", fontsize=9)
im = ax_pr.imshow(pr_2d, cmap="viridis", vmin=vmin, vmax=vmax, **kw)
ax_pr.set_title("CNN Prediction $u_z$ (µm)", fontsize=9)
ax_er.imshow(er_2d, cmap="RdBu_r", vmin=-elim, vmax=elim, **kw)
ax_er.set_title("Residual (µm)", fontsize=9)

for ax in (ax_cb, ax_gt, ax_pr, ax_er):
    ax.set_xticks([])
    ax.set_yticks([])
plt.colorbar(im, cax=cax).set_label("$u_z$ (µm)", fontsize=8)

fig.suptitle(
    f"ConvDecoderPredictor — {fig_sim} " f"| RMSE = {rmse_fig:.2f} µm  | Pearson r = {r_fig:.3f}",
    fontsize=9,
    y=0.98,
)

out = os.path.join(ROOT, "images", "pred_vs_gt.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Figure -> {out}")

print()
print("=" * 60)
print("NUMBERS FOR paper.md:")
print("  Dataset:   500 sims, 30x30 mesh (961 nodes), 10x10 checkerboard")
print(f"  Train/Val/Test: {ntr}/{nva}/{nte}")
print(f"  Test RMSE: {mean_rmse:.2f} um  (affected nodes >5pct of peak)")
print(f"  Pearson r: {mean_r:.3f}")
print("  Physics: Shen-Atluri model, steel shot, 300-1000 shots/sim")
print("=" * 60)
