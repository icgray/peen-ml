"""Generate images/pipeline_overview.png for the JOSS paper."""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ── layout constants ────────────────────────────────────────────────────────
FIG_W, FIG_H = 14, 4.6
DPI = 300

# x-centres of the five main boxes (evenly spaced, room for arrows between)
XC = [1.1, 3.5, 6.0, 8.5, 11.0]
YC = 2.9  # top row y-centre
YC2 = 1.2  # sub-label row y-centre (grey detail text)

BOX_W = 1.7  # box width
BOX_H = 1.05  # box height
RADIUS = 0.12  # rounded corner radius
ARROW_Y = YC  # arrow y-level

# ── colour palette (viridis-inspired, colourblind-safe) ─────────────────────
C_INPUT = "#4d9de0"  # blue  — inputs
C_SIM = "#e15554"  # red   — simulator
C_DATA = "#f4a261"  # amber — dataset
C_MODEL = "#3bb273"  # green — model/training
C_OUT = "#7768ae"  # purple— output

COLORS = [C_INPUT, C_SIM, C_DATA, C_MODEL, C_OUT]

# ── box labels (title + subtitle) ───────────────────────────────────────────
BOXES = [
    ("Shot\nParameters", "diameter · velocity\nmaterial · coverage"),
    ("Physics\nSimulator", "Shen & Atluri (2006)\n~2 s / simulation"),
    ("Training\nDataset", "checkerboard.npy\ndisplacements.npy"),
    ("CNN\nTraining", "ConvDecoder · FC\nSIREN · material cond."),
    ("Displacement\nField", "flat plate · STL surface\n< 1 s inference"),
]

# ── arrow labels ────────────────────────────────────────────────────────────
ARROW_LABELS = [
    "shot pattern\nG×G grid",
    "N simulations\nparallel CPU",
    "train / val / test\n70 / 15 / 15 %",
    "trained model\n(.pth)",
]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")

# ── draw boxes ──────────────────────────────────────────────────────────────
for i, (xc, (title, sub), col) in enumerate(zip(XC, BOXES, COLORS)):
    x0 = xc - BOX_W / 2
    y0 = YC - BOX_H / 2

    fancy = FancyBboxPatch(
        (x0, y0),
        BOX_W,
        BOX_H,
        boxstyle=f"round,pad=0,rounding_size={RADIUS}",
        linewidth=1.4,
        edgecolor="white",
        facecolor=col,
        zorder=3,
    )
    ax.add_patch(fancy)

    # title text
    ax.text(
        xc,
        YC + 0.18,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
        zorder=4,
        linespacing=1.35,
    )

    # sub-text box (lighter background strip at bottom of box)
    sub_h = 0.42
    sub_patch = FancyBboxPatch(
        (x0 + 0.04, y0 + 0.04),
        BOX_W - 0.08,
        sub_h,
        boxstyle=f"round,pad=0,rounding_size={RADIUS * 0.6}",
        linewidth=0,
        facecolor=(0, 0, 0, 0.18),
        zorder=4,
    )
    ax.add_patch(sub_patch)
    ax.text(
        xc, y0 + 0.04 + sub_h / 2, sub, ha="center", va="center", fontsize=7.5, color="white", zorder=5, linespacing=1.3
    )

# ── draw arrows ─────────────────────────────────────────────────────────────
for i, lbl in enumerate(ARROW_LABELS):
    x_start = XC[i] + BOX_W / 2 + 0.04
    x_end = XC[i + 1] - BOX_W / 2 - 0.04

    ax.annotate(
        "",
        xy=(x_end, ARROW_Y),
        xytext=(x_start, ARROW_Y),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#444444",
            lw=1.8,
            mutation_scale=16,
        ),
        zorder=2,
    )

    xm = (x_start + x_end) / 2
    ax.text(xm, ARROW_Y - 0.58, lbl, ha="center", va="top", fontsize=7, color="#555555", linespacing=1.25)

# ── title banner ─────────────────────────────────────────────────────────────
ax.text(
    FIG_W / 2,
    FIG_H - 0.32,
    "peen-ml  —  Shot Peening Surrogate Modeling Pipeline",
    ha="center",
    va="top",
    fontsize=13,
    fontweight="bold",
    color="#222222",
)

# ── bottom note ──────────────────────────────────────────────────────────────
ax.text(
    FIG_W / 2,
    0.18,
    "Three CNN architectures: ConvDecoderPredictor (170 K params, recommended)  ·  "
    "DisplacementPredictor (30 M params)  ·  SIRENPredictor (2 M params, memory-safe)",
    ha="center",
    va="bottom",
    fontsize=7.5,
    color="#666666",
    style="italic",
)

os.makedirs("images", exist_ok=True)
out = os.path.join("images", "pipeline_overview.png")
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved -> {out}")
