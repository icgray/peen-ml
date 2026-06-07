#!/usr/bin/env python3
"""Retrain Model C (MatCond MultiMat, mat_dim=10) and Model E (ConvDecoder HighRes, reflect padding)."""
from __future__ import annotations
import sys, os, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src" / "peen-ml"
for _p in [str(_SRC), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model as M
from large_scale_train import train_standard, train_conv_decoder

RUN_DIR = _HERE / "LargeScaleRun1"
MODELS = RUN_DIR / "Models"

# ---- Model C: ImprovedDisplacementPredictor, mat_dim=10, 5000-sim MultiMat ----
print("\n" + "=" * 62)
print("  C_MatCond_MultiMat  (mat_dim=10, per-sim norm)")
print("=" * 62)
c_ds = str(RUN_DIR / "Dataset_MultiMat_Merged")
c_dir = str(MODELS / "C_MatCond_MultiMat")
if os.path.isdir(c_ds):
    t0 = time.perf_counter()
    res = train_standard(
        dataset_dir=c_ds,
        model_save_dir=c_dir,
        epochs=100,
        patience=20,
        batch_size=16,  # smaller for 5000-sim dataset
        use_improved=True,
        use_material=True,  # mat_dim = FULL_COND_DIM = 10
    )
    if res.success:
        print(f"  [OK]  RMSE={res.rmse_um:.2f} µm  epochs={res.epochs_trained}  " f"time={time.perf_counter()-t0:.0f}s")
    else:
        print(f"  [FAIL] {res.error}")
else:
    print(f"  SKIP — dataset not found: {c_ds}")

# ---- Model E: ConvDecoderPredictor, reflect padding, HighRes 300-sim ----
print("\n" + "=" * 62)
print("  E_ConvDecoder_HighRes  (reflect padding)")
print("=" * 62)
e_ds = str(RUN_DIR / "Dataset_HighRes_Ti_Steel_300")
e_dir = str(MODELS / "E_ConvDecoder_HighRes")
if os.path.isdir(e_ds):
    t0 = time.perf_counter()
    res = train_conv_decoder(
        dataset_dir=e_ds,
        model_save_dir=e_dir,
        epochs=60,
        patience=15,
        batch_size=8,  # smaller for 101×101 grid
        use_material=False,
    )
    if res.success:
        print(f"  [OK]  RMSE={res.rmse_um:.2f} µm  epochs={res.epochs_trained}  " f"time={time.perf_counter()-t0:.0f}s")
    else:
        print(f"  [FAIL] {res.error}")
else:
    print(f"  SKIP — dataset not found: {e_ds}")

print("\nC and E retrain complete.")
