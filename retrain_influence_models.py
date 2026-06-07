#!/usr/bin/env python3
"""Retrain all five I-variant InfluenceField models with per-sim normalization."""
from __future__ import annotations
import sys, os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE / "src" / "peen-ml"
for _p in [str(_SRC), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model as M

RUN_DIR = _HERE / "LargeScaleRun1"

VARIANTS = [
    ("I_InfluenceField_Ti_6Al_4V__steel",    "Dataset_Ti_6Al_4V__steel_200"),
    ("I_InfluenceField_316L_SS__ceramic",    "Dataset_316L_SS__ceramic_200"),
    ("I_InfluenceField_Inconel_718__tungsten","Dataset_Inconel_718__tungsten_200"),
    ("I_InfluenceField_Al_7075_T6__glass",   "Dataset_Al_7075_T6__glass_200"),
    ("I_InfluenceField_4340_Steel__cast_iron","Dataset_4340_Steel__cast_iron_200"),
]

# Map old directory names (as used in Models/) to the new training runs
MODEL_DIR_NAMES = {
    "I_InfluenceField_Ti_6Al_4V__steel":     "I_InfluenceField_Ti_Steel",
    "I_InfluenceField_316L_SS__ceramic":     "I_InfluenceField_316L_ceramic",
    "I_InfluenceField_Inconel_718__tungsten":"I_InfluenceField_Inconel_tungsten",
    "I_InfluenceField_Al_7075_T6__glass":    "I_InfluenceField_Al_glass",
    "I_InfluenceField_4340_Steel__cast_iron":"I_InfluenceField_4340_cast_iron",
}

for variant_key, dataset_rel in VARIANTS:
    model_dir_name = MODEL_DIR_NAMES[variant_key]
    dataset_dir    = str(RUN_DIR / dataset_rel)
    model_save_dir = str(RUN_DIR / "Models" / model_dir_name)

    print(f"\n{'='*60}")
    print(f"  {model_dir_name}")
    print(f"  dataset : {dataset_rel}")
    print(f"  save to : {model_save_dir}")
    print(f"{'='*60}")

    if not os.path.isdir(dataset_dir):
        print(f"  SKIP — dataset not found: {dataset_dir}")
        continue

    result = M.train_influence_field_model(
        dataset_dir    = dataset_dir,
        model_save_dir = model_save_dir,
        epochs         = 120,
        patience       = 20,
    )

    if result.get("success"):
        print(f"  [OK]  RMSE={result['rmse_um']:.2f} µm  "
              f"epochs={result['epochs_trained']}  "
              f"disp_scale={result['disp_scale']:.3e}")
    else:
        print(f"  [FAIL] {result.get('error', 'unknown error')}")

print("\nAll I variants retrained.")
