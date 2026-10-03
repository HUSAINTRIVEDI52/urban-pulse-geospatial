"""
reproduce_series_control.py - UrbanPulse Series Reproduction Control
Re-classifies existing normalised feature rasters from data/ahmedabad/normalized/
using the trained pooled Random Forest model and 3x3 majority filter
to verify reproduction of the operational validated series.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import rasterio

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.train_classifier import apply_majority_filter_3x3

# Classifier inputs in exact training order
FEATURE_NAMES = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]


def reproduce_series_control(data_dir: Path = Path("data/ahmedabad")):
    out_dir = data_dir / "control_rebuild"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = data_dir / "rf_model_pooled.pkl"
    if not model_path.exists():
        model_path = Path("data/rf_model_pooled.pkl")
    rf = joblib.load(model_path)

    years = [2020, 2021, 2023, 2024]
    results = {}

    print("=" * 80)
    print("REPRODUCING AHMEDABAD (2020, 2021, 2023, 2024) - CONTROL RUN")
    print("Re-classifying existing normalised feature rasters from data/ahmedabad/normalized/")
    print(f"Classifier Inputs (8 Bands): {FEATURE_NAMES}")
    print("Class Code Counted as Built-up: Class 1 (Built-up)")
    print("=" * 80)

    for yr in years:
        # Load normalized features
        norm_dir = data_dir / "normalized"
        feature_files = [norm_dir / f"ahmedabad_{yr}_{feat}.tif" for feat in FEATURE_NAMES]

        with rasterio.open(feature_files[0]) as ref_src:
            profile = ref_src.profile.copy()
            transform = ref_src.transform
            h, w = ref_src.height, ref_src.width

        feature_arrays = []
        for fpath in feature_files:
            with rasterio.open(fpath) as src:
                feature_arrays.append(src.read(1).astype(np.float32))

        stack_2d = np.column_stack([arr.ravel() for arr in feature_arrays])
        valid_1d = np.all(np.isfinite(stack_2d) & (stack_2d != -9999.0), axis=1)

        preds_1d = np.zeros(stack_2d.shape[0], dtype=np.uint8)
        if np.any(valid_1d):
            preds_1d[valid_1d] = rf.predict(stack_2d[valid_1d]).astype(np.uint8)

        raw_cls = preds_1d.reshape((h, w))
        valid_mask = valid_1d.reshape((h, w))

        # 3x3 majority filter
        filtered_cls = apply_majority_filter_3x3(raw_cls, valid_mask)

        # Write output raster
        out_tif = out_dir / f"ahmedabad_{yr}_control_rebuild.tif"
        profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="lzw")
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(filtered_cls, 1)

        # Pixel size and area
        px_km2 = (abs(transform.a) * abs(transform.e)) / 1e6
        built_px = int(np.sum((filtered_cls == 1) & valid_mask))
        built_km2 = round(built_px * px_km2, 2)

        results[yr] = {
            "builtup_km2": built_km2,
            "builtup_pixels": built_px,
            "raster_path": str(out_tif.resolve()),
        }
        print(f"Year {yr}: {built_km2} km2 ({built_px} pixels) -> {out_tif.name}")

    net_growth_km2 = round(results[2024]["builtup_km2"] - results[2020]["builtup_km2"], 2)
    net_growth_pct = round((net_growth_km2 / results[2020]["builtup_km2"]) * 100.0, 2)
    print(f"\nNet Growth (2020 -> 2024): +{net_growth_km2} km2 (+{net_growth_pct}%)")

    results_file = out_dir / "rebuild_summary.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved summary to: {results_file.resolve()}")
    return results


if __name__ == "__main__":
    reproduce_series_control()
