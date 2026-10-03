"""
pipeline/refit_tls_sensitivity.py - Radiometric TLS Sensitivity Analysis

Re-classifies existing normalised feature rasters from data/ahmedabad/normalized/
using refit Total Least Squares (TLS) coefficients against 2021 reference PIFs
versus reused coefficients from radiometric_normalization_coefficients.json.
This script evaluates sensitivity of the classification to TLS refitting;
it does not rebuild satellite composites.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import rasterio

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.normalize_radiometry import (
    extract_pif_mask,
    fit_tls_regression,
    safe_normalized_difference,
)
from pipeline.train_classifier import apply_majority_filter_3x3

FEATURE_NAMES = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]
OPTICAL_BANDS = ["blue", "green", "red", "nir", "swir16"]


def run_tls_refit_sensitivity(data_dir: Path = Path("data/ahmedabad")):
    start_time = time.time()
    print("=" * 80)
    print("TLS 2021-REFERENCE REFIT VS REUSED COEFFICIENTS (AHMEDABAD 2024)")
    print("Note: Re-classifies existing normalised rasters; does not rebuild composites.")
    print("=" * 80)

    norm_dir = data_dir / "normalized"

    # 2021 Reference rasters
    ref_2021 = {}
    for b in OPTICAL_BANDS:
        with rasterio.open(norm_dir / f"ahmedabad_2021_{b}.tif") as src:
            ref_2021[b] = src.read(1).astype(np.float32)
            transform = src.transform
            h, w = src.height, src.width

    # Load 2024 composite
    coef_json_path = data_dir / "radiometric_normalization_coefficients.json"
    with open(coef_json_path, encoding="utf-8") as f:
        coef_data = json.load(f)

    reused_tls_2024 = coef_data["coefficients"]["2024"]

    # Invert to raw composite reflectances
    raw_2024 = {}
    for b in OPTICAL_BANDS:
        with rasterio.open(norm_dir / f"ahmedabad_2024_{b}.tif") as src:
            norm_arr = src.read(1).astype(np.float32)
        slope = reused_tls_2024[b]["slope"]
        intercept = reused_tls_2024[b]["intercept"]
        raw_2024[b] = (norm_arr - intercept) / slope

    # Extract PIF mask
    clean_dir = data_dir / "clean"
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    clean_stack = []
    for y in years:
        cp = clean_dir / f"ahmedabad_{y}_clean.tif"
        if cp.exists():
            with rasterio.open(cp) as src:
                clean_stack.append(src.read(1))
        else:
            with rasterio.open(data_dir / f"ahmedabad_{y}_classified.tif") as src:
                clean_stack.append(src.read(1))

    clean_stack_3d = np.stack(clean_stack, axis=0)
    pif_mask, water_pifs, built_pifs = extract_pif_mask(clean_stack_3d)
    print(
        f"PIF Mask: {np.sum(pif_mask):,} pixels (Water: {np.sum(water_pifs):,}, Built-up: {np.sum(built_pifs):,})"
    )

    # (a) Fit TLS regression against 2021 reference PIFs
    refit_coefficients = {}
    refit_norm_bands = {}
    print("\n(a) Refit 2021-Reference TLS Coefficients:")
    for b in OPTICAL_BANDS:
        x_pif = raw_2024[b][pif_mask]
        y_pif = ref_2021[b][pif_mask]
        valid_p = np.isfinite(x_pif) & np.isfinite(y_pif) & (x_pif > 0) & (y_pif > 0)
        slope, intercept, r2 = fit_tls_regression(x_pif[valid_p], y_pif[valid_p])
        refit_coefficients[b] = {"slope": slope, "intercept": intercept, "r2": r2}
        print(f"  - Band {b:<6}: slope = {slope:.6f}, intercept = {intercept:+.6f}, R² = {r2:.4f}")
        refit_norm_bands[b] = np.clip(raw_2024[b] * slope + intercept, 0.0, 1.0)

    # Compute indices for refit
    refit_norm_bands["ndvi"] = safe_normalized_difference(
        refit_norm_bands["nir"], refit_norm_bands["red"]
    )
    refit_norm_bands["ndbi"] = safe_normalized_difference(
        refit_norm_bands["swir16"], refit_norm_bands["nir"]
    )
    refit_norm_bands["mndwi"] = safe_normalized_difference(
        refit_norm_bands["green"], refit_norm_bands["swir16"]
    )

    # Classify (a) Refit
    rf = joblib.load(data_dir / "rf_model_pooled.pkl")
    stack_2d_a = np.column_stack([refit_norm_bands[feat].ravel() for feat in FEATURE_NAMES])
    valid_1d_a = np.all(np.isfinite(stack_2d_a) & (stack_2d_a != -9999.0), axis=1)

    preds_1d_a = np.zeros(stack_2d_a.shape[0], dtype=np.uint8)
    preds_1d_a[valid_1d_a] = rf.predict(stack_2d_a[valid_1d_a]).astype(np.uint8)
    raw_cls_a = preds_1d_a.reshape((h, w))
    valid_mask_a = valid_1d_a.reshape((h, w))
    filtered_cls_a = apply_majority_filter_3x3(raw_cls_a, valid_mask_a)

    px_km2 = (abs(transform.a) * abs(transform.e)) / 1e6
    built_px_a = int(np.sum((filtered_cls_a == 1) & valid_mask_a))
    built_km2_a = round(built_px_a * px_km2, 2)

    # (b) Reused stored TLS coefficients
    reused_norm_bands = {}
    print("\n(b) Reused Stored TLS Coefficients (radiometric_normalization_coefficients.json):")
    for b in OPTICAL_BANDS:
        slope = reused_tls_2024[b]["slope"]
        intercept = reused_tls_2024[b]["intercept"]
        print(f"  - Band {b:<6}: slope = {slope:.6f}, intercept = {intercept:+.6f}")
        reused_norm_bands[b] = np.clip(raw_2024[b] * slope + intercept, 0.0, 1.0)

    reused_norm_bands["ndvi"] = safe_normalized_difference(
        reused_norm_bands["nir"], reused_norm_bands["red"]
    )
    reused_norm_bands["ndbi"] = safe_normalized_difference(
        reused_norm_bands["swir16"], reused_norm_bands["nir"]
    )
    reused_norm_bands["mndwi"] = safe_normalized_difference(
        reused_norm_bands["green"], reused_norm_bands["swir16"]
    )

    stack_2d_b = np.column_stack([reused_norm_bands[feat].ravel() for feat in FEATURE_NAMES])
    valid_1d_b = np.all(np.isfinite(stack_2d_b) & (stack_2d_b != -9999.0), axis=1)

    preds_1d_b = np.zeros(stack_2d_b.shape[0], dtype=np.uint8)
    preds_1d_b[valid_1d_b] = rf.predict(stack_2d_b[valid_1d_b]).astype(np.uint8)
    raw_cls_b = preds_1d_b.reshape((h, w))
    valid_mask_b = valid_1d_b.reshape((h, w))
    filtered_cls_b = apply_majority_filter_3x3(raw_cls_b, valid_mask_b)

    built_px_b = int(np.sum((filtered_cls_b == 1) & valid_mask_b))
    built_km2_b = round(built_px_b * px_km2, 2)

    print("\n" + "=" * 80)
    print("RESULTS: BUILT-UP AREA COMPARISON")
    print("=" * 80)
    print(f"(a) With Refit TLS Coefficients : {built_km2_a:.2f} km² ({built_px_a:,} pixels)")
    print(f"(b) With Reused TLS Coefficients: {built_km2_b:.2f} km² ({built_px_b:,} pixels)")
    print(f"Difference (Refit - Reused)     : {built_km2_a - built_km2_b:+.2f} km²")
    print(f"Runtime: {round(time.time() - start_time, 4)}s")
    print("=" * 80)


if __name__ == "__main__":
    run_tls_refit_sensitivity()
