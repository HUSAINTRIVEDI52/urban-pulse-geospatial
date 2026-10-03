from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import binary_erosion
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path("f:/gis-project/UrbanPulse")
DATA_DIR = PROJECT_ROOT / "data"

PROJECT_CLASS_NAMES = {
    1: "Built-up",
    2: "Vegetation",
    3: "Water",
    4: "Agriculture",
    5: "Open land",
}
FEATURE_NAMES = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]
OPTICAL_BANDS = ["blue", "green", "red", "nir", "swir16"]
YEARS = list(range(2018, 2025))
REF_YEAR = 2021

def safe_norm_diff(a, b):
    denom = a + b
    valid = (np.abs(denom) > 1e-5) & np.isfinite(a) & np.isfinite(b)
    res = np.full_like(a, np.nan, dtype=np.float32)
    np.divide(a - b, denom, out=res, where=valid)
    return np.clip(res, -1.0, 1.0)

def apply_majority_filter_3x3(arr, valid_mask):
    h, w = arr.shape
    padded = np.pad(arr, 1, mode="edge")
    neighbors = np.stack([
        padded[0:h, 0:w], padded[0:h, 1:w+1], padded[0:h, 2:w+2],
        padded[1:h+1, 0:w], padded[1:h+1, 1:w+1], padded[1:h+1, 2:w+2],
        padded[2:h+2, 0:w], padded[2:h+2, 1:w+1], padded[2:h+2, 2:w+2],
    ], axis=0)
    class_votes = np.stack([(neighbors == c).sum(axis=0) for c in range(1, 6)], axis=0)
    majority_class = (np.argmax(class_votes, axis=0) + 1).astype(np.uint8)
    return np.where(valid_mask, majority_class, 0).astype(np.uint8)

for city in ["ahmedabad", "pune"]:
    print("\n" + "=" * 95)
    print(f"COMPLETE PIPELINE EVALUATION FOR: {city.upper()}")
    print("=" * 95)

    clean_dir = DATA_DIR / city / "clean"
    norm_dir = DATA_DIR / city / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    clean_rasters = []
    profiles = []
    for y in YEARS:
        p = clean_dir / f"{city}_{y}_classified.tif"
        if not p.exists():
            p = DATA_DIR / f"{city}_{y}_classified.tif"
        with rasterio.open(p) as src:
            clean_rasters.append(src.read(1))
            profiles.append(src.profile.copy())

    stack_clean = np.stack(clean_rasters, axis=0) # (7, H, W)
    transform = profiles[0]["transform"]
    raster_crs = profiles[0]["crs"]
    px_km2 = (abs(transform.a) * abs(transform.e)) / 1e6
    total_aoi_km2 = (stack_clean[0].shape[0] * stack_clean[0].shape[1]) * px_km2

    # 1. PIFs
    water_pif = np.sum(stack_clean == 3, axis=0) >= 6
    built_pif = np.sum(stack_clean == 1, axis=0) >= 6
    struct_3x3 = np.ones((3, 3), dtype=bool)
    water_eroded = binary_erosion(water_pif, structure=struct_3x3)
    built_eroded = binary_erosion(built_pif, structure=struct_3x3)
    pif_mask = water_eroded | built_eroded

    # 2. Fit Normalization & Create Normalized Composites
    ref_bands = {}
    for b in OPTICAL_BANDS:
        with rasterio.open(DATA_DIR / f"{city}_{REF_YEAR}_{b}.tif") as src:
            ref_bands[b] = src.read(1).astype(np.float32)

    norm_coeffs = {}
    norm_rasters = {y: {} for y in YEARS}

    for y in YEARS:
        norm_coeffs[y] = {}
        for b in OPTICAL_BANDS:
            with rasterio.open(DATA_DIR / f"{city}_{y}_{b}.tif") as src:
                raw_b = src.read(1).astype(np.float32)

            x_vals = raw_b[pif_mask]
            y_vals = ref_bands[b][pif_mask]
            valid = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals != -9999.0) & (y_vals != -9999.0)

            if y == REF_YEAR:
                slope, intercept, r2 = 1.0, 0.0, 1.0
            else:
                reg = LinearRegression().fit(x_vals[valid].reshape(-1, 1), y_vals[valid])
                slope = float(reg.coef_[0])
                intercept = float(reg.intercept_)
                r2 = float(reg.score(x_vals[valid].reshape(-1, 1), y_vals[valid]))

            norm_b = np.where(raw_b != -9999.0, np.clip(slope * raw_b + intercept, 0.0, 1.5), -9999.0)
            norm_rasters[y][b] = norm_b
            norm_coeffs[y][b] = {"slope": slope, "intercept": intercept, "r2": r2}

        # Recompute indices on normalized bands
        red = norm_rasters[y]["red"]
        green = norm_rasters[y]["green"]
        nir = norm_rasters[y]["nir"]
        swir16 = norm_rasters[y]["swir16"]

        norm_rasters[y]["ndvi"] = safe_norm_diff(nir, red)
        norm_rasters[y]["ndbi"] = safe_norm_diff(swir16, nir)
        norm_rasters[y]["mndwi"] = safe_norm_diff(green, swir16)

    # 3. Sample Stable Points across all 7 years
    # Label is stable if clean_series is the same for all 7 years
    stable_mask = (stack_clean == stack_clean[0:1]).all(axis=0) & (stack_clean[0] > 0)
    # Exclude edges with erosion
    stable_eroded = np.zeros_like(stable_mask)
    for cid in range(1, 6):
        c_mask = stable_mask & (stack_clean[0] == cid)
        c_eroded = binary_erosion(c_mask, structure=struct_3x3)
        stable_eroded |= (c_eroded if np.sum(c_eroded) >= 50 else c_mask)

    rng = np.random.default_rng(42)
    samples_per_class = 300
    sampled_records = []

    for cid in range(1, 6):
        c_mask = stable_eroded & (stack_clean[0] == cid)
        rows, cols = np.where(c_mask)
        n_avail = len(rows)
        n_sel = min(samples_per_class, n_avail)
        sel_idx = rng.choice(n_avail, size=n_sel, replace=False)
        sel_r = rows[sel_idx]
        sel_c = cols[sel_idx]
        xs, ys = rasterio.transform.xy(transform, sel_r, sel_c)
        for i in range(n_sel):
            sampled_records.append({
                "class_id": cid,
                "class_name": PROJECT_CLASS_NAMES[cid],
                "geometry": Point(xs[i], ys[i]),
                "raster_row": int(sel_r[i]),
                "raster_col": int(sel_c[i]),
            })

    gdf_base = gpd.GeoDataFrame(sampled_records, crs=raster_crs)

    # Spatial block split
    n_blocks = 8
    minx, miny, maxx, maxy = gdf_base.total_bounds
    x_bins = np.linspace(minx, maxx, n_blocks + 1)
    y_bins = np.linspace(miny, maxy, n_blocks + 1)
    bx = np.clip(np.digitize(gdf_base.geometry.x, x_bins) - 1, 0, n_blocks - 1)
    by = np.clip(np.digitize(gdf_base.geometry.y, y_bins) - 1, 0, n_blocks - 1)
    gdf_base["block_id"] = bx * n_blocks + by

    gss = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    tr_idx, te_idx = next(gss.split(gdf_base, groups=gdf_base["block_id"]))

    tr_base = gdf_base.iloc[tr_idx].copy().reset_index(drop=True)
    te_base = gdf_base.iloc[te_idx].copy().reset_index(drop=True)

    print(f"Stable base points sampled: {len(gdf_base)} (Train: {len(tr_base)}, Test: {len(te_base)})")

    # Extract features for ALL 7 years from NORMALIZED rasters
    tr_norm_pooled = []
    te_norm_pooled = []

    for y in YEARS:
        # Extract at row, col
        for subset, pooled_list in [(tr_base, tr_norm_pooled), (te_base, te_norm_pooled)]:
            sub_df = subset.copy()
            sub_df["year"] = y
            rows = sub_df["raster_row"].values
            cols = sub_df["raster_col"].values
            for f in FEATURE_NAMES:
                sub_df[f] = norm_rasters[y][f][rows, cols]
            # filter valid
            valid_m = np.all(np.isfinite(sub_df[FEATURE_NAMES].values) & (sub_df[FEATURE_NAMES].values != -9999.0), axis=1)
            pooled_list.append(sub_df[valid_m].copy())

    gdf_tr_norm = pd.concat(tr_norm_pooled, ignore_index=True)
    gdf_te_norm = pd.concat(te_norm_pooled, ignore_index=True)

    # Also extract RAW un-normalized features for baseline comparison on same stable points
    tr_raw_pooled = []
    te_raw_pooled = []
    for y in YEARS:
        for subset, pooled_list in [(tr_base, tr_raw_pooled), (te_base, te_raw_pooled)]:
            sub_df = subset.copy()
            sub_df["year"] = y
            rows = sub_df["raster_row"].values
            cols = sub_df["raster_col"].values
            for f in FEATURE_NAMES:
                with rasterio.open(DATA_DIR / f"{city}_{y}_{f}.tif") as src:
                    sub_df[f] = src.read(1)[rows, cols]
            valid_m = np.all(np.isfinite(sub_df[FEATURE_NAMES].values) & (sub_df[FEATURE_NAMES].values != -9999.0), axis=1)
            pooled_list.append(sub_df[valid_m].copy())
    gdf_tr_raw = pd.concat(tr_raw_pooled, ignore_index=True)
    gdf_te_raw = pd.concat(te_raw_pooled, ignore_index=True)

    # 4. LOYO Validation: Before vs After for 2018, 2021, 2024
    print("\n[*] LEAVE-ONE-YEAR-OUT (LOYO) COMPARISON: BEFORE VS AFTER NORMALISATION")
    print(f"{'Year':<6} | {'Stage':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Area Bias (km²)':<16} | {'True Built %':<12} | {'Pred Built %'}")
    print("-" * 95)

    for holdout_yr in [2018, 2021, 2024]:
        # BEFORE (Raw)
        tr_raw_sub = gdf_tr_raw[gdf_tr_raw["year"] != holdout_yr]
        te_raw_sub = gdf_te_raw[gdf_te_raw["year"] == holdout_yr]
        rf_raw = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_raw.fit(tr_raw_sub[FEATURE_NAMES].values, tr_raw_sub["class_id"].values)
        y_pred_raw = rf_raw.predict(te_raw_sub[FEATURE_NAMES].values)

        y_true_raw_bin = (te_raw_sub["class_id"].values == 1).astype(int)
        y_pred_raw_bin = (y_pred_raw == 1).astype(int)

        p_raw = precision_score(y_true_raw_bin, y_pred_raw_bin, zero_division=0)
        r_raw = recall_score(y_true_raw_bin, y_pred_raw_bin, zero_division=0)
        f_raw = f1_score(y_true_raw_bin, y_pred_raw_bin, zero_division=0)
        bias_raw = (np.mean(y_pred_raw_bin) - np.mean(y_true_raw_bin)) * total_aoi_km2

        # AFTER (Norm)
        tr_norm_sub = gdf_tr_norm[gdf_tr_norm["year"] != holdout_yr]
        te_norm_sub = gdf_te_norm[gdf_te_norm["year"] == holdout_yr]
        rf_norm = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_norm.fit(tr_norm_sub[FEATURE_NAMES].values, tr_norm_sub["class_id"].values)
        y_pred_norm = rf_norm.predict(te_norm_sub[FEATURE_NAMES].values)

        y_true_norm_bin = (te_norm_sub["class_id"].values == 1).astype(int)
        y_pred_norm_bin = (y_pred_norm == 1).astype(int)

        p_norm = precision_score(y_true_norm_bin, y_pred_norm_bin, zero_division=0)
        r_norm = recall_score(y_true_norm_bin, y_pred_norm_bin, zero_division=0)
        f_norm = f1_score(y_true_norm_bin, y_pred_norm_bin, zero_division=0)
        bias_norm = (np.mean(y_pred_norm_bin) - np.mean(y_true_norm_bin)) * total_aoi_km2

        print(f"{holdout_yr:<6} | {'BEFORE':<8} | {p_raw:<10.4f} | {r_raw:<10.4f} | {f_raw:<10.4f} | {bias_raw:>+14.2f} km² | {np.mean(y_true_raw_bin)*100:>10.2f}% | {np.mean(y_pred_raw_bin)*100:>10.2f}%")
        print(f"{holdout_yr:<6} | {'AFTER':<8} | {p_norm:<10.4f} | {r_norm:<10.4f} | {f_norm:<10.4f} | {bias_norm:>+14.2f} km² | {np.mean(y_true_norm_bin)*100:>10.2f}% | {np.mean(y_pred_norm_bin)*100:>10.2f}%")
        print("-" * 95)

    # 5. Full Retrained Pooled Model on ALL 7 Years
    print(f"\n[*] Retraining Final Pooled Model on ALL 7 Normalised Years (Train samples: {len(gdf_tr_norm)})...")
    rf_final_norm = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_final_norm.fit(gdf_tr_norm[FEATURE_NAMES].values, gdf_tr_norm["class_id"].values)

    # 6. Built-up km2 per year: BEFORE vs AFTER
    print("\n[*] BUILT-UP AREA (KM²) PER YEAR: BEFORE VS AFTER NORMALISATION")
    print(f"{'Year':<6} | {'Raw Classified (km²)':<22} | {'Clean Series (km²)':<20} | {'After Norm Pooled (km²)':<24} | {'Norm vs Clean Diff'}")
    print("-" * 95)

    for y in YEARS:
        # Before: raw classified tif
        p_raw_cl = DATA_DIR / f"{city}_{y}_classified.tif"
        with rasterio.open(p_raw_cl) as s:
            raw_arr = s.read(1)
            raw_bup_km2 = float(np.sum(raw_arr == 1) * px_km2)

        # Clean series tif
        p_clean_cl = clean_dir / f"{city}_{y}_classified.tif"
        if not p_clean_cl.exists():
            p_clean_cl = DATA_DIR / f"{city}_{y}_classified.tif"
        with rasterio.open(p_clean_cl) as s:
            clean_arr = s.read(1)
            clean_bup_km2 = float(np.sum(clean_arr == 1) * px_km2)

        # After: predict on normalized composite
        feat_stack = [norm_rasters[y][f] for f in FEATURE_NAMES]
        stack_2d = np.column_stack([arr.ravel() for arr in feat_stack])
        valid_m = np.all(np.isfinite(stack_2d) & (stack_2d != -9999.0), axis=1)
        pred_norm_1d = np.zeros(len(stack_2d), dtype=np.uint8)
        pred_norm_1d[valid_m] = rf_final_norm.predict(stack_2d[valid_m])
        pred_norm_2d = pred_norm_1d.reshape(stack_clean[0].shape)
        smooth_norm = apply_majority_filter_3x3(pred_norm_2d, valid_m.reshape(stack_clean[0].shape))
        norm_bup_km2 = float(np.sum(smooth_norm == 1) * px_km2)
        diff_km2 = norm_bup_km2 - clean_bup_km2

        print(f"{y:<6} | {raw_bup_km2:>18.2f} km² | {clean_bup_km2:>16.2f} km² | {norm_bup_km2:>20.2f} km² | {diff_km2:>+14.2f} km²")
    print("=" * 95)
