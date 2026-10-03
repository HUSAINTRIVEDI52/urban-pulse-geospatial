from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from scipy.ndimage import binary_erosion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

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

# Orthogonal / Total Least Squares (TLS) function
def fit_tls_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_c = x - x_mean
    y_c = y - y_mean
    M = np.column_stack([x_c, y_c])
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    v1 = Vt[0]
    slope = float(v1[1] / v1[0]) if v1[0] != 0 else 1.0
    intercept = float(y_mean - slope * x_mean)
    r = np.corrcoef(x, y)[0, 1]
    return slope, intercept, float(r**2)

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

def olofsson_area_estimation(cm, mapped_area_km2_by_class, target_class_idx=0):
    K = cm.shape[0]
    A_total = np.sum(mapped_area_km2_by_class)
    W = mapped_area_km2_by_class / A_total
    n_i_dot = np.sum(cm, axis=1) # row sums = mapped class total samples

    p = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        if n_i_dot[i] > 0:
            p[i, :] = W[i] * (cm[i, :] / n_i_dot[i])

    p_dot_k = np.sum(p[:, target_class_idx])
    A_adj_km2 = A_total * p_dot_k

    var_p_dot_k = 0.0
    for i in range(K):
        if n_i_dot[i] > 1:
            n_ik = cm[i, target_class_idx]
            n_i = n_i_dot[i]
            sample_prop = n_ik / n_i
            var_term = (W[i]**2) * (sample_prop * (1.0 - sample_prop)) / (n_i - 1)
            var_p_dot_k += var_term

    se_p_dot_k = np.sqrt(var_p_dot_k)
    se_A_adj_km2 = A_total * se_p_dot_k
    ci_95_km2 = 1.96 * se_A_adj_km2
    mapped_target_km2 = mapped_area_km2_by_class[target_class_idx]

    return {
        "mapped_area_km2": mapped_target_km2,
        "adjusted_area_km2": A_adj_km2,
        "se_km2": se_A_adj_km2,
        "ci_95_km2": ci_95_km2,
        "ci_lower_km2": max(0.0, A_adj_km2 - ci_95_km2),
        "ci_upper_km2": A_adj_km2 + ci_95_km2,
    }

for city in ["ahmedabad", "pune"]:
    print("\n" + "=" * 115)
    print(f"COMPLETE RECONCILED RUN FOR: {city.upper()}")
    print("=" * 115)

    # 1. Pixel Area & AOI Reconciliation
    sample_tif = DATA_DIR / f"{city}_2021_classified.tif"
    with rasterio.open(sample_tif) as src:
        h, w = src.height, src.width
        res_x, res_y = abs(src.transform.a), abs(src.transform.e)
        px_km2 = (res_x * res_y) / 1e6
        total_px = h * w
        aoi_km2 = total_px * px_km2
        grid_transform = src.transform
        raster_crs = src.crs

    print("\n[1] Spatial Geometry & Pixel Area Reconciliation:")
    print(f"    - Grid Shape         : ({h}, {w}) = {total_px:,} pixels")
    print(f"    - Pixel Resolution   : {res_x:.1f} m x {res_y:.1f} m")
    print(f"    - Pixel Area         : {res_x*res_y:.1f} m² = {px_km2:.6f} km²")
    print(f"    - Total AOI Area     : {total_px:,} * {px_km2:.6f} km² = {aoi_km2:.4f} km²")

    # 2. WorldCover Labels vs 2021
    wc_p = DATA_DIR / f"{city}_worldcover_labels.tif"
    if not wc_p.exists():
        wc_p = DATA_DIR / city / f"{city}_worldcover_labels.tif"

    with rasterio.open(wc_p) as src:
        wc_arr = src.read(1)
        wc_res_x, wc_res_y = abs(src.transform.a), abs(src.transform.e)
        wc_px_km2 = (wc_res_x * wc_res_y) / 1e6
        wc_valid_px = int(np.sum((wc_arr > 0) & (wc_arr != src.nodata)))
        wc_built_px = int(np.sum(wc_arr == 1))
        wc_built_km2 = wc_built_px * wc_px_km2
        wc_total_km2 = wc_valid_px * wc_px_km2

    # 2021 Raw, Clean
    p_raw_2021 = DATA_DIR / f"{city}_2021_classified.tif"
    with rasterio.open(p_raw_2021) as s:
        raw_2021_arr = s.read(1)
        raw_2021_built_km2 = float(np.sum(raw_2021_arr == 1) * px_km2)

    p_clean_2021 = DATA_DIR / city / "clean" / f"{city}_2021_classified.tif"
    if not p_clean_2021.exists():
        p_clean_2021 = DATA_DIR / f"{city}_2021_classified.tif"
    with rasterio.open(p_clean_2021) as s:
        clean_2021_arr = s.read(1)
        clean_2021_built_km2 = float(np.sum(clean_2021_arr == 1) * px_km2)

    print("\n[2] ESA WorldCover Built-up Ground Truth vs 2021 Estimates:")
    print(f"    - WorldCover Grid Shape       : {wc_arr.shape} at {wc_res_x:.1f} m ({wc_px_km2:.8f} km²/px)")
    print(f"    - WorldCover Total Valid Area : {wc_total_km2:.2f} km²")
    print(f"    - WorldCover 2021 Built-up    : {wc_built_px:,} px = {wc_built_km2:.2f} km² ({wc_built_km2/wc_total_km2*100:.2f}%)")
    print(f"    - UrbanPulse Raw 2021 Built-up: {raw_2021_built_km2:.2f} km² ({raw_2021_built_km2/aoi_km2*100:.2f}%)")
    print(f"    - UrbanPulse Clean 2021       : {clean_2021_built_km2:.2f} km² ({clean_2021_built_km2/aoi_km2*100:.2f}%)")

    # 3. Fit Orthogonal / Total Least Squares (TLS) Normalization
    excluded_years = [2019] if city == "ahmedabad" else [2018, 2019]
    print(f"\n[3] Radiometric Normalisation with Total Least Squares (TLS) (Excluding: {city.capitalize()} {excluded_years}):")

    # PIFs
    clean_stack = []
    for y in YEARS:
        cp = DATA_DIR / city / "clean" / f"{city}_{y}_classified.tif"
        if not cp.exists():
            cp = DATA_DIR / f"{city}_{y}_classified.tif"
        with rasterio.open(cp) as src:
            clean_stack.append(src.read(1))
    clean_stack = np.stack(clean_stack, axis=0)

    water_pif = np.sum(clean_stack == 3, axis=0) >= 6
    built_pif = np.sum(clean_stack == 1, axis=0) >= 6
    struct_3x3 = np.ones((3, 3), dtype=bool)
    pif_mask = binary_erosion(water_pif, structure=struct_3x3) | binary_erosion(built_pif, structure=struct_3x3)

    ref_bands = {}
    for b in OPTICAL_BANDS:
        with rasterio.open(DATA_DIR / f"{city}_{REF_YEAR}_{b}.tif") as src:
            ref_bands[b] = src.read(1).astype(np.float32)

    tls_coeffs = {}
    norm_rasters_tls = {y: {} for y in YEARS}

    print(f"{'Year':<6} | {'Band':<8} | {'TLS Slope (m)':<14} | {'TLS Intercept (c)':<18} | {'R²':<8} | {'Before Mean':<12} | {'After Mean':<12} | {'Status'}")
    print("-" * 115)

    for y in YEARS:
        tls_coeffs[y] = {}
        is_excluded = y in excluded_years
        for b in OPTICAL_BANDS:
            with rasterio.open(DATA_DIR / f"{city}_{y}_{b}.tif") as src:
                raw_b = src.read(1).astype(np.float32)

            x_vals = raw_b[pif_mask]
            y_vals = ref_bands[b][pif_mask]
            valid = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals != -9999.0) & (y_vals != -9999.0)

            if y == REF_YEAR:
                m, c, r2 = 1.0, 0.0, 1.0
                status = "Reference Year"
            elif is_excluded:
                m, c, r2 = 1.0, 0.0, 1.0
                status = "Excluded from Fit (Identity)"
            else:
                m, c, r2 = fit_tls_regression(x_vals[valid], y_vals[valid])
                status = "TLS Calibrated"

            norm_b = np.where(raw_b != -9999.0, np.clip(m * raw_b + c, 0.0, 1.5), -9999.0)
            norm_rasters_tls[y][b] = norm_b

            before_m = float(np.mean(x_vals[valid]))
            after_m = float(np.mean(norm_b[pif_mask & (norm_b != -9999.0)]))

            tls_coeffs[y][b] = {"slope": m, "intercept": c, "r2": r2, "status": status}
            print(f"{y:<6} | {b:<8} | {m:<14.4f} | {c:<18.4f} | {r2:<8.4f} | {before_m:<12.4f} | {after_m:<12.4f} | {status}")

        # Indices
        red = norm_rasters_tls[y]["red"]
        green = norm_rasters_tls[y]["green"]
        nir = norm_rasters_tls[y]["nir"]
        swir16 = norm_rasters_tls[y]["swir16"]
        norm_rasters_tls[y]["ndvi"] = safe_norm_diff(nir, red)
        norm_rasters_tls[y]["ndbi"] = safe_norm_diff(swir16, nir)
        norm_rasters_tls[y]["mndwi"] = safe_norm_diff(green, swir16)

    # 4. Point Datasets: ALL Held-Out Points
    tr_pooled_gdf = gpd.read_file(DATA_DIR / city / "train_points_pooled.geojson").to_crs(raster_crs)
    te_pooled_gdf = gpd.read_file(DATA_DIR / city / "test_points_pooled.geojson").to_crs(raster_crs)

    # Extract features for all train/test points from normalized composites using (x, y) coordinates
    tr_all_norm = tr_pooled_gdf.copy()
    te_all_norm = te_pooled_gdf.copy()

    tr_coords = [(geom.x, geom.y) for geom in tr_pooled_gdf.geometry]
    te_coords = [(geom.x, geom.y) for geom in te_pooled_gdf.geometry]

    tr_rowcols = [rasterio.transform.rowcol(grid_transform, x, y) for x, y in tr_coords]
    te_rowcols = [rasterio.transform.rowcol(grid_transform, x, y) for x, y in te_coords]

    for feat in FEATURE_NAMES:
        tr_vals = []
        for i, row in tr_pooled_gdf.iterrows():
            yr = int(row["year"])
            r, c = tr_rowcols[i]
            r = min(max(r, 0), h - 1)
            c = min(max(c, 0), w - 1)
            tr_vals.append(norm_rasters_tls[yr][feat][r, c])
        tr_all_norm[feat] = np.array(tr_vals, dtype=np.float32)

        te_vals = []
        for i, row in te_pooled_gdf.iterrows():
            yr = int(row["year"])
            r, c = te_rowcols[i]
            r = min(max(r, 0), h - 1)
            c = min(max(c, 0), w - 1)
            te_vals.append(norm_rasters_tls[yr][feat][r, c])
        te_all_norm[feat] = np.array(te_vals, dtype=np.float32)

    # Train Final Pooled Model
    rf_final_tls = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_final_tls.fit(tr_all_norm[FEATURE_NAMES].values, tr_all_norm["class_id"].values)

    # 5. Built-up Area per Year
    norm_built_km2_by_year = {}
    mapped_proportions_by_year = {}

    print("\n[4] Built-up Area per Year: Raw vs Cleaned vs TLS-Normalised:")
    print(f"{'Year':<6} | {'Raw (km²)':<12} | {'Clean (km²)':<12} | {'TLS-Norm (km²)':<16} | {'Diff vs Clean'}")
    print("-" * 70)

    for y in YEARS:
        with rasterio.open(DATA_DIR / f"{city}_{y}_classified.tif") as s:
            raw_km2 = float(np.sum(s.read(1) == 1) * px_km2)
        cp = DATA_DIR / city / "clean" / f"{city}_{y}_classified.tif"
        if not cp.exists():
            cp = DATA_DIR / f"{city}_{y}_classified.tif"
        with rasterio.open(cp) as s:
            clean_km2 = float(np.sum(s.read(1) == 1) * px_km2)

        stack_2d = np.column_stack([norm_rasters_tls[y][f].ravel() for f in FEATURE_NAMES])
        valid_m = np.all(np.isfinite(stack_2d) & (stack_2d != -9999.0), axis=1)
        pred_1d = np.zeros(len(stack_2d), dtype=np.uint8)
        pred_1d[valid_m] = rf_final_tls.predict(stack_2d[valid_m])
        smooth = apply_majority_filter_3x3(pred_1d.reshape((h, w)), valid_m.reshape((h, w)))
        norm_km2 = float(np.sum(smooth == 1) * px_km2)
        norm_built_km2_by_year[y] = norm_km2

        # Mapped class areas for Olofsson weighting
        class_areas = np.array([np.sum(smooth == cid) * px_km2 for cid in range(1, 6)], dtype=np.float64)
        mapped_proportions_by_year[y] = class_areas

        print(f"{y:<6} | {raw_km2:>10.2f} km² | {clean_km2:>10.2f} km² | {norm_km2:>12.2f} km² | {norm_km2 - clean_km2:>+10.2f} km²")

    # 2021 comparison
    print(f"\n    * Reconciled 2021 Built-up Area Comparison for {city.upper()}:")
    print(f"      - WorldCover 2021 Reference : {wc_built_km2:.2f} km²")
    print(f"      - Raw 2021 Classified       : {raw_2021_built_km2:.2f} km²")
    print(f"      - Cleaned 2021 Series       : {clean_2021_built_km2:.2f} km²")
    print(f"      - TLS Normalised 2021       : {norm_built_km2_by_year[2021]:.2f} km²")

    # 6. LOYO with Area-Weighted Olofsson (2014) CI on ALL Held-Out Test Points
    print(f"\n[5] Leave-One-Year-Out (LOYO) on ALL Held-Out Test Points (N={len(te_pooled_gdf[te_pooled_gdf['year']==2021])} pts/year):")
    print("    * Test Set: ALL spatial-block test points for the held-out year (test_points_pooled.geojson)")
    print(f"{'Year':<6} | {'Stage':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Mapped Area':<14} | {'Olofsson Adjusted Area (95% CI)'}")
    print("-" * 115)

    for holdout_yr in [2018, 2021, 2024]:
        # BEFORE (Raw pooled data)
        tr_raw = tr_pooled_gdf[tr_pooled_gdf["year"] != holdout_yr]
        te_raw = te_pooled_gdf[te_pooled_gdf["year"] == holdout_yr]
        rf_r = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_r.fit(tr_raw[FEATURE_NAMES].values, tr_raw["class_id"].values)
        y_pred_r = rf_r.predict(te_raw[FEATURE_NAMES].values)
        y_true_r = te_raw["class_id"].values

        # Confusion matrix for Olofsson (row=mapped/pred, col=reference/true)
        cm_r = confusion_matrix(y_pred_r, y_true_r, labels=[1, 2, 3, 4, 5])

        # Raw mapped areas from classified tif
        with rasterio.open(DATA_DIR / f"{city}_{holdout_yr}_classified.tif") as s:
            raw_map_arr = s.read(1)
        raw_mapped_by_class = np.array([np.sum(raw_map_arr == cid) * px_km2 for cid in range(1, 6)], dtype=np.float64)

        olof_r = olofsson_area_estimation(cm_r, raw_mapped_by_class, target_class_idx=0)

        p_r = precision_score((y_true_r == 1).astype(int), (y_pred_r == 1).astype(int), zero_division=0)
        r_r = recall_score((y_true_r == 1).astype(int), (y_pred_r == 1).astype(int), zero_division=0)
        f1_r = f1_score((y_true_r == 1).astype(int), (y_pred_r == 1).astype(int), zero_division=0)

        # AFTER (TLS Normalized)
        tr_n = tr_all_norm[tr_all_norm["year"] != holdout_yr]
        te_n = te_all_norm[te_all_norm["year"] == holdout_yr]
        rf_n = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_n.fit(tr_n[FEATURE_NAMES].values, tr_n["class_id"].values)
        y_pred_n = rf_n.predict(te_n[FEATURE_NAMES].values)
        y_true_n = te_n["class_id"].values

        cm_n = confusion_matrix(y_pred_n, y_true_n, labels=[1, 2, 3, 4, 5])
        norm_mapped_by_class = mapped_proportions_by_year[holdout_yr]
        olof_n = olofsson_area_estimation(cm_n, norm_mapped_by_class, target_class_idx=0)

        p_n = precision_score((y_true_n == 1).astype(int), (y_pred_n == 1).astype(int), zero_division=0)
        r_n = recall_score((y_true_n == 1).astype(int), (y_pred_n == 1).astype(int), zero_division=0)
        f1_n = f1_score((y_true_n == 1).astype(int), (y_pred_n == 1).astype(int), zero_division=0)

        print(f"{holdout_yr:<6} | {'BEFORE':<8} | {p_r:<10.4f} | {r_r:<10.4f} | {f1_r:<10.4f} | {olof_r['mapped_area_km2']:>10.2f} km² | {olof_r['adjusted_area_km2']:>10.2f} ± {olof_r['ci_95_km2']:>6.2f} km² [{olof_r['ci_lower_km2']:.2f}, {olof_r['ci_upper_km2']:.2f}]")
        print(f"{holdout_yr:<6} | {'AFTER':<8} | {p_n:<10.4f} | {r_n:<10.4f} | {f1_n:<10.4f} | {olof_n['mapped_area_km2']:>10.2f} km² | {olof_n['adjusted_area_km2']:>10.2f} ± {olof_n['ci_95_km2']:>6.2f} km² [{olof_n['ci_lower_km2']:.2f}, {olof_n['ci_upper_km2']:.2f}]")
        print("-" * 115)
