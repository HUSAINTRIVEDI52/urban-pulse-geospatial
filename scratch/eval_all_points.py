import json
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

PROJECT_ROOT = Path("f:/gis-project/UrbanPulse")
DATA_DIR = PROJECT_ROOT / "data"

FEATURE_NAMES = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]
OPTICAL_BANDS = ["blue", "green", "red", "nir", "swir16"]

def olofsson_area_estimation(cm, mapped_area_km2_by_class, target_class_idx=0):
    K = cm.shape[0]
    A_total = np.sum(mapped_area_km2_by_class)
    W = mapped_area_km2_by_class / A_total
    n_i_dot = np.sum(cm, axis=1)
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
    return {
        "mapped_area_km2": float(mapped_area_km2_by_class[target_class_idx]),
        "adjusted_area_km2": float(A_adj_km2),
        "ci_95_km2": float(ci_95_km2),
    }

for city in ["ahmedabad", "pune"]:
    print(f"\n=======================================================")
    print(f"=== CITY: {city.upper()} ===")
    print(f"=======================================================")
    with rasterio.open(DATA_DIR / f"{city}_2021_classified.tif") as src:
        h, w = src.height, src.width
        res_x, res_y = abs(src.transform.a), abs(src.transform.e)
        px_km2 = (res_x * res_y) / 1e6
        grid_transform = src.transform
        raster_crs = src.crs

    tr_pooled = gpd.read_file(DATA_DIR / city / "train_points_pooled.geojson").to_crs(raster_crs)
    te_pooled = gpd.read_file(DATA_DIR / city / "test_points_pooled.geojson").to_crs(raster_crs)
    
    # Load normalization coeffs or compute normalized features for points
    # Extract coords
    tr_coords = [(g.x, g.y) for g in tr_pooled.geometry]
    te_coords = [(g.x, g.y) for g in te_pooled.geometry]
    tr_rc = [rasterio.transform.rowcol(grid_transform, x, y) for x, y in tr_coords]
    te_rc = [rasterio.transform.rowcol(grid_transform, x, y) for x, y in te_coords]
    
    # Check if normalized rasters exist
    norm_rasters = {}
    for y in [2018, 2021, 2024]:
        norm_dir = DATA_DIR / city / "normalized"
        if (norm_dir / f"{city}_{y}_red.tif").exists():
            norm_rasters[y] = {}
            for f in FEATURE_NAMES:
                with rasterio.open(norm_dir / f"{city}_{y}_{f}.tif") as s:
                    norm_rasters[y][f] = s.read(1)
                    
    tr_norm = tr_pooled.copy()
    te_norm = te_pooled.copy()
    if norm_rasters:
        for f in FEATURE_NAMES:
            tr_norm[f] = np.array([norm_rasters[int(row["year"])][f][min(max(tr_rc[i][0], 0), h-1), min(max(tr_rc[i][1], 0), w-1)] if int(row["year"]) in norm_rasters else row[f] for i, row in tr_pooled.iterrows()])
            te_norm[f] = np.array([norm_rasters[int(row["year"])][f][min(max(te_rc[i][0], 0), h-1), min(max(te_rc[i][1], 0), w-1)] if int(row["year"]) in norm_rasters else row[f] for i, row in te_pooled.iterrows()])

    # Mapped areas for TLS
    # In export_web / scratch
    tls_mapped_by_year = {
        "ahmedabad": {2018: [406.31, 300, 50, 400, 200], 2021: [407.74, 300, 50, 400, 200], 2024: [466.19, 300, 50, 400, 200]},
        "pune": {2018: [199.7, 300, 50, 400, 200], 2021: [377.92, 300, 50, 400, 200], 2024: [461.92, 300, 50, 400, 200]}
    }

    for holdout_yr in [2018, 2021, 2024]:
        # --- RAW ---
        tr_r = tr_pooled[tr_pooled["year"] != holdout_yr]
        te_r = te_pooled[te_pooled["year"] == holdout_yr]
        rf_r = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_r.fit(tr_r[FEATURE_NAMES].values, tr_r["class_id"].values)
        y_pred_r = rf_r.predict(te_r[FEATURE_NAMES].values)
        y_true_r = te_r["class_id"].values
        cm_r = confusion_matrix(y_pred_r, y_true_r, labels=[1, 2, 3, 4, 5])
        
        with rasterio.open(DATA_DIR / f"{city}_{holdout_yr}_classified.tif") as s:
            raw_map_arr = s.read(1)
        raw_map_areas = np.array([np.sum(raw_map_arr == cid) * px_km2 for cid in range(1, 6)], dtype=np.float64)
        olof_r = olofsson_area_estimation(cm_r, raw_map_areas, target_class_idx=0)
        
        p_r = precision_score((y_true_r == 1).astype(int), (y_pred_r == 1).astype(int), zero_division=0)
        r_r = recall_score((y_true_r == 1).astype(int), (y_pred_r == 1).astype(int), zero_division=0)
        f1_r = f1_score((y_true_r == 1).astype(int), (y_pred_r == 1).astype(int), zero_division=0)
        
        print(f"[{city.upper()} {holdout_yr} RAW] N={len(te_r)}, Prec={p_r:.4f}, Rec={r_r:.4f}, F1={f1_r:.4f}, Mapped={olof_r['mapped_area_km2']:.2f} km2, Adj={olof_r['adjusted_area_km2']:.2f} +- {olof_r['ci_95_km2']:.2f} km2")

        # --- TLS NORM ---
        if norm_rasters:
            tr_n = tr_norm[tr_norm["year"] != holdout_yr]
            te_n = te_norm[te_norm["year"] == holdout_yr]
            rf_n = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
            rf_n.fit(tr_n[FEATURE_NAMES].values, tr_n["class_id"].values)
            y_pred_n = rf_n.predict(te_n[FEATURE_NAMES].values)
            y_true_n = te_n["class_id"].values
            cm_n = confusion_matrix(y_pred_n, y_true_n, labels=[1, 2, 3, 4, 5])
            
            # If normalized classified raster exists or we use proxy class areas
            olof_n = olofsson_area_estimation(cm_n, raw_map_areas, target_class_idx=0)
            p_n = precision_score((y_true_n == 1).astype(int), (y_pred_n == 1).astype(int), zero_division=0)
            r_n = recall_score((y_true_n == 1).astype(int), (y_pred_n == 1).astype(int), zero_division=0)
            f1_n = f1_score((y_true_n == 1).astype(int), (y_pred_n == 1).astype(int), zero_division=0)
            print(f"[{city.upper()} {holdout_yr} TLS] N={len(te_n)}, Prec={p_n:.4f}, Rec={r_n:.4f}, F1={f1_n:.4f}, Mapped={olof_n['mapped_area_km2']:.2f} km2, Adj={olof_n['adjusted_area_km2']:.2f} +- {olof_n['ci_95_km2']:.2f} km2")
