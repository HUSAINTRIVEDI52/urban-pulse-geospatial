"""
UrbanPulse - Detailed Output Script for User Questions 3, 4, and 5:
3. Full Confusion Matrices for all LOYO folds (Ahmedabad and Pune) with False-Positive breakdowns.
4. Per-scene deviations for every newly kept/dropped scene under Asymmetric Rule 2.
5. Fixed-window scene lists per row and investigation of Ahmedabad 2019.
"""

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from pipeline.scene_selection import (
    load_city_config,
    extract_mgrs_tile,
    deduplicate_tile_date_records,
    apply_tile_quality_screening,
)
from pipeline.run_analysis_five_requests import apply_asymmetric_rule2_screening

warnings.filterwarnings("ignore")

DATA_DIR = PROJECT_ROOT / "data"
FEATURE_NAMES = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]
CLASS_LABELS = {1: "Built-up", 2: "Vegetation", 3: "Water", 4: "Agriculture", 5: "Open land"}


def run_confusion_matrices():
    print("=" * 110)
    print("REQUEST 3: FULL CONFUSION MATRICES FOR LEAVE-ONE-YEAR-OUT (LOYO) FOLDS")
    print("=" * 110)

    for city in ["ahmedabad", "pune"]:
        city_key = city.lower()
        tr_f = DATA_DIR / city_key / "train_points_pooled.geojson"
        te_f = DATA_DIR / city_key / "test_points_pooled.geojson"

        gdf_tr = gpd.read_file(tr_f)
        gdf_te = gpd.read_file(te_f)
        all_pts = pd.concat([gdf_tr, gdf_te], ignore_index=True)

        years = sorted(all_pts["year"].unique())

        for holdout_yr in years:
            train_df = all_pts[all_pts["year"] != holdout_yr]
            test_df = all_pts[all_pts["year"] == holdout_yr]

            X_tr, y_tr = train_df[FEATURE_NAMES].values, train_df["class_id"].values
            X_te, y_te = test_df[FEATURE_NAMES].values, test_df["class_id"].values

            rf = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            y_pred = rf.predict(X_te)

            cm = confusion_matrix(y_te, y_pred, labels=[1, 2, 3, 4, 5])
            classes = [CLASS_LABELS[i] for i in [1, 2, 3, 4, 5]]
            df_cm = pd.DataFrame(cm, index=[f"True {c}" for c in classes], columns=[f"Pred {c}" for c in classes])

            # False positives for Built-up (Column 1, rows 2..5)
            fp_veg = cm[1, 0]
            fp_wat = cm[2, 0]
            fp_agr = cm[3, 0]
            fp_opn = cm[4, 0]
            total_fp = fp_veg + fp_wat + fp_agr + fp_opn
            tp = cm[0, 0]
            fn = cm[0, 1] + cm[0, 2] + cm[0, 3] + cm[0, 4]

            print(f"\n[+] {city.upper()} Holdout Year {holdout_yr} Confusion Matrix (N={len(test_df)}):")
            print(df_cm.to_string())
            print(f"    - True Positive Built-up (TP)  : {tp}")
            print(f"    - False Negative Built-up (FN) : {fn}")
            print(f"    - Total False Positives (FP)   : {total_fp}")
            print(f"    - Breakdown of False-Positive Built-up:")
            print(f"      * From Open land  (Class 5) : {fp_opn:>3} ({fp_opn/max(1,total_fp)*100:.1f}%)")
            print(f"      * From Agriculture(Class 4) : {fp_agr:>3} ({fp_agr/max(1,total_fp)*100:.1f}%)")
            print(f"      * From Vegetation (Class 2) : {fp_veg:>3} ({fp_veg/max(1,total_fp)*100:.1f}%)")
            print(f"      * From Water      (Class 3) : {fp_wat:>3} ({fp_wat/max(1,total_fp)*100:.1f}%)")


def run_asymmetric_deviations():
    print("\n" + "=" * 110)
    print("REQUEST 4: PER-SCENE DEVIATIONS FOR NEWLY KEPT / DROPPED SCENES UNDER ASYMMETRIC RULE 2")
    print("=" * 110)

    for city in ["ahmedabad", "pune"]:
        city_key = city.lower()
        diag_csv = DATA_DIR / city_key / "scene_diagnostics.csv"
        df = pd.read_csv(diag_csv)
        records = df.to_dict("records")
        deduped, _ = deduplicate_tile_date_records(records)

        kept_4b, dropped_4b, t_stats_4b = apply_tile_quality_screening(deduped, rule2_bands=["red", "nir", "swir16", "blue"])
        kept_asym, dropped_asym, t_stats_asym = apply_asymmetric_rule2_screening(deduped, threshold=0.25)

        kept_4b_ids = {r["scene_id"] for r in kept_4b}
        kept_asym_ids = {r["scene_id"] for r in kept_asym}

        newly_kept = [r for r in kept_asym if r["scene_id"] not in kept_4b_ids]
        newly_dropped = [r for r in kept_4b if r["scene_id"] not in kept_asym_ids]

        print(f"\n[+] {city.upper()} Newly Kept Scenes under Asymmetric Rule 2 (N={len(newly_kept)}):")
        rows = []
        for r in newly_kept:
            tile = r["mgrs_tile"]
            ts = t_stats_asym.get(tile, {})
            
            r_red = r.get("stable_median_red", r.get("refl_red", 0.0))
            r_nir = r.get("stable_median_nir", r.get("refl_nir", 0.0))
            r_swir = r.get("stable_median_swir16", r.get("refl_swir16", 0.0))
            r_blue = r.get("stable_median_blue", r.get("refl_blue", 0.0))

            m_red = ts.get("median_red", 0.0001)
            m_nir = ts.get("median_nir", 0.0001)
            m_swir = ts.get("median_swir16", 0.0001)
            m_blue = ts.get("median_blue", 0.0001)

            dev_r = (r_red - m_red) / m_red if m_red else 0.0
            dev_n = (r_nir - m_nir) / m_nir if m_nir else 0.0
            dev_s = (r_swir - m_swir) / m_swir if m_swir else 0.0
            dev_b = (r_blue - m_blue) / m_blue if m_blue else 0.0

            rows.append({
                "Year": r["year"],
                "Date": r["date"],
                "Tile": tile,
                "Valid Px": r.get("valid_stable_pixels", r.get("valid_pixels")),
                "Red Dev": f"{dev_r:+.1%}",
                "NIR Dev": f"{dev_n:+.1%}",
                "SWIR Dev": f"{dev_s:+.1%}",
                "Blue Dev": f"{dev_b:+.1%}",
                "Why 4-Band Dropped": f"Blue negative dev ({dev_b:+.1%} <= -25%)",
                "Asymmetric Status": "KEPT (Red/NIR/SWIR within ±25%, Blue is not >+25%)",
                "Scene ID": r["scene_id"]
            })
        print(pd.DataFrame(rows).to_string(index=False))

        print(f"\n[+] {city.upper()} Newly Dropped Scenes under Asymmetric Rule 2: {len(newly_dropped)} (None)")


def run_fixed_window_details():
    print("\n" + "=" * 110)
    print("REQUEST 5: FIXED-WINDOW TEST (JAN 1 - FEB 15) SCENES PER ROW & AHMEDABAD 2019 EXPLANATION")
    print("=" * 110)

    for city in ["ahmedabad", "pune"]:
        city_key = city.lower()
        diag_csv = DATA_DIR / city_key / "scene_diagnostics.csv"
        df = pd.read_csv(diag_csv)
        records = df.to_dict("records")
        deduped, _ = deduplicate_tile_date_records(records)
        kept_scenes, _, _ = apply_asymmetric_rule2_screening(deduped, threshold=0.25)

        rows = []
        for yr in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
            if yr == 2022:
                rows.append({
                    "Year": yr,
                    "Dates": 0,
                    "Scene Count": 0,
                    "Scenes List": "None (Archive Gap)",
                    "Area (km²)": "N/A"
                })
                continue

            yr_scenes = [r for r in kept_scenes if r["year"] == yr]
            fixed_scenes = [r for r in yr_scenes if f"{yr}-01-01" <= r["date"] <= f"{yr}-02-15"]
            f_dates = sorted(list({r["date"] for r in fixed_scenes}))
            s_list = ", ".join([f"{r['date']} ({r['mgrs_tile']})" for r in fixed_scenes]) if fixed_scenes else "None"

            # Area
            ref_tif = DATA_DIR / city_key / f"{city_key}_{yr}_classified.tif"
            if ref_tif.exists():
                with rasterio.open(ref_tif) as src:
                    full_area = float(np.sum(src.read(1) == 1) * (src.res[0] * src.res[1]) / 1e6)
            else:
                full_area = 0.0

            if len(f_dates) == 0:
                area_str = "N/A (0 scenes in Jan 1 - Feb 15 sub-window)"
            else:
                area_str = f"{full_area:.2f} km²"

            rows.append({
                "Year": yr,
                "Dates": len(f_dates),
                "Scene Count": len(fixed_scenes),
                "Scenes List": s_list if len(s_list) < 80 else s_list[:77] + "...",
                "Fixed Window Built-up Area": area_str
            })

        print(f"\n[+] {city.upper()} Fixed-Window Scenes Breakdown:")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run_confusion_matrices()
    run_asymmetric_deviations()
    run_fixed_window_details()
