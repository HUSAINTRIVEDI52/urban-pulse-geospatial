"""
UrbanPulse - Full Stabilisation Pipeline for Pune (2018-2024)
Executes:
1. Rebuilding Sentinel-2 composites & spectral indices (60m)
2. Sampling pooled multi-year training points (2018, 2021, 2024) with spatial-block split
3. Training pooled Random Forest model (200 trees, class_weight='balanced')
4. Classifying all years with 3x3 majority filter
5. Running temporal consistency & persistence cleanup
6. Downstream analytics (change detection with min-patch 8, concentric rings, sprawl metrics, web export)
7. Evaluating Data Quality Gate
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

from pipeline.build_composite import build_composite
from pipeline.change_detection import detect_changes
from pipeline.compute_indices import compute_indices
from pipeline.export_web import export_web_data
from pipeline.load_db import load_city_data_to_db
from pipeline.quality_gate import validate_quality_gate
from pipeline.ring_analysis import run_ring_analysis
from pipeline.sample_points import sample_training_points
from pipeline.sprawl_metrics import run_sprawl_metrics
from pipeline.temporal_cleanup import run_temporal_cleanup
from pipeline.train_classifier import (
    FEATURE_NAMES,
    PROJECT_CLASS_NAMES,
    classify_raster,
    train_and_evaluate_classifier,
)


def run_pune_stabilisation():
    city = "pune"
    years = list(range(2018, 2025))
    train_years = [2018, 2021, 2024]
    data_dir = Path("data")

    print("=" * 80)
    print(f"[*] Starting UrbanPulse Full Stabilisation for {city.upper()} ({years[0]}-{years[-1]})")
    print("=" * 80)

    start_total = time.time()

    # STEP 1: Rebuild Composites and Indices for all years
    print("\n" + "=" * 80)
    print(">>> STEP 1: Building Sentinel-2 Composites & Spectral Indices (60m)")
    print("=" * 80)
    for yr in years:
        print(f"\n--- Checking Composite & Indices for {city} {yr} ---")
        build_composite(
            city=city,
            year=yr,
            resolution=60.0,
            scenes_per_tile=10,
            min_valid_obs=4,
            force=False,
        )
        compute_indices(city=city, year=yr, force=False)

    # STEP 2: Multi-Year Pooled Sampling
    print("\n" + "=" * 80)
    print(f">>> STEP 2: Sampling Pooled Training/Test Points Across {train_years}")
    print("=" * 80)
    sample_training_points(
        city=city,
        train_years=train_years,
        samples_per_class=300,
        train_ratio=0.70,
        n_spatial_blocks=8,
    )

    # STEP 3: Train Pooled Random Forest Classifier
    print("\n" + "=" * 80)
    print(">>> STEP 3: Training Pooled Multi-Year Random Forest Model")
    print("=" * 80)
    rf_model, pooled_metrics = train_and_evaluate_classifier(
        city=city,
        train_years=train_years,
        force=True,
    )

    # STEP 4: Classify All Years (with 3x3 Majority Filter)
    print("\n" + "=" * 80)
    print(">>> STEP 4: Classifying Land Cover for All Years with Pooled Model & 3x3 Filter")
    print("=" * 80)
    records = []
    for yr in years:
        print(f"\n--- Classifying {city} {yr} ---")
        cls_file, area_summary = classify_raster(city=city, year=yr, force=True)
        rec = {"City": "Pune", "Year": yr, "Composite_NoData_pct": 0.0}
        total_km2 = 0.0
        for cid in range(1, 6):
            cname = PROJECT_CLASS_NAMES[cid]
            km2 = area_summary.get(cname, {}).get("area_km2", 0.0)
            rec[cname] = round(km2, 2)
            total_km2 += km2
        rec["Total_Area_km2"] = round(total_km2, 2)
        records.append(rec)

    df_raw = pd.DataFrame(records)
    df_raw.to_csv(data_dir / f"{city}_class_areas.csv", index=False)

    # STEP 5: Temporal Cleanup (Consistency + Persistence)
    print("\n" + "=" * 80)
    print(">>> STEP 5: Temporal Consistency & Persistence Cleanup")
    print("=" * 80)
    df_clean_summary = run_temporal_cleanup(
        city=city,
        data_dir=data_dir,
        start_year=years[0],
        end_year=years[-1],
    )

    # STEP 6: Downstream Analytics
    print("\n" + "=" * 80)
    print(">>> STEP 6: Downstream Analytics (Change Detection, Rings, Metrics, Web Export, DB Load)")
    print("=" * 80)
    change_res = detect_changes(
        city=city,
        start_year=years[0],
        end_year=years[-1],
        min_patch_size=8,
        use_raw=False,
    )
    run_ring_analysis(city=city, use_raw=False)
    run_sprawl_metrics(city=city, use_raw=False)
    export_web_data(city=city, use_raw=False)
    try:
        load_city_data_to_db(city=city)
    except Exception as db_err:
        print(f"[!] Note: Database load skipped or encountered error: {db_err}")

    # STEP 7: Per-Year Accuracy & Data Quality Gate
    print("\n" + "=" * 80)
    print(">>> STEP 7: Evaluating Data Quality Gate")
    print("=" * 80)
    # Compute per-year accuracy on held-out test points
    test_pooled_file = data_dir / f"{city}_test_points_pooled.geojson"
    per_year_accs = {}
    if test_pooled_file.exists():
        test_gdf = gpd.read_file(test_pooled_file)
        for yr in train_years:
            sub_te = test_gdf[test_gdf["year"] == yr]
            if len(sub_te) > 0:
                X_sub = sub_te[FEATURE_NAMES].values
                y_sub = sub_te["class_id"].values
                y_p = rf_model.predict(X_sub)
                per_year_accs[int(yr)] = float(accuracy_score(y_sub, y_p))

    gate_res = validate_quality_gate(
        df_areas=df_clean_summary,
        per_year_accuracies=per_year_accs,
        gross_gain_km2=change_res["gross_gain_km2"],
        gross_loss_km2=change_res["gross_loss_km2"],
        max_nodata_pct=5.0,
        max_builtup_change_pct=15.0,
        min_accuracy=0.70,
        max_loss_to_gain_ratio=0.30,
    )

    elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"[+] Full Stabilisation for Pune COMPLETED in {elapsed:.1f}s")
    print(f"    - Quality Gate Status: {gate_res['status']}")
    print("=" * 80)


if __name__ == "__main__":
    run_pune_stabilisation()
