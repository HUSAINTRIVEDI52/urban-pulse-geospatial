"""
UrbanPulse - End-to-End City Processing Orchestrator
Executes the full satellite geospatial intelligence chain for any configured city:
1. Multi-year Sentinel-2 compositing, spectral indices, and Random Forest classification
2. Multi-temporal urban land cover change detection & transition matrix analysis
3. Concentric radial distance ring density gradient analysis
4. Multi-year sprawl velocity, Shannon spatial entropy, and core vs periphery distribution
5. Web asset generation (WGS84 transparent PNG overlays, meta.json, stats.json)
6. PostGIS spatial database ingestion (optional / fallback).
"""

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.change_detection import detect_changes
from pipeline.export_web import export_web_data
from pipeline.ring_analysis import run_ring_analysis
from pipeline.run_all_years import run_all_years
from pipeline.sprawl_metrics import run_sprawl_metrics


def load_city_config(city: str, config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads city YAML configuration."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_city_pipeline(
    city: str = "pune",
    start_year: int | None = None,
    end_year: int | None = None,
    force: bool = False,
    skip_db: bool = False,
) -> dict[str, Any]:
    """
    Executes the complete satellite sprawl analytics pipeline for a city.
    """
    city_key = city.lower()
    data_path = Path(f"data/{city_key}") if Path(f"data/{city_key}").exists() else Path("data")
    config = load_city_config(city=city_key)
    city_name = config.get("city", {}).get("name", city.capitalize())

    temporal_cfg = config.get("temporal", {}).get("analysis_years", {})
    cfg_start_year = temporal_cfg.get("start_year", 2017)
    cfg_end_year = temporal_cfg.get("end_year", 2024)

    if start_year is None:
        start_year = cfg_start_year
    if end_year is None:
        end_year = cfg_end_year

    overall_start = time.time()
    print("=" * 80)
    print(f"[*] UrbanPulse Complete City Pipeline: {city_name} ({start_year} - {end_year})")
    print(f"    - Bounding Box : {config.get('spatial', {}).get('bbox')}")
    print(f"    - Started At   : {datetime.now(UTC).isoformat()}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # STEP 1: Multi-Year Processing (Compositing, Indices, Classifier, Class Areas)
    # --------------------------------------------------------------------------
    print(f"\n[PHASE 1/5] Processing Multi-Year Satellite Series for {city_name}...")
    df_areas, successful_years = run_all_years(
        city=city_key,
        start_year=start_year,
        end_year=end_year,
        force=force,
    )

    if not successful_years:
        raise RuntimeError(f"No years were successfully processed for {city_name}.")

    actual_start_year = min(successful_years)
    actual_end_year = max(successful_years)

    # --------------------------------------------------------------------------
    # DATA QUALITY GATE: Validate NoData, built-up volatility, and accuracy
    # --------------------------------------------------------------------------
    print(f"\n[*] Evaluating Data Quality Gate for {city_name}...")
    try:
        from pipeline.quality_gate import DataQualityGateError, validate_quality_gate

        # Evaluate model accuracy if available
        acc = None
        try:
            import geopandas as gpd
            import joblib
            import rasterio
            from sklearn.metrics import accuracy_score

            model_file = data_path / f"{city_key}_rf_model_{actual_end_year}.pkl"
            test_file = data_path / f"{city_key}_test_points.geojson"
            if model_file.exists() and test_file.exists():
                rf_clf = joblib.load(model_file)
                test_gdf = gpd.read_file(test_file).to_crs(epsg=32643)
                feats = ["blue", "green", "red", "nir", "swir16", "ndvi", "ndbi", "mndwi"]
                rasters = {
                    f: rasterio.open(data_path / f"{city_key}_{actual_end_year}_{f}.tif")
                    for f in feats
                    if (data_path / f"{city_key}_{actual_end_year}_{f}.tif").exists()
                }
                if len(rasters) == len(feats):
                    coords = [(g.x, g.y) for g in test_gdf.geometry]
                    X_vals = [[s[0] for s in rasters[f].sample(coords)] for f in feats]
                    df_X = pd.DataFrame(dict(zip(feats, X_vals)))
                    y_true = test_gdf["class_id"].values
                    valid_m = ~df_X.isna().any(axis=1) & (y_true > 0)
                    if valid_m.sum() > 0:
                        y_p = rf_clf.predict(df_X[valid_m].values)
                        acc = float(accuracy_score(y_true[valid_m], y_p))
        except Exception:
            pass

        gate_summary = validate_quality_gate(
            df_areas=df_areas,
            overall_accuracy=acc,
            max_nodata_pct=5.0,
            max_builtup_change_pct=25.0,
            min_accuracy=0.70,
        )
        print(f"[+] Data Quality Gate PASSED: {gate_summary}")

    except DataQualityGateError as qe:
        total_duration = time.time() - overall_start
        print(f"\n[!] DATA QUALITY GATE REJECTION: {qe}")

        # Record failure in pipeline_runs table
        try:
            from pipeline.load_db import get_db_connection, log_pipeline_run

            db_url = os.getenv("DATABASE_URL")
            if db_url:
                conn = get_db_connection(db_url)
                log_pipeline_run(
                    conn,
                    city_id=city_key,
                    year=actual_end_year,
                    status="FAILED",
                    started_at=datetime.fromtimestamp(overall_start, tz=UTC),
                    finished_at=datetime.now(UTC),
                    error=str(qe),
                )
                conn.close()
        except Exception:
            pass

        # Export failure metrics to Prometheus
        try:
            from pipeline.metrics_exporter import export_pipeline_metrics

            export_pipeline_metrics(
                city=city_key,
                status="FAILED",
                duration_seconds=total_duration,
                year=actual_end_year,
            )
        except Exception:
            pass

        raise qe

    # --------------------------------------------------------------------------
    # STEP 2: Land Cover Change Detection & Transition Matrix
    # --------------------------------------------------------------------------
    print(
        f"\n[PHASE 2/5] Running Urban Land Cover Change Detection ({actual_start_year} -> {actual_end_year})..."
    )
    change_res = detect_changes(
        city=city_key,
        start_year=actual_start_year,
        end_year=actual_end_year,
        min_patch_size=3,
    )

    # --------------------------------------------------------------------------
    # STEP 3: Concentric Ring Urban Gradient Analysis
    # --------------------------------------------------------------------------
    print(f"\n[PHASE 3/5] Running Concentric Ring Sprawl Analysis for {city_name}...")
    df_rings = run_ring_analysis(
        city=city_key,
        ring_width_km=2.0,
        max_dist_km=22.0,
    )

    # --------------------------------------------------------------------------
    # STEP 4: Sprawl Velocity & Shannon Spatial Entropy Metrics
    # --------------------------------------------------------------------------
    print(f"\n[PHASE 4/5] Computing Shannon Entropy & Sprawl Metrics for {city_name}...")
    df_metrics = run_sprawl_metrics(
        city=city_key,
    )

    # --------------------------------------------------------------------------
    # STEP 5: Web Asset Export (Transparent PNGs, meta.json, stats.json)
    # --------------------------------------------------------------------------
    print(f"\n[PHASE 5/5] Exporting Web Client Datasets for {city_name}...")
    web_res = export_web_data(
        city=city_key,
    )

    # --------------------------------------------------------------------------
    # OPTIONAL STEP 6: PostGIS Database Ingestion
    # --------------------------------------------------------------------------
    db_res = None
    if not skip_db:
        try:
            from pipeline.load_db import load_city_data_to_db

            print(f"\n[*] Ingesting {city_name} data into PostGIS database...")
            db_res = load_city_data_to_db(city=city_key)
            print("[+] PostGIS Ingestion Successful.")
        except Exception as e:
            print(f"[!] PostGIS ingestion skipped or unavailable: {e}")

    total_duration = time.time() - overall_start
    print("\n" + "=" * 80)
    print(f"[+] Complete UrbanPulse Pipeline Finished for {city_name} in {total_duration:.1f}s")
    print("=" * 80)

    # Export metrics to Prometheus Pushgateway / Textfile Collector
    try:
        from pipeline.metrics_exporter import export_pipeline_metrics

        avg_nodata = None
        if "Composite_NoData_pct" in df_areas.columns:
            avg_nodata = float(df_areas["Composite_NoData_pct"].mean())

        export_pipeline_metrics(
            city=city_key,
            status="SUCCESS",
            duration_seconds=total_duration,
            nodata_pct=avg_nodata,
            scenes_count=len(successful_years),
            year=successful_years[-1] if successful_years else None,
        )
    except Exception as me:
        print(f"[*] Note: Prometheus metrics export notification: {me}")

    return {
        "city": city_key,
        "processed_years": successful_years,
        "class_areas": df_areas,
        "change_detection": change_res,
        "rings": df_rings,
        "metrics": df_metrics,
        "web_assets": web_res,
        "database": db_res,
        "total_duration_s": total_duration,
    }


def main():
    parser = argparse.ArgumentParser(
        description="UrbanPulse Complete City Pipeline Runner",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="pune",
        help="Target city key (default: pune)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start year (default: from config)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End year (default: from config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of intermediate raster steps",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip PostGIS database ingestion",
    )

    args = parser.parse_args()

    run_city_pipeline(
        city=args.city,
        start_year=args.start,
        end_year=args.end,
        force=args.force,
        skip_db=args.skip_db,
    )


if __name__ == "__main__":
    main()
