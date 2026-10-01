"""
UrbanPulse - End-to-End Annual Analysis Pipeline Runner
Executes annual processing pipeline:
1. Sentinel-2 Surface Reflectance Composite Building
2. Multi-Spectral Indices Computation (NDVI, NDBI, MNDWI)
3. Land Cover Classification (Random Forest with 3x3 majority filter)

Measures and reports per-step elapsed time, validates composite NoData coverage,
and generates class area statistics.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_composite import build_composite, load_city_config
from pipeline.compute_indices import compute_indices
from pipeline.train_classifier import classify_raster


def run_year_pipeline(
    city: str = "ahmedabad",
    year: int = 2023,
    resolution: float = 60.0,
    model_path: str | Path | None = None,
    force: bool = False,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
) -> dict[str, Any]:
    """
    Executes the full annual pipeline for a target city and year.

    Args:
        city: City name (e.g. 'ahmedabad').
        year: Year to process (e.g. 2023).
        resolution: Spatial resolution in meters (default: 60.0).
        model_path: Path to trained Random Forest model.
        force: If True, re-runs and overwrites existing outputs.
        config_path: Optional custom path to city YAML config.
        data_dir: Output directory for rasters and artifacts.

    Returns:
        Dictionary containing step timings, file outputs, and area statistics.
    """
    total_start_time = time.time()
    city_key = city.lower()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    config = load_city_config(city=city, config_path=config_path)
    city_name = config.get("city", {}).get("name", city.capitalize())

    print("\n" + "=" * 80)
    print(" URBANPULSE ANNUAL PROCESSING PIPELINE")
    print(f" City: {city_name} | Year: {year} | Target Resolution: {resolution}m")
    print(f" Configuration: configs/{city_key}.yaml | Force: {force}")
    print("=" * 80)

    timings: dict[str, float] = {}

    # ---------------------------------------------------------
    # STEP 1: Sentinel-2 Dry-Season Surface Reflectance Composite
    # ---------------------------------------------------------
    print(f"\n>>> [1/3] STEP 1: Building Sentinel-2 Composite for {city_name} ({year})...")
    step1_start = time.time()
    band_paths, nodata_pct = build_composite(
        city=city,
        year=year,
        resolution=resolution,
        force=force,
        config_path=config_path,
        data_dir=data_path,
    )
    step1_elapsed = time.time() - step1_start
    timings["composite"] = step1_elapsed
    print(f"\n[+] Step 1 (Composite) Completed in {step1_elapsed:.2f}s | NoData: {nodata_pct:.4f}%")

    # ---------------------------------------------------------
    # STEP 2: Spectral Indices Computation (NDVI, NDBI, MNDWI)
    # ---------------------------------------------------------
    print(f"\n>>> [2/3] STEP 2: Computing Spectral Indices for {city_name} ({year})...")
    step2_start = time.time()
    indices_summary = compute_indices(
        city=city,
        year=year,
        force=force,
        config_path=config_path,
        data_dir=data_path,
    )
    step2_elapsed = time.time() - step2_start
    timings["indices"] = step2_elapsed
    print(f"[+] Step 2 (Spectral Indices) Completed in {step2_elapsed:.2f}s")

    # ---------------------------------------------------------
    # STEP 3: Land Cover Classification & 3x3 Majority Smoothing
    # ---------------------------------------------------------
    print(f"\n>>> [3/3] STEP 3: Classifying Land Cover for {city_name} ({year})...")
    step3_start = time.time()
    classified_file, area_summary = classify_raster(
        city=city,
        year=year,
        model_path=model_path,
        force=force,
        data_dir=data_path,
    )
    step3_elapsed = time.time() - step3_start
    timings["classification"] = step3_elapsed
    print(f"[+] Step 3 (Classification & Filter) Completed in {step3_elapsed:.2f}s")

    # ---------------------------------------------------------
    # PIPELINE SUMMARY & TIMING REPORT
    # ---------------------------------------------------------
    total_elapsed = time.time() - total_start_time
    timings["total"] = total_elapsed

    print("\n" + "=" * 80)
    print(f" PIPELINE EXECUTION SUMMARY: {city_name} ({year})")
    print("=" * 80)
    print(
        f"  Step 1: Composite Generation     : {timings['composite']:>8.2f}s (NoData: {nodata_pct:.4f}%)"
    )
    print(f"  Step 2: Indices Computation      : {timings['indices']:>8.2f}s")
    print(
        f"  Step 3: Random Forest Inference  : {timings['classification']:>8.2f}s (with 3x3 majority filter)"
    )
    print("-" * 80)
    print(f"  Total Pipeline Elapsed Time      : {total_elapsed:>8.2f}s")
    print("=" * 80 + "\n")

    return {
        "city": city,
        "year": year,
        "resolution": resolution,
        "nodata_pct": nodata_pct,
        "timings": timings,
        "band_paths": band_paths,
        "indices_summary": indices_summary,
        "classified_file": classified_file,
        "area_summary": area_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run end-to-end UrbanPulse annual satellite processing and classification pipeline."
    )
    parser.add_argument(
        "--city",
        type=str,
        default="ahmedabad",
        help="Target city name matching configs/{city}.yaml (default: ahmedabad)",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Target year to process (e.g. 2023, 2024)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=60.0,
        help="Spatial resolution in meters (default: 60)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained classifier model (default: data/{city}/rf_model_pooled.pkl)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run and overwrite existing outputs",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Custom path to city config YAML file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory where data rasters are read and saved (default: data)",
    )

    args = parser.parse_args()
    run_year_pipeline(
        city=args.city,
        year=args.year,
        resolution=args.resolution,
        model_path=args.model,
        force=args.force,
        config_path=args.config,
        data_dir=args.data_dir,
    )
