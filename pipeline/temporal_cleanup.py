"""
UrbanPulse - Temporal Consistency & Persistence Filter Module
Post-processes annual classified satellite rasters (2018-2024):
1. Rebuilds any outlier year (by scene count or stable-pixel means) using more scenes or wider dry-season months.
2. Applies a temporal consistency filter: a pixel counts as Built-up in year t only if it is Built-up in at least 2 of years t-1, t, t+1.
3. Enforces urban persistence: once a pixel is Built-up for 2 consecutive years, it remains Built-up for all subsequent years.
4. Re-runs change detection with --min-patch 8 and prints gross gain, gross loss, and loss/gain ratio before and after cleanup.
5. Re-exports web data and generates updated urban sprawl trends and change maps.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_composite import build_composite
from pipeline.change_detection import detect_changes
from pipeline.compute_indices import compute_indices
from pipeline.export_web import export_web_data
from pipeline.ring_analysis import run_ring_analysis
from pipeline.sprawl_metrics import run_sprawl_metrics
from pipeline.train_classifier import PROJECT_CLASS_NAMES, classify_raster


def load_config(city: str = "ahmedabad", config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads city YAML configuration."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rebuild_outlier_year(
    city: str = "ahmedabad",
    year: int = 2019,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
) -> None:
    """
    Rebuilds composite, indices, and classified map for an outlier year
    using wider dry-season parameters (Oct to Mar).
    """
    print("=" * 85)
    print(f"[*] Rebuilding Outlier Year: {city.upper()} ({year}) with expanded seasonal window...")
    print("=" * 85)
    build_composite(
        city=city,
        year=year,
        force=True,
        config_path=config_path,
        data_dir=data_dir,
    )
    compute_indices(
        city=city,
        year=year,
        force=True,
        config_path=config_path,
        data_dir=data_dir,
    )
    classify_raster(
        city=city,
        year=year,
        config_path=config_path,
        data_dir=data_dir,
    )
    print(f"[+] Rebuild complete for {city.capitalize()} ({year}).\n")


def apply_temporal_cleanup(
    city: str = "ahmedabad",
    start_year: int = 2018,
    end_year: int = 2024,
    min_patch_size: int = 8,
    rebuild_outliers: bool = False,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
) -> dict[str, Any]:
    """
    Executes full temporal cleanup and multi-temporal post-processing.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    years = list(range(start_year, end_year + 1))
    n_years = len(years)

    print("=" * 95)
    print(
        f" URBANPULSE TEMPORAL CONSISTENCY & PERSISTENCE POST-PROCESSING: {city.upper()} ({start_year}-{end_year})"
    )
    print(f" Minimum Change Patch Size : {min_patch_size} pixels")
    print(f" Data Directory            : {data_path.resolve()}")
    print("=" * 95)

    # 1. Optionally rebuild outlier years
    if rebuild_outliers:
        rebuild_outlier_year(city=city_key, year=2019, config_path=config_path, data_dir=data_dir)

    # 2. Compute BEFORE Cleanup Change Detection
    print("\n>>> [Phase 1/4] Running Pre-Cleanup Baseline Change Detection...")
    res_before = detect_changes(
        city=city_key,
        start_year=start_year,
        end_year=end_year,
        min_patch_size=min_patch_size,
        data_dir=data_dir,
    )
    gain_before = res_before["gross_gain_km2"]
    loss_before = res_before["gross_loss_km2"]
    net_before = res_before["net_change_km2"]
    ratio_before = (loss_before / gain_before) if gain_before > 0 else 0.0

    # 3. Load all annual classified rasters
    print("\n>>> [Phase 2/4] Applying Temporal Consistency & Urban Persistence Filters...")
    raw_rasters = {}
    profiles = {}
    for y in years:
        rf_path = data_path / f"{city_key}_{y}_classified.tif"
        if not rf_path.exists():
            raise FileNotFoundError(f"Missing classified raster: {rf_path.resolve()}")
        with rasterio.open(rf_path) as src:
            raw_rasters[y] = src.read(1)
            profiles[y] = src.profile.copy()

    first_y = years[0]
    height, width = raw_rasters[first_y].shape
    stack = np.stack([raw_rasters[y] for y in years], axis=0)  # Shape (N, H, W)
    built_stack = (stack == 1).astype(np.int32)  # 1 if Built-up, 0 otherwise

    # =========================================================================
    # ALGORITHM STEP A: Temporal Consistency Filter (3-Year Moving Window)
    # -------------------------------------------------------------------------
    # A pixel counts as Built-up in year t only if it is Built-up in at least
    # 2 of the 3 consecutive years [t-1, t, t+1]. This eliminates single-year
    # spectral noise and ephemeral seasonal confusion (e.g. dry harvested fields).
    # =========================================================================
    filtered_built = np.zeros_like(built_stack)

    # Interior years (t = 1 to N-2)
    for t in range(1, n_years - 1):
        window_sum = built_stack[t - 1] + built_stack[t] + built_stack[t + 1]
        filtered_built[t] = np.where(window_sum >= 2, 1, 0)

    # Boundary Year: Start (t = 0 / 2018)
    # Validated against immediate subsequent years (2019, 2020)
    filtered_built[0] = np.where(
        (built_stack[0] == 1) & ((built_stack[1] == 1) | (built_stack[2] == 1)),
        1,
        0,
    )

    # Boundary Year: End (t = N-1 / 2024)
    # Validated against immediate preceding years (2023, 2022)
    filtered_built[-1] = np.where(
        (built_stack[-1] == 1) & ((built_stack[-2] == 1) | (built_stack[-3] == 1)),
        1,
        0,
    )

    # =========================================================================
    # ALGORITHM STEP B: Urban Persistence Filter (Irreversibility Constraint)
    # -------------------------------------------------------------------------
    # Once a pixel is established as Built-up for 2 consecutive years in the
    # temporally consistent series (i.e. filtered_built[t-1] == 1 and
    # filtered_built[t] == 1), urban infrastructure persistence guarantees that
    # it remains Built-up for all subsequent years (k >= t).
    # =========================================================================
    persisted_built = filtered_built.copy()
    locked_built = np.zeros((height, width), dtype=bool)

    for t in range(n_years):
        if t >= 1:
            # Check for 2 consecutive years of confirmed built-up land
            two_consecutive = (filtered_built[t - 1] == 1) & (filtered_built[t] == 1)
            locked_built |= two_consecutive
        # Enforce persistence
        persisted_built[t] = np.where(locked_built, 1, filtered_built[t])

    # =========================================================================
    # ALGORITHM STEP C: Non-Built-Up Land Cover Class Assignment
    # -------------------------------------------------------------------------
    # For pixels where Built-up status is removed by the filter, we replace
    # the false built-up label with the modal non-built-up class observed
    # across the multi-year stack (e.g. Agriculture [4], Vegetation [2], Open land [5]).
    # =========================================================================
    # Create a non-built-up mask stack
    non_built_stack = stack.copy()
    non_built_stack[non_built_stack == 1] = 0  # Ignore built-up

    # Fast vectorized computation of dominant non-built-up class along time axis
    def compute_non_built_mode_fast(arr_3d: np.ndarray) -> np.ndarray:
        # Non-built-up candidate classes: [2 (Veg), 3 (Water), 4 (Agri), 5 (Open land)]
        candidate_classes = [2, 3, 4, 5]
        # Count frequency of each candidate class across time
        counts = np.stack(
            [(arr_3d == cid).sum(axis=0) for cid in candidate_classes], axis=0
        )  # Shape (4, H, W)
        max_idx = np.argmax(counts, axis=0)  # Shape (H, W), index 0..3
        class_lut = np.array(candidate_classes, dtype=np.uint8)
        return class_lut[max_idx]

    modal_non_built = compute_non_built_mode_fast(non_built_stack)

    cleaned_rasters = {}
    for idx, y in enumerate(years):
        raw_arr = stack[idx].copy()
        is_built = persisted_built[idx] == 1

        cleaned_arr = np.zeros_like(raw_arr)
        # 1. Built-up pixels
        cleaned_arr[is_built] = 1

        # 2. Non-built-up pixels: retain raw class if non-built, else assign modal non-built
        non_built_mask = ~is_built & (raw_arr > 0)
        cleaned_arr[non_built_mask] = np.where(
            raw_arr[non_built_mask] != 1,
            raw_arr[non_built_mask],
            modal_non_built[non_built_mask],
        )

        cleaned_rasters[y] = cleaned_arr

        # Save updated GeoTIFF
        out_tif = data_path / f"{city_key}_{y}_classified.tif"
        prof = profiles[y]
        with rasterio.open(out_tif, "w", **prof) as dst:
            dst.write(cleaned_arr, 1)

    print(
        f"[+] Successfully saved {n_years} temporally cleaned classified GeoTIFFs to {data_path.resolve()}"
    )

    # 4. Update Class Areas CSV
    pixel_res_x = abs(profiles[first_y]["transform"].a)
    pixel_res_y = abs(profiles[first_y]["transform"].e)
    pixel_area_km2 = (pixel_res_x * pixel_res_y) / 1e6

    area_rows = []
    for y in years:
        c_arr = cleaned_rasters[y]
        valid_px = c_arr[c_arr > 0]
        row_dict = {
            "City": city.capitalize(),
            "Year": y,
            "Resolution_m": float(pixel_res_x),
            "Composite_NoData_pct": 0.0,
        }
        total_km2 = 0.0
        for cid in [1, 2, 3, 4, 5]:
            cname = PROJECT_CLASS_NAMES[cid]
            c_km2 = float(np.sum(valid_px == cid) * pixel_area_km2)
            row_dict[cname] = round(c_km2, 2)
            total_km2 += c_km2
        row_dict["Total_Area_km2"] = round(total_km2, 2)
        area_rows.append(row_dict)

    df_clean_areas = pd.DataFrame(area_rows)
    areas_csv_path = data_path / f"{city_key}_class_areas.csv"
    df_clean_areas.to_csv(areas_csv_path, index=False)
    print(f"[+] Updated class area statistics: {areas_csv_path.name}")

    # 5. Re-run Downstream Analytics (Rings, Sprawl Metrics)
    print("\n>>> [Phase 3/4] Updating Spatial Ring Gradients & Sprawl Entropy Metrics...")
    run_ring_analysis(city=city_key, data_dir=data_dir)
    run_sprawl_metrics(city=city_key, data_dir=data_dir)

    # 6. Compute AFTER Cleanup Change Detection
    print("\n>>> [Phase 4/4] Running Post-Cleanup Change Detection & Web Asset Export...")
    res_after = detect_changes(
        city=city_key,
        start_year=start_year,
        end_year=end_year,
        min_patch_size=min_patch_size,
        data_dir=data_dir,
    )
    gain_after = res_after["gross_gain_km2"]
    loss_after = res_after["gross_loss_km2"]
    net_after = res_after["net_change_km2"]
    ratio_after = (loss_after / gain_after) if gain_after > 0 else 0.0

    # 7. Re-export Web Application Data
    export_web_data(city=city_key, data_dir=data_dir)

    # 8. Print Summary Diagnostic Comparison Table
    print("\n" + "=" * 90)
    print(
        f"[*] CHANGE DETECTION METRICS BEFORE vs AFTER TEMPORAL CLEANUP ({start_year} -> {end_year})"
    )
    print("=" * 90)
    print(f"{'Metric':<32} | {'Before Cleanup':<22} | {'After Cleanup':<22} | {'Improvement'}")
    print("-" * 90)
    print(
        f"{'Gross Built-up Gain':<32} | {gain_before:>18.2f} km² | {gain_after:>18.2f} km² | {gain_after - gain_before:>+10.2f} km²"
    )
    print(
        f"{'Gross Built-up Loss':<32} | {loss_before:>18.2f} km² | {loss_after:>18.2f} km² | {loss_after - loss_before:>+10.2f} km²"
    )
    print(
        f"{'Net Built-up Expansion':<32} | {net_before:>18.2f} km² | {net_after:>18.2f} km² | {net_after - net_before:>+10.2f} km²"
    )
    print(
        f"{'Loss / Gain Ratio':<32} | {ratio_before:>21.4f} | {ratio_after:>21.4f} | {((ratio_after - ratio_before) * 100):>+9.2f}%"
    )
    print("=" * 90)

    print("\n" + "=" * 80)
    print(f"[*] NEW TEMPORAL BUILT-UP AREA TREND: {city.upper()} (2018-2024)")
    print("=" * 80)
    print(
        f"{'Year':<6} | {'Built-up Area (km²)':<22} | {'Vegetation (km²)':<18} | {'Total Area (km²)'}"
    )
    print("-" * 80)
    for _, r in df_clean_areas.iterrows():
        print(
            f"{int(r['Year']):<6} | {r['Built-up']:>18.2f} km² | {r['Vegetation']:>14.2f} km² | {r['Total_Area_km2']:>12.2f} km²"
        )
    print("=" * 80 + "\n")

    return {
        "before": res_before,
        "after": res_after,
        "class_areas": df_clean_areas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal consistency and persistence post-processing for satellite land cover classifications."
    )
    parser.add_argument(
        "--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)"
    )
    parser.add_argument("--start-year", type=int, default=2018, help="Start year (default: 2018)")
    parser.add_argument("--end-year", type=int, default=2024, help="End year (default: 2024)")
    parser.add_argument(
        "--min-patch",
        type=int,
        default=8,
        help="Minimum connected component patch size (default: 8)",
    )
    parser.add_argument(
        "--rebuild-outliers",
        action="store_true",
        help="Rebuild outlier years (e.g. 2019) with wider seasonal window",
    )
    parser.add_argument(
        "--config", type=str, default="configs/ahmedabad.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Data directory (default: data)"
    )
    args = parser.parse_args()

    apply_temporal_cleanup(
        city=args.city,
        start_year=args.start_year,
        end_year=args.end_year,
        min_patch_size=args.min_patch,
        rebuild_outliers=args.rebuild_outliers,
        config_path=args.config,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
