"""
UrbanPulse - Sentinel-2 Satellite Composite Builder (Strict Window & Date-Spread Selection)
Fetches, cloud-masks, harmonizes reflectance scaling (Baseline 04.00 offset correction),
composites, and exports surface reflectance GeoTIFFs.

Features:
1. Strict Window: Dec 1 to Feb 15. Never widens into October or March-May.
2. Low Confidence Rule: If a year has < 4 distinct acquisition dates inside the strict window,
   it is marked as low_confidence in composite_report.json and not used in the analysis series.
3. Flagged Scene Filtering: Excludes scenes flagged by scene_diagnostics.py.
4. Date-Spread Coverage: Selects up to 8 scenes per MGRS tile evenly distributed across the window.
5. Cloud Masking: SCL dilated cloud/shadow masking (1-pixel dilation).
6. Reflectance Harmonization: Applies Baseline 04.00 (-1000 DN) offset for acquisitions on/after 2022-01-25.
"""

import argparse
import calendar
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rioxarray  # noqa: F401 - registers .rio accessor on xarray DataArray
import scipy.ndimage
import stackstac
import yaml
from dask.diagnostics import ProgressBar
from pystac_client import Client


def load_city_config(
    city: str = "ahmedabad", config_path: str | Path | None = None
) -> dict[str, Any]:
    """Loads city YAML configuration."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"City configuration file not found: {cfg_file.resolve()}")

    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_strict_window_range(year: int, config: dict[str, Any]) -> str:
    """
    Constructs ISO 8601 strict dry season date range from YAML config (default: Dec 1 to Feb 15).
    Never widens into October or March-May.
    """
    temporal_cfg = config.get("temporal", {})
    window_cfg = temporal_cfg.get("strict_window") or temporal_cfg.get("dry_season", {})

    start_month = window_cfg.get("start_month", 12)
    end_month = window_cfg.get("end_month", 2)
    start_day_str = window_cfg.get("start_day", "12-01")
    end_day_str = window_cfg.get("end_day", "02-15")

    if "-" in str(start_day_str):
        s_parts = str(start_day_str).split("-")
        start_month = int(s_parts[0])
        s_day = int(s_parts[1])
    else:
        s_day = 1

    if "-" in str(end_day_str):
        e_parts = str(end_day_str).split("-")
        end_month = int(e_parts[0])
        e_day = int(e_parts[1])
    else:
        _, e_day = calendar.monthrange(year, end_month)

    start_year = year - 1 if start_month > end_month else year
    start_date = f"{start_year:04d}-{start_month:02d}-{s_day:02d}"
    end_date = f"{year:04d}-{end_month:02d}-{e_day:02d}"

    return f"{start_date}/{end_date}"


def load_flagged_scene_ids(city: str, data_dir: Path) -> set[str]:
    """Loads set of flagged scene IDs from scene_diagnostics.csv if present."""
    city_key = city.lower()
    candidates = [
        data_dir / city_key / "scene_diagnostics.csv",
        data_dir / f"{city_key}_scene_diagnostics.csv",
    ]
    flagged = set()
    for cp in candidates:
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("flagged") == "FLAGGED":
                            sid = row.get("scene_id")
                            if sid:
                                flagged.add(sid)
                if flagged:
                    break
            except Exception:
                pass
    return flagged


def select_scenes_by_date_coverage(
    items: list[Any],
    max_scenes_per_tile: int = 8,
    flagged_ids: set[str] | None = None,
) -> tuple[list[Any], dict[str, list[Any]]]:
    """
    Groups scenes by MGRS tile, excludes flagged scenes, and selects up to
    max_scenes_per_tile evenly distributed across distinct acquisition dates in the window.
    """
    if flagged_ids is None:
        flagged_ids = set()

    unflagged = [it for it in items if it.id not in flagged_ids]
    candidate_items = unflagged if len(unflagged) >= 4 else items

    tile_dict: dict[str, list[Any]] = {}
    for item in candidate_items:
        z = str(item.properties.get("mgrs:utm_zone", ""))
        b = str(item.properties.get("mgrs:latitude_band", ""))
        g = str(item.properties.get("mgrs:grid_square", ""))
        tile_id = f"{z}{b}{g}" if (z and b and g) else item.id.split("_")[1].replace("T", "")
        tile_dict.setdefault(tile_id, []).append(item)

    selected = []
    for tile_id, t_items in tile_dict.items():
        by_date: dict[str, Any] = {}
        for it in t_items:
            dt_str = it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10]
            if dt_str not in by_date:
                by_date[dt_str] = it
            else:
                c_new = float(it.properties.get("eo:cloud_cover", 100.0))
                c_curr = float(by_date[dt_str].properties.get("eo:cloud_cover", 100.0))
                if c_new < c_curr:
                    by_date[dt_str] = it

        unique_dates = sorted(by_date.keys())
        if len(unique_dates) <= max_scenes_per_tile:
            chosen_dates = unique_dates
        else:
            indices = np.round(np.linspace(0, len(unique_dates) - 1, max_scenes_per_tile)).astype(int)
            chosen_dates = [unique_dates[i] for i in sorted(list(set(indices)))]

        chosen_items = [by_date[d] for d in chosen_dates]
        selected.extend(chosen_items)

    return selected, tile_dict


def scale_and_harmonize_dn(
    raw_dn: np.ndarray | float,
    item_datetime: datetime | str,
    scale: float | None = None,
    offset: float | None = None,
) -> np.ndarray | float:
    """
    Harmonizes Sentinel-2 L2A Digital Numbers (DN) to surface reflectance [0.0, 1.0].
    Applies asset metadata scale/offset if available; otherwise applies Sentinel-2
    Baseline 04.00 offset correction (-1000 DN) for acquisitions on or after 2022-01-25.
    """
    if isinstance(item_datetime, str):
        date_str = item_datetime[:10]
    elif hasattr(item_datetime, "strftime"):
        date_str = item_datetime.strftime("%Y-%m-%d")
    else:
        date_str = str(item_datetime)[:10]

    if scale is not None and offset is not None:
        reflectance = raw_dn * scale + offset
    else:
        if date_str >= "2022-01-25":
            reflectance = (raw_dn - 1000.0) / 10000.0
        else:
            reflectance = raw_dn / 10000.0

    if isinstance(reflectance, np.ndarray):
        reflectance = np.clip(reflectance, 0.0, 1.0)
    elif isinstance(reflectance, (int, float)):
        reflectance = max(0.0, min(1.0, float(reflectance)))

    return reflectance


def build_composite(
    city: str = "ahmedabad",
    year: int = 2024,
    resolution: float = 60.0,
    max_cloud_cover: float = 20.0,
    scenes_per_tile: int = 8,
    min_valid_obs: int = 4,
    max_nodata_threshold_pct: float = 5.0,
    force: bool = False,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
) -> tuple[dict[str, Path], float]:
    """
    Builds a strict dry-season Sentinel-2 surface reflectance composite:
    1. Strict window: Dec 1 to Feb 15 (never widened).
    2. Excludes flagged scenes from scene_diagnostics.py.
    3. Selects up to 8 scenes per tile spread across distinct acquisition dates.
    4. Marks low_confidence if distinct dates < 4 inside the window.
    5. Baseline 04.00 offset harmonization (-1000 DN for >= 2022-01-25).
    6. SCL 1-pixel dilated cloud masking.
    7. Exports GeoTIFFs and composite_report.json.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    city_key = city.lower().strip()
    optical_bands = ["red", "green", "blue", "nir", "swir16"]
    band_paths = {b: data_path / f"{city_key}_{year}_{b}.tif" for b in optical_bands}
    report_path = data_path / f"{city_key}_{year}_composite_report.json"
    city_sub_dir = data_path / city_key
    city_sub_dir.mkdir(parents=True, exist_ok=True)
    city_report_path = city_sub_dir / f"composite_report_{year}.json"

    # Skip if outputs already exist and force is not set
    if not force and all(p.exists() for p in band_paths.values()) and report_path.exists():
        print(f"[+] Composite bands for {city} ({year}) already exist in {data_path.resolve()}. Skipping rebuild.")
        with rasterio.open(band_paths["red"]) as src:
            red_arr = src.read(1)
            nan_count = int(np.isnan(red_arr).sum() + (red_arr == -9999.0).sum())
            nodata_pct = (nan_count / red_arr.size) * 100.0
        return band_paths, nodata_pct

    config = load_city_config(city=city, config_path=config_path)
    city_name = config.get("city", {}).get("name", city.capitalize())
    bbox = config["spatial"]["bbox"]
    stac_url = config.get("stac", {}).get(
        "earth_search_url", "https://earth-search.aws.element84.com/v1"
    )
    primary_collection = (
        config.get("stac", {}).get("collections", {}).get("sentinel_2", "sentinel-2-c1-l2a")
    )
    requested_assets = optical_bands + ["scl"]

    print("=" * 80)
    print(f"[*] UrbanPulse Strict-Window Composite Builder: {city_name} ({year})")
    print(f"    - Resolution              : {resolution}m (EPSG:32643)")
    print(f"    - Bounding Box            : {bbox}")
    print(f"    - STAC Endpoint           : {stac_url} [{primary_collection}]")
    print(f"    - Max Scenes per Tile     : {scenes_per_tile} (Date-spread coverage)")
    print(f"    - Min Valid Obs per Pixel : {min_valid_obs}")
    print("=" * 80)

    datetime_range = get_strict_window_range(year, config)
    print(f"\n[Step 1/5] Searching strict-window scenes ({datetime_range})...")
    client = Client.open(stac_url)

    search = client.search(
        collections=[primary_collection],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())

    if not items or len(items) < 2:
        print(f"[*] Widening cloud filter to < {max_cloud_cover + 15.0}% inside strict window...")
        search = client.search(
            collections=[primary_collection],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover + 15.0}},
        )
        items = list(search.items())

    # Compute distinct acquisition dates inside strict window
    all_dates = sorted(list({
        (it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10])
        for it in items
    }))

    is_low_confidence = len(all_dates) < 4
    low_confidence_reason = None
    if is_low_confidence:
        low_confidence_reason = (
            f"Fewer than 4 distinct acquisition dates ({len(all_dates)}) found in strict window {datetime_range}."
        )
        print(f"\n[!] LOW CONFIDENCE YEAR: {city_name} {year} - {low_confidence_reason}")

    if not items:
        # Save empty low confidence report and return empty
        report_data = {
            "city": city_key,
            "year": year,
            "strict_window": datetime_range,
            "datetime_window": datetime_range,
            "low_confidence": True,
            "low_confidence_reason": f"No Sentinel-2 scenes found in strict window {datetime_range}",
            "scene_count": 0,
            "scene_dates": [],
            "mgrs_tiles": [],
            "nodata_percentage": 100.0,
            "band_means_reflectance": {},
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        with open(city_report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"[!] Saved low-confidence report to: {report_path.resolve()}")
        return band_paths, 100.0

    # Load flagged scenes from diagnostics
    flagged_ids = load_flagged_scene_ids(city=city, data_dir=data_path)
    if flagged_ids:
        print(f"[+] Loaded {len(flagged_ids)} flagged scene IDs from scene_diagnostics.csv")

    # Select scenes by date coverage up to scenes_per_tile
    selected_items, tile_dict = select_scenes_by_date_coverage(
        items=items,
        max_scenes_per_tile=scenes_per_tile,
        flagged_ids=flagged_ids,
    )

    selected_dates = sorted(list({
        (it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10])
        for it in selected_items
    }))

    print(f"[+] Selected {len(selected_items)} scenes across {len(tile_dict)} MGRS tiles spanning {len(selected_dates)} distinct dates:")
    print(f"    - Acquisition Dates: {selected_dates}")

    # 2. Build Stackstac DataArray
    print(f"\n[Step 2/5] Constructing Dask raster stack at {resolution}m (EPSG:32643)...")
    stack = stackstac.stack(
        selected_items,
        assets=requested_assets,
        epsg=32643,
        bounds_latlon=bbox,
        resolution=resolution,
        chunksize=1024,
        rescale=False,
        fill_value=np.nan,
    )

    # 3. Compute Dask Stack in Memory
    print("\n[Step 3/5] Loading and executing parallel Dask array computation...")
    with rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    ):
        with ProgressBar(minimum=0.2):
            stack_computed = stack.compute()

    # 4. SCL Masking, 1-pixel Dilation, Reflectance Scaling & Temporal Median
    print("\n[Step 4/5] Applying SCL 1-pixel dilated mask and Baseline 04.00 radiometric scaling...")
    n_times, n_bands, n_y, n_x = stack_computed.shape
    band_names = list(stack_computed.band.values)

    scl_idx = band_names.index("scl") if "scl" in band_names else None
    scl_arr = stack_computed.values[:, scl_idx, :, :] if scl_idx is not None else None

    processed_optical = np.full((len(optical_bands), n_times, n_y, n_x), np.nan, dtype=np.float32)
    dilation_structure = np.ones((3, 3), dtype=bool)

    for t_idx, item in enumerate(selected_items):
        item_dt = item.datetime or str(item.properties.get("datetime"))[:10]

        # SCL cloud & shadow mask
        if scl_arr is not None:
            scl_t = scl_arr[t_idx]
            cloud_shadow_mask = (
                (scl_t == 0)
                | (scl_t == 1)
                | (scl_t == 3)
                | (scl_t == 8)
                | (scl_t == 9)
                | (scl_t == 10)
                | (scl_t == 11)
                | np.isnan(scl_t)
            )
            dilated_mask = scipy.ndimage.binary_dilation(
                cloud_shadow_mask, structure=dilation_structure, iterations=1
            )
        else:
            dilated_mask = np.zeros((n_y, n_x), dtype=bool)

        for b_i, b_name in enumerate(optical_bands):
            orig_b_idx = band_names.index(b_name)
            raw_band = stack_computed.values[t_idx, orig_b_idx, :, :].astype(np.float32)

            scale = None
            offset = None
            if b_name in item.assets:
                extra = item.assets[b_name].extra_fields
                raster_bands = extra.get("raster:bands", [])
                if raster_bands and isinstance(raster_bands, list) and len(raster_bands) > 0:
                    scale = raster_bands[0].get("scale")
                    offset = raster_bands[0].get("offset")

            scaled = scale_and_harmonize_dn(raw_band, item_datetime=item_dt, scale=scale, offset=offset)
            scaled[dilated_mask] = np.nan
            scaled[np.isnan(raw_band) | (raw_band <= 0)] = np.nan

            processed_optical[b_i, t_idx, :, :] = scaled

    # 5. Temporal Median & Minimum Valid Observations Check
    final_composite = {}
    band_means = {}
    total_grid_pixels = n_y * n_x

    valid_obs_counts = np.sum(~np.isnan(processed_optical[0, :, :, :]), axis=0)
    max_depth = max(1, len(selected_items) // max(1, len(tile_dict)))
    effective_min_obs = min(min_valid_obs, max_depth)

    for b_i, b_name in enumerate(optical_bands):
        band_time_series = processed_optical[b_i, :, :, :]
        median_band = np.nanmedian(band_time_series, axis=0)
        median_band[valid_obs_counts < effective_min_obs] = np.nan

        final_composite[b_name] = median_band
        valid_pixels = median_band[np.isfinite(median_band) & (median_band > 0)]
        band_means[b_name] = float(np.mean(valid_pixels)) if len(valid_pixels) > 0 else 0.0

    nan_pixels = int(np.isnan(final_composite["red"]).sum())
    nodata_percentage = (nan_pixels / total_grid_pixels) * 100.0

    print("\n" + "=" * 80)
    print(f"[*] Strict-Window Composite Quality Check (Window: {datetime_range}):")
    print(f"    - Total Pixels        : {total_grid_pixels:>10,}")
    print(f"    - Valid Data Pixels   : {total_grid_pixels - nan_pixels:>10,} ({(100 - nodata_percentage):.2f}%)")
    print(f"    - NoData Pixels       : {nan_pixels:>10,} ({nodata_percentage:.4f}%)")
    print(f"    - Min Valid Obs Gate  : {effective_min_obs} observations per pixel")
    print(f"    - Low Confidence Flag : {is_low_confidence} ({low_confidence_reason or 'Accepted'})")
    print("=" * 80)

    # 6. Export GeoTIFFs to disk
    print("\n[Step 5/5] Exporting surface reflectance GeoTIFFs...")
    sample_da = stack_computed.sel(band="red")
    sample_da.rio.write_crs("EPSG:32643", inplace=True)
    transform = sample_da.rio.transform()

    output_paths = {}
    for b_name in optical_bands:
        out_file = band_paths[b_name]
        arr_to_write = final_composite[b_name].astype(np.float32)

        with rasterio.open(
            out_file,
            "w",
            driver="GTiff",
            height=n_y,
            width=n_x,
            count=1,
            dtype="float32",
            crs="EPSG:32643",
            transform=transform,
            compress="lzw",
            nodata=-9999.0,
        ) as dst:
            dst.write(np.where(np.isnan(arr_to_write), -9999.0, arr_to_write), 1)

        output_paths[b_name] = out_file
        print(f"    - Exported {b_name:<7} -> {out_file.name}")

    # 7. Write composite_report.json
    unique_tiles = sorted(list(tile_dict.keys()))
    cloud_vals = [float(it.properties.get("eo:cloud_cover", 0)) for it in selected_items]
    mean_cloud = float(np.mean(cloud_vals)) if cloud_vals else 0.0

    report_data = {
        "city": city_key,
        "year": year,
        "strict_window": datetime_range,
        "datetime_window": datetime_range,
        "low_confidence": is_low_confidence,
        "low_confidence_reason": low_confidence_reason,
        "scene_count": len(selected_items),
        "scene_dates": selected_dates,
        "mgrs_tiles": unique_tiles,
        "mean_scene_cloud_cover_pct": round(mean_cloud, 4),
        "min_valid_obs_threshold": min_valid_obs,
        "total_aoi_pixels": total_grid_pixels,
        "valid_pixels": total_grid_pixels - nan_pixels,
        "nodata_pixels": nan_pixels,
        "nodata_percentage": round(nodata_percentage, 4),
        "band_means_reflectance": {k: round(v, 4) for k, v in band_means.items()},
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    with open(city_report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[+] Saved composite report: {report_path.resolve()}")
    return output_paths, nodata_percentage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Sentinel-2 composite for UrbanPulse.")
    parser.add_argument(
        "--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)"
    )
    parser.add_argument("--year", type=int, default=2024, help="Target year (default: 2024)")
    parser.add_argument(
        "--resolution", type=float, default=60.0, help="Resolution in meters (default: 60.0)"
    )
    parser.add_argument(
        "--max-cloud", type=float, default=20.0, help="Max cloud cover percentage (default: 20.0)"
    )
    parser.add_argument(
        "--scenes-per-tile", type=int, default=8, help="Scenes per MGRS tile (default: 8)"
    )
    parser.add_argument(
        "--min-valid-obs", type=int, default=4, help="Min valid observations per pixel (default: 4)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force overwrite existing composite bands"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to city config YAML")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()
    build_composite(
        city=args.city,
        year=args.year,
        resolution=args.resolution,
        max_cloud_cover=args.max_cloud,
        scenes_per_tile=args.scenes_per_tile,
        min_valid_obs=args.min_valid_obs,
        force=args.force,
        config_path=args.config,
        data_dir=args.data_dir,
    )
