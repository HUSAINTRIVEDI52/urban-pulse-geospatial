"""
UrbanPulse - Sentinel-2 Satellite Composite Builder
Fetches, cloud-masks, harmonizes reflectance scaling (Baseline 04.00), composites,
and exports surface reflectance GeoTIFFs with SCL 1-pixel dilation, valid observation
thresholding, and automatic seasonal window widening.
"""

import argparse
import calendar
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


def get_dry_season_range(year: int, config: dict[str, Any], widen_months: int = 0) -> str:
    """
    Constructs ISO 8601 dry season date range from YAML config.
    Optionally widens the window by widen_months at both start and end.
    """
    temporal_cfg = config.get("temporal", {})
    dry_season_cfg = temporal_cfg.get("dry_season", {})

    start_month = dry_season_cfg.get("start_month", 10)
    end_month = dry_season_cfg.get("end_month", 3)

    if widen_months > 0:
        start_month = max(1, start_month - widen_months)
        end_month = min(12, end_month + widen_months)

    start_year = year - 1 if start_month > end_month else year
    start_date = f"{start_year:04d}-{start_month:02d}-01"

    _, last_day = calendar.monthrange(year, end_month)
    end_date = f"{year:04d}-{end_month:02d}-{last_day:02d}"

    return f"{start_date}/{end_date}"


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

    # If explicit STAC scale and offset are provided
    if scale is not None and offset is not None:
        reflectance = raw_dn * scale + offset
    else:
        # Baseline 04.00 deployed 2022-01-25 added +1000 DN (+0.1 reflectance)
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
    max_cloud_cover: float = 10.0,
    scenes_per_tile: int = 10,
    min_valid_obs: int = 4,
    max_nodata_threshold_pct: float = 5.0,
    force: bool = False,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
) -> tuple[dict[str, Path], float]:
    """
    Builds a dry-season Sentinel-2 surface reflectance median composite with:
    1. STAC scale & offset harmonization (Baseline 04.00 correction).
    2. SCL cloud, cloud shadow, cirrus, and snow masking dilated by 1 pixel.
    3. Up to 10 scenes per MGRS tile sorted by cloud cover.
    4. Minimum 4 valid observations per pixel threshold.
    5. Automatic 1-month seasonal window widening if NoData > 5%.
    6. Export of composite_report.json.
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
    print(f"[*] UrbanPulse Refactored Composite Builder: {city_name} ({year})")
    print(f"    - Resolution              : {resolution}m (EPSG:32643)")
    print(f"    - Bounding Box            : {bbox}")
    print(f"    - STAC Endpoint           : {stac_url} [{primary_collection}]")
    print(f"    - Max Scenes per Tile     : {scenes_per_tile}")
    print(f"    - Min Valid Obs per Pixel : {min_valid_obs}")
    print(f"    - Output Directory        : {data_path.resolve()}")
    print("=" * 80)

    widen_months = 0
    max_widen_attempts = 2

    while widen_months <= max_widen_attempts:
        datetime_range = get_dry_season_range(year, config, widen_months=widen_months)
        print(f"\n[Step 1/5] Searching dry-season scenes ({datetime_range}, widen={widen_months}mo)...")
        client = Client.open(stac_url)

        search = client.search(
            collections=[primary_collection],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )
        items = list(search.items())

        if not items or len(items) < 2:
            print(f"[*] Widening cloud search filter to < {max_cloud_cover + 15.0}%...")
            search = client.search(
                collections=[primary_collection],
                bbox=bbox,
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": max_cloud_cover + 15.0}},
            )
            items = list(search.items())

        if not items:
            if widen_months < max_widen_attempts:
                widen_months += 1
                continue
            raise RuntimeError(f"No Sentinel-2 scenes found for {city_name} in {year}.")

        # Group by MGRS tile and sort by cloud cover
        tile_dict: dict[str, list[Any]] = {}
        for item in items:
            z = str(item.properties.get("mgrs:utm_zone", ""))
            b = str(item.properties.get("mgrs:latitude_band", ""))
            g = str(item.properties.get("mgrs:grid_square", ""))
            tile_id = f"{z}{b}{g}" if (z and b and g) else item.id.split("_")[1].replace("T", "")
            tile_dict.setdefault(tile_id, []).append(item)

        selected_items = []
        print(f"[+] Found {len(items)} matching scenes across {len(tile_dict)} MGRS tiles:")
        for tile_id, t_items in tile_dict.items():
            t_items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
            chosen = t_items[:scenes_per_tile]
            selected_items.extend(chosen)
            clouds = [round(it.properties.get("eo:cloud_cover", 0), 2) for it in chosen]
            print(f"    - Tile {tile_id:<6}: Selected {len(chosen)} scenes (Clouds: {clouds}%)")

        print(f"[+] Total scenes selected for stack: {len(selected_items)}")

        # 2. Build Stackstac DataArray (raw values without auto-rescaling to control exact math)
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
        with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with ProgressBar(minimum=0.2):
                stack_computed = stack.compute()

        # 4. Process SCL Masking, 1-pixel Dilation, Reflectance Scaling & Temporal Median
        print("\n[Step 4/5] Applying SCL 1-pixel dilated mask and Baseline 04.00 radiometric scaling...")
        n_times, n_bands, n_y, n_x = stack_computed.shape
        band_names = list(stack_computed.band.values)

        scl_idx = band_names.index("scl") if "scl" in band_names else None
        scl_arr = stack_computed.values[:, scl_idx, :, :] if scl_idx is not None else None

        # Prepare container for scaled optical bands
        optical_indices = [band_names.index(b) for b in optical_bands]
        # Shape: (n_bands, n_times, n_y, n_x)
        processed_optical = np.full((len(optical_bands), n_times, n_y, n_x), np.nan, dtype=np.float32)

        # 3x3 footprint for 1-pixel dilation
        dilation_structure = np.ones((3, 3), dtype=bool)

        for t_idx, item in enumerate(selected_items):
            item_dt = item.datetime or str(item.properties.get("datetime"))[:10]

            # Build SCL cloud/shadow mask for this scene
            if scl_arr is not None:
                scl_t = scl_arr[t_idx]
                # SCL classes to mask: 0=NoData, 1=Saturated, 3=Shadow, 8=Medium Cloud, 9=High Cloud, 10=Cirrus, 11=Snow
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
                # Dilate by 1 pixel to remove cloud edges & fringe shadows
                dilated_mask = scipy.ndimage.binary_dilation(
                    cloud_shadow_mask, structure=dilation_structure, iterations=1
                )
            else:
                dilated_mask = np.zeros((n_y, n_x), dtype=bool)

            # Scale and mask each optical band
            for b_i, b_name in enumerate(optical_bands):
                orig_b_idx = band_names.index(b_name)
                raw_band = stack_computed.values[t_idx, orig_b_idx, :, :].astype(np.float32)

                # Check asset metadata for scale & offset
                scale = None
                offset = None
                if b_name in item.assets:
                    extra = item.assets[b_name].extra_fields
                    raster_bands = extra.get("raster:bands", [])
                    if raster_bands and isinstance(raster_bands, list) and len(raster_bands) > 0:
                        scale = raster_bands[0].get("scale")
                        offset = raster_bands[0].get("offset")

                scaled = scale_and_harmonize_dn(raw_band, item_datetime=item_dt, scale=scale, offset=offset)

                # Apply dilated cloud/shadow mask
                scaled[dilated_mask] = np.nan
                scaled[np.isnan(raw_band) | (raw_band <= 0)] = np.nan

                processed_optical[b_i, t_idx, :, :] = scaled

        # 5. Temporal Median and Minimum Valid Observations Check
        final_composite = {}
        band_means = {}
        total_grid_pixels = n_y * n_x

        # Compute valid observation counts per pixel across time
        # Shape: (n_y, n_x)
        valid_obs_counts = np.sum(~np.isnan(processed_optical[0, :, :, :]), axis=0)

        # Effective min valid obs: cannot exceed the number of scenes per tile available
        max_depth = max(1, len(selected_items) // max(1, len(tile_dict)))
        effective_min_obs = min(min_valid_obs, max_depth)

        for b_i, b_name in enumerate(optical_bands):
            band_time_series = processed_optical[b_i, :, :, :]
            # Per-pixel median across valid observations
            median_band = np.nanmedian(band_time_series, axis=0)
            # Require minimum valid observations per pixel
            median_band[valid_obs_counts < effective_min_obs] = np.nan

            final_composite[b_name] = median_band
            valid_pixels = median_band[np.isfinite(median_band) & (median_band > 0)]
            band_means[b_name] = float(np.mean(valid_pixels)) if len(valid_pixels) > 0 else 0.0

        # Calculate NoData % on the final composite
        nan_pixels = int(np.isnan(final_composite["red"]).sum())
        nodata_percentage = (nan_pixels / total_grid_pixels) * 100.0

        print("\n" + "=" * 80)
        print(f"[*] Composite Quality Check (Window: {datetime_range}):")
        print(f"    - Total Pixels        : {total_grid_pixels:>10,}")
        print(f"    - Valid Data Pixels   : {total_grid_pixels - nan_pixels:>10,} ({(100 - nodata_percentage):.2f}%)")
        print(f"    - NoData Pixels       : {nan_pixels:>10,} ({nodata_percentage:.4f}%)")
        print(f"    - Min Valid Obs Gate  : {effective_min_obs} observations per pixel (requested: {min_valid_obs})")
        print("=" * 80)

        # If NoData > threshold and we haven't reached max widening, widen window
        if nodata_percentage > max_nodata_threshold_pct and widen_months < max_widen_attempts:
            print(f"[!] Warning: NoData is {nodata_percentage:.2f}% > {max_nodata_threshold_pct}%. Widening seasonal window by 1 month...")
            widen_months += 1
            continue

        # Otherwise accept composite
        break

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
    scene_dates = sorted(list({
        (it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10])
        for it in selected_items
    }))
    unique_tiles = sorted(list(tile_dict.keys()))
    cloud_vals = [float(it.properties.get("eo:cloud_cover", 0)) for it in selected_items]
    mean_cloud = float(np.mean(cloud_vals)) if cloud_vals else 0.0

    report_data = {
        "city": city_key,
        "year": year,
        "datetime_window": datetime_range,
        "window_widened_months": widen_months,
        "scene_count": len(selected_items),
        "scene_dates": scene_dates,
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
        "--max-cloud", type=float, default=10.0, help="Max cloud cover percentage (default: 10.0)"
    )
    parser.add_argument(
        "--scenes-per-tile", type=int, default=10, help="Scenes per MGRS tile (default: 10)"
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
