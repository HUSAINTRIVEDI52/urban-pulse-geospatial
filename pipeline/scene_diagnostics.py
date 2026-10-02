"""
UrbanPulse - Sentinel-2 Scene Diagnostics & Radiometric Level Analysis
Evaluates every scene used across all multi-year composites (2018-2024):
1. STAC Metadata: Date, MGRS Tile, s2:processing_baseline, eo:cloud_cover.
2. Stable-Pixel Median Reflectance: Computes median Red, NIR, SWIR16, and Blue reflectance
   (after SCL 1-pixel dilated cloud masking and Baseline 04.00 radiometric scaling/offset)
   over fixed stable pixels (classified as Water or Built-up in >= 6 of 7 years).
3. Flagging: Flags any scene deviating by > 15% from the all-scene median for the same band.
4. Export: Saves data/{city}/scene_diagnostics.csv.
"""

import argparse
import calendar
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import scipy.ndimage
import yaml
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_city_config(city: str, config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads city YAML configuration file."""
    cfg_file = Path(config_path) if config_path else PROJECT_ROOT / "configs" / f"{city.lower()}.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dry_season_range(year: int, config: dict[str, Any], widen_months: int = 0) -> str:
    """Constructs ISO 8601 dry season date range from YAML config."""
    temporal_cfg = config.get("temporal", {})
    dry_season_cfg = temporal_cfg.get("dry_season", {})

    start_month = dry_season_cfg.get("start_month", 11)
    end_month = dry_season_cfg.get("end_month", 2)

    if widen_months > 0:
        start_month = max(1, start_month - widen_months)
        end_month = min(12, end_month + widen_months)

    start_year = year - 1 if start_month > end_month else year
    start_date = f"{start_year:04d}-{start_month:02d}-01"

    _, last_day = calendar.monthrange(year, end_month)
    end_date = f"{year:04d}-{end_month:02d}-{last_day:02d}"

    return f"{start_date}/{end_date}"


def scale_and_harmonize_dn(
    raw_dn: np.ndarray,
    item_datetime: str,
    scale: float | None = None,
    offset: float | None = None,
) -> np.ndarray:
    """Harmonizes raw Sentinel-2 DN to surface reflectance [0.0, 1.0]."""
    date_str = str(item_datetime)[:10]

    if scale is not None and offset is not None:
        reflectance = raw_dn * scale + offset
    else:
        # Sentinel-2 Processing Baseline 04.00 shift (+1000 DN) effective 2022-01-25
        if date_str >= "2022-01-25":
            reflectance = (raw_dn - 1000.0) / 10000.0
        else:
            reflectance = raw_dn / 10000.0

    return np.clip(reflectance, 0.0, 1.0)


def compute_stable_pixels_mask(
    city: str,
    years: list[int],
    data_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Computes a fixed 2D boolean mask of stable pixels across 2018-2024:
    Pixels classified as Built-up (class 1) or Water (class 3) in at least 6 of the 7 years.
    """
    city_key = city.lower()
    classes_dict = {}
    profile = None

    for y in years:
        candidates = [
            data_dir / city_key / "clean" / f"{city_key}_{y}_classified.tif",
            data_dir / "clean" / f"{city_key}_{y}_classified.tif",
            data_dir / city_key / f"{city_key}_{y}_classified.tif",
            data_dir / f"{city_key}_{y}_classified.tif",
        ]
        chosen = next((p for p in candidates if p.exists()), None)
        if not chosen:
            raise FileNotFoundError(f"Missing classified map for {city} in {y}. Looked in: {[str(c) for c in candidates]}")

        with rasterio.open(chosen) as src:
            classes_dict[y] = src.read(1)
            if profile is None:
                profile = src.profile

    first_year = years[0]
    shape = classes_dict[first_year].shape
    stable_counts = np.zeros(shape, dtype=np.int32)

    for y in years:
        arr = classes_dict[y]
        # Water = 3, Built-up = 1
        stable_counts += ((arr == 1) | (arr == 3)).astype(np.int32)

    min_stable_years = max(1, len(years) - 1)  # 6 of 7 years
    stable_mask = stable_counts >= min_stable_years

    return stable_mask, profile


def query_year_scenes(
    city: str,
    year: int,
    config: dict[str, Any],
    data_dir: Path,
    scenes_per_tile: int = 10,
    max_cloud_cover: float = 10.0,
) -> list[Any]:
    """Queries and returns the exact STAC items selected for a given year's composite."""
    city_key = city.lower()
    bbox = config["spatial"]["bbox"]
    stac_url = config.get("stac", {}).get("earth_search_url", "https://earth-search.aws.element84.com/v1")
    collection = config.get("stac", {}).get("collections", {}).get("sentinel_2", "sentinel-2-c1-l2a")

    # Check if widening was recorded in composite_report_{year}.json
    widen_months = 0
    report_candidates = [
        data_dir / city_key / f"composite_report_{year}.json",
        data_dir / f"{city_key}_{year}_composite_report.json",
    ]
    for rp in report_candidates:
        if rp.exists():
            try:
                with open(rp, encoding="utf-8") as f:
                    rdata = json.load(f)
                    widen_months = int(rdata.get("window_widened_months", 0))
                    break
            except Exception:
                pass

    dt_range = get_dry_season_range(year, config, widen_months=widen_months)
    client = Client.open(stac_url)

    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=dt_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())

    if not items or len(items) < 2:
        search = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=dt_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover + 15.0}},
        )
        items = list(search.items())

    if not items and widen_months < 2:
        dt_range = get_dry_season_range(year, config, widen_months=widen_months + 1)
        search = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=dt_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover + 15.0}},
        )
        items = list(search.items())

    # Group by MGRS tile and sort by cloud cover
    tile_dict: dict[str, list[Any]] = {}
    for item in items:
        z = str(item.properties.get("mgrs:utm_zone", ""))
        b = str(item.properties.get("mgrs:latitude_band", ""))
        g = str(item.properties.get("mgrs:grid_square", ""))
        tile_id = f"{z}{b}{g}" if (z and b and g) else item.id.split("_")[1].replace("T", "")
        tile_dict.setdefault(tile_id, []).append(item)

    selected_items = []
    for tile_id, t_items in tile_dict.items():
        t_items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100.0))
        selected_items.extend(t_items[:scenes_per_tile])

    return selected_items


def process_scene_diagnostics_for_city(
    city: str,
    years: list[int] | None = None,
    data_dir: str | Path = "data",
    config_path: str | Path | None = None,
    flag_threshold: float = 0.15,
) -> list[dict[str, Any]]:
    """
    Computes scene-by-scene diagnostics across all composite scenes.
    """
    if years is None:
        years = list(range(2018, 2025))

    data_path = Path(data_dir)
    config = load_city_config(city=city, config_path=config_path)
    city_key = city.lower()
    city_name = config.get("city", {}).get("name", city.capitalize())
    bbox = config["spatial"]["bbox"]

    print("=" * 125, flush=True)
    print(f"[*] UrbanPulse Scene Diagnostics & Radiometric Level Analysis: {city_name} ({years[0]}-{years[-1]})", flush=True)
    print(f"    - Bounding Box      : {bbox}", flush=True)
    print(f"    - Stable Mask Filter: Built-up [1] or Water [3] in >= {len(years)-1} of {len(years)} years", flush=True)
    print(f"    - Flag Threshold    : > {flag_threshold*100:.1f}% deviation from all-scene baseline median", flush=True)
    print("=" * 125, flush=True)

    # 1. Compute stable pixels mask
    print("\n[Step 1/3] Computing multi-temporal stable pixel mask...", flush=True)
    stable_mask, profile = compute_stable_pixels_mask(city=city_key, years=years, data_dir=data_path)
    total_px = stable_mask.size
    stable_px = int(np.sum(stable_mask))
    print(f"[+] Stable pixels: {stable_px:,} / {total_px:,} ({stable_px/total_px*100:.2f}% of metropolitan AOI)", flush=True)

    profile_bounds = array_bounds(profile["height"], profile["width"], profile["transform"])
    profile_shape = (profile["height"], profile["width"])

    # 2. Process scenes year by year
    print("\n[Step 2/3] Processing individual scenes and computing stable-pixel median reflectance...", flush=True)
    requested_assets = ["red", "blue", "nir", "swir16", "scl"]
    target_bands = ["red", "nir", "swir16", "blue"]
    dilation_structure = np.ones((3, 3), dtype=bool)

    scene_records = []

    for year in years:
        items = query_year_scenes(city=city_key, year=year, config=config, data_dir=data_path)
        if not items:
            print(f"    [!] Warning: No STAC items found for {city_name} {year}", flush=True)
            continue

        print(f"  -> Year {year}: Evaluating {len(items)} scenes...", flush=True)

        gdal_env = {
            "AWS_NO_SIGN_REQUEST": "YES",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
            "VSI_CACHE": "TRUE",
            "VSI_CACHE_SIZE": "50000000",
            "GDAL_HTTP_MAX_RETRY": "5",
            "GDAL_HTTP_RETRY_DELAY": "1",
        }

        def fetch_band_data(item_asset_tuple):
            b_name, asset_href, resampling_type = item_asset_tuple
            for attempt in range(1, 4):
                try:
                    with rasterio.Env(**gdal_env):
                        with rasterio.open(asset_href) as src:
                            win = from_bounds(*profile_bounds, transform=src.transform)
                            arr = src.read(1, window=win, out_shape=profile_shape, resampling=resampling_type)
                            return b_name, arr.astype(np.float32)
                except Exception as e:
                    if attempt < 3:
                        time.sleep(1.5)
                    else:
                        print(f"       [!] Failed reading {b_name} from {asset_href}: {e}", flush=True)
                        return b_name, None

        for t_idx, item in enumerate(items):
            dt_str = item.datetime.strftime("%Y-%m-%d") if item.datetime else str(item.properties.get("datetime"))[:10]

            z = str(item.properties.get("mgrs:utm_zone", ""))
            b = str(item.properties.get("mgrs:latitude_band", ""))
            g = str(item.properties.get("mgrs:grid_square", ""))
            tile_id = f"{z}{b}{g}" if (z and b and g) else item.id.split("_")[1].replace("T", "")

            baseline = str(item.properties.get("s2:processing_baseline", "N/A"))
            cloud_cover = float(item.properties.get("eo:cloud_cover", 0.0))

            band_tasks = []
            for b_name in target_bands:
                if b_name in item.assets:
                    band_tasks.append((b_name, item.assets[b_name].href, Resampling.bilinear))
            if "scl" in item.assets:
                band_tasks.append(("scl", item.assets["scl"].href, Resampling.nearest))

            band_arrays = {}
            with ThreadPoolExecutor(max_workers=min(len(band_tasks), 6)) as pool:
                for b_name, arr in pool.map(fetch_band_data, band_tasks):
                    if arr is not None:
                        band_arrays[b_name] = arr

            # Build SCL cloud mask
            scl_arr = band_arrays.get("scl")
            if scl_arr is not None:
                cloud_mask = (
                    (scl_arr == 0)
                    | (scl_arr == 1)
                    | (scl_arr == 3)
                    | (scl_arr == 8)
                    | (scl_arr == 9)
                    | (scl_arr == 10)
                    | (scl_arr == 11)
                    | np.isnan(scl_arr)
                )
                dilated_mask = scipy.ndimage.binary_dilation(cloud_mask, structure=dilation_structure, iterations=1)
            else:
                dilated_mask = np.zeros(stable_mask.shape, dtype=bool)

            scene_medians = {}

            for b_name in target_bands:
                raw_band = band_arrays.get(b_name)
                if raw_band is None:
                    scene_medians[b_name] = np.nan
                    continue

                scale = None
                offset = None
                if b_name in item.assets:
                    extra = item.assets[b_name].extra_fields
                    raster_bands = extra.get("raster:bands", [])
                    if raster_bands and isinstance(raster_bands, list) and len(raster_bands) > 0:
                        scale = raster_bands[0].get("scale")
                        offset = raster_bands[0].get("offset")

                scaled = scale_and_harmonize_dn(raw_band, item_datetime=dt_str, scale=scale, offset=offset)
                scaled[dilated_mask] = np.nan
                scaled[np.isnan(raw_band) | (raw_band <= 0)] = np.nan

                # Compute median over valid stable pixels
                valid_stable = scaled[stable_mask & np.isfinite(scaled)]
                if len(valid_stable) > 0:
                    med_val = float(np.nanmedian(valid_stable))
                else:
                    med_val = np.nan

                scene_medians[b_name] = med_val

            r_val = f"{scene_medians['red']:.4f}" if np.isfinite(scene_medians['red']) else "N/A"
            n_val = f"{scene_medians['nir']:.4f}" if np.isfinite(scene_medians['nir']) else "N/A"
            s_val = f"{scene_medians['swir16']:.4f}" if np.isfinite(scene_medians['swir16']) else "N/A"
            b_val = f"{scene_medians['blue']:.4f}" if np.isfinite(scene_medians['blue']) else "N/A"

            print(
                f"     [{t_idx+1:>2}/{len(items):>2}] {dt_str} ({tile_id:<5}) Baseline={baseline:<5} Cloud={cloud_cover:>5.2f}% -> Medians: R={r_val} N={n_val} S={s_val} B={b_val}",
                flush=True,
            )

            scene_records.append({
                "city": city_key,
                "year": year,
                "scene_id": item.id,
                "date": dt_str,
                "mgrs_tile": tile_id,
                "processing_baseline": baseline,
                "cloud_cover_pct": round(cloud_cover, 2),
                "stable_median_red": round(scene_medians["red"], 4) if np.isfinite(scene_medians["red"]) else None,
                "stable_median_nir": round(scene_medians["nir"], 4) if np.isfinite(scene_medians["nir"]) else None,
                "stable_median_swir16": round(scene_medians["swir16"], 4) if np.isfinite(scene_medians["swir16"]) else None,
                "stable_median_blue": round(scene_medians["blue"], 4) if np.isfinite(scene_medians["blue"]) else None,
            })

    # 3. Compute All-Scene Medians and Flag Anomalies (>15% deviation)
    print("\n[Step 3/3] Calculating all-scene baselines and flagging anomalies (> 15% threshold)...", flush=True)
    all_medians = {}
    for b_name in target_bands:
        col = f"stable_median_{b_name}"
        vals = [r[col] for r in scene_records if r[col] is not None]
        all_medians[b_name] = float(np.median(vals)) if vals else 0.0

    print(f"[+] All-Scene Stable-Pixel Baseline Medians:", flush=True)
    for b_name in target_bands:
        print(f"    - {b_name.upper():<7}: {all_medians[b_name]:.4f}", flush=True)

    flagged_count = 0
    for r in scene_records:
        flags = []
        for b_name in target_bands:
            col = f"stable_median_{b_name}"
            val = r[col]
            base_val = all_medians[b_name]
            flag_key = f"flag_{b_name}"
            if val is not None and base_val > 0:
                diff_pct = (val - base_val) / base_val
                if abs(diff_pct) > flag_threshold:
                    r[flag_key] = f"FLAG ({diff_pct*100:+.1f}%)"
                    flags.append(f"{b_name} ({diff_pct*100:+.1f}%)")
                else:
                    r[flag_key] = "OK"
            else:
                r[flag_key] = "N/A"

        r["flagged"] = "FLAGGED" if len(flags) > 0 else "OK"
        r["flag_details"] = "; ".join(flags) if flags else "Within ±15%"
        if r["flagged"] == "FLAGGED":
            flagged_count += 1

    # 4. Save CSV
    out_dir = data_path / city_key
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = [
        out_dir / "scene_diagnostics.csv",
        data_path / f"{city_key}_scene_diagnostics.csv",
    ]

    fieldnames = [
        "year",
        "date",
        "mgrs_tile",
        "processing_baseline",
        "cloud_cover_pct",
        "stable_median_red",
        "stable_median_nir",
        "stable_median_swir16",
        "stable_median_blue",
        "flag_red",
        "flag_nir",
        "flag_swir16",
        "flag_blue",
        "flagged",
        "flag_details",
        "scene_id",
    ]

    for out_csv in csv_paths:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in scene_records:
                writer.writerow(r)
        print(f"[+] Saved scene diagnostics CSV to: {out_csv.resolve()}", flush=True)

    # 5. Print Formatted Table
    print("\n" + "=" * 135, flush=True)
    print(f"{'Year':<5} {'Date':<10} {'Tile':<6} {'Baseline':<9} {'Cloud%':<7} {'Red':<8} {'NIR':<8} {'SWIR16':<8} {'Blue':<8} {'Status':<10} {'Flag Details'}", flush=True)
    print("-" * 135, flush=True)
    for r in scene_records:
        r_str = f"{r['stable_median_red']:.4f}" if r['stable_median_red'] is not None else "N/A"
        n_str = f"{r['stable_median_nir']:.4f}" if r['stable_median_nir'] is not None else "N/A"
        s_str = f"{r['stable_median_swir16']:.4f}" if r['stable_median_swir16'] is not None else "N/A"
        b_str = f"{r['stable_median_blue']:.4f}" if r['stable_median_blue'] is not None else "N/A"
        status_str = f"[*] {r['flagged']}" if r['flagged'] == "FLAGGED" else "OK"
        print(f"{r['year']:<5} {r['date']:<10} {r['mgrs_tile']:<6} {r['processing_baseline']:<9} {r['cloud_cover_pct']:<7.2f} {r_str:<8} {n_str:<8} {s_str:<8} {b_str:<8} {status_str:<10} {r['flag_details']}", flush=True)
    print("=" * 135, flush=True)
    print(f"[+] Summary: {len(scene_records)} total scenes analyzed. {flagged_count} scene(s) flagged (> 15% deviation).\n", flush=True)

    return scene_records


def main():
    parser = argparse.ArgumentParser(
        description="UrbanPulse - Sentinel-2 Scene-by-Scene Diagnostic & Radiometric Verification"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="ahmedabad",
        help="Target city key (e.g. ahmedabad, pune)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Path to data directory (default: data)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to city YAML config",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Flagging threshold fraction (default: 0.15 for 15 percent)",
    )

    args = parser.parse_args()
    process_scene_diagnostics_for_city(
        city=args.city,
        data_dir=args.data_dir,
        config_path=args.config,
        flag_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
