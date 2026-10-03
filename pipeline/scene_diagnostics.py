"""
UrbanPulse - Sentinel-2 Scene Diagnostics & Radiometric Level Analysis
Strict Window (Nov 1 - Feb 28), No Scene Cap, Per-Tile Threshold & Quality Screening:
1. Window: Strictly Nov 1 to Feb 28/29 (no widening into Oct or March).
2. Archive Gap: 2022 is marked as an archive-gap year (no composite).
3. STAC Metadata: Date, MGRS Tile, s2:processing_baseline, eo:cloud_cover.
4. Stable-Pixel Analysis: Computes valid stable-pixel count and median Red, NIR, SWIR16,
   and Blue reflectance (after SCL 1-pixel dilated cloud masking and radiometric scaling)
   over fixed stable pixels (Water or Built-up in >= 6 of 7 years).
5. Quality Filtering & Dropping:
   - Drops any scene with < 50% of its tile's median valid stable-pixel count.
   - Drops any scene whose stable-pixel median differs > 25% from its own tile's median in any band.
6. Export: Saves data/{city}/scene_diagnostics.csv.
"""

import argparse
import calendar
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    cfg_file = (
        Path(config_path) if config_path else PROJECT_ROOT / "configs" / f"{city.lower()}.yaml"
    )
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_strict_dry_season_range(year: int) -> str:
    """
    Constructs strict ISO 8601 dry season date range: Nov 1 (Y-1) to Feb 28/29 (Y).
    Never widens.
    """
    start_year = year - 1
    start_date = f"{start_year:04d}-11-01"

    _, last_day = calendar.monthrange(year, 2)
    end_date = f"{year:04d}-02-{last_day:02d}"

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
        # Fallback date check for Baseline 04.00
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
    Computes a fixed 2D boolean mask of stable pixels across multi-year series:
    Pixels classified as Built-up (class 1) or Water (class 3) in at least 6 of the 7 years.
    """
    city_key = city.lower()
    classes_dict = {}
    profile = None
    eval_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    for y in eval_years:
        candidates = [
            data_dir / city_key / "clean" / f"{city_key}_{y}_classified.tif",
            data_dir / "clean" / f"{city_key}_{y}_classified.tif",
            data_dir / city_key / f"{city_key}_{y}_classified.tif",
            data_dir / f"{city_key}_{y}_classified.tif",
        ]
        chosen = next((p for p in candidates if p.exists()), None)
        if not chosen:
            raise FileNotFoundError(
                f"Missing classified map for {city} in {y}. Looked in: {[str(c) for c in candidates]}"
            )

        with rasterio.open(chosen) as src:
            classes_dict[y] = src.read(1)
            if profile is None:
                profile = src.profile

    first_year = eval_years[0]
    shape = classes_dict[first_year].shape
    stable_counts = np.zeros(shape, dtype=np.int32)

    for y in eval_years:
        arr = classes_dict[y]
        # Water = 3, Built-up = 1
        stable_counts += ((arr == 1) | (arr == 3)).astype(np.int32)

    min_stable_years = max(1, len(eval_years) - 1)  # 6 of 7 years
    stable_mask = stable_counts >= min_stable_years

    return stable_mask, profile


def query_strict_year_scenes(
    city: str,
    year: int,
    config: dict[str, Any],
    max_cloud_cover: float = 20.0,
) -> list[Any]:
    """
    Queries all Sentinel-2 L2A STAC scenes within strict Nov 1 - Feb 28 window.
    No 20-scene cap.
    """
    if year == 2022:
        return []

    bbox = config["spatial"]["bbox"]
    stac_url = config.get("stac", {}).get(
        "earth_search_url", "https://earth-search.aws.element84.com/v1"
    )
    collection = (
        config.get("stac", {}).get("collections", {}).get("sentinel_2", "sentinel-2-c1-l2a")
    )

    dt_range = get_strict_dry_season_range(year)
    client = Client.open(stac_url)

    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=dt_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())

    items.sort(
        key=lambda it: (
            (
                it.datetime.strftime("%Y-%m-%d")
                if it.datetime
                else str(it.properties.get("datetime"))[:10]
            ),
            float(it.properties.get("eo:cloud_cover", 100.0)),
        )
    )
    return items


def process_single_scene(
    item: Any,
    year: int,
    city_key: str,
    profile_bounds: tuple[float, float, float, float],
    profile_shape: tuple[int, int],
    stable_mask: np.ndarray,
) -> dict[str, Any]:
    """Processes a single STAC item to extract stable pixel medians."""
    dt_str = (
        item.datetime.strftime("%Y-%m-%d")
        if item.datetime
        else str(item.properties.get("datetime"))[:10]
    )

    z = str(item.properties.get("mgrs:utm_zone", ""))
    b = str(item.properties.get("mgrs:latitude_band", ""))
    g = str(item.properties.get("mgrs:grid_square", ""))
    tile_id = f"{z}{b}{g}" if (z and b and g) else item.id.split("_")[1].replace("T", "")

    baseline = str(item.properties.get("s2:processing_baseline", "N/A"))
    cloud_cover = float(item.properties.get("eo:cloud_cover", 0.0))

    target_bands = ["red", "nir", "swir16", "blue"]
    dilation_structure = np.ones((3, 3), dtype=bool)

    gdal_env = {
        "AWS_NO_SIGN_REQUEST": "YES",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "VSI_CACHE": "TRUE",
        "VSI_CACHE_SIZE": "50000000",
        "GDAL_HTTP_MAX_RETRY": "5",
        "GDAL_HTTP_RETRY_DELAY": "1",
    }

    band_arrays = {}
    for b_name in target_bands + ["scl"]:
        if b_name not in item.assets:
            continue
        asset_href = item.assets[b_name].href
        resampling_type = Resampling.nearest if b_name == "scl" else Resampling.bilinear
        for attempt in range(1, 4):
            try:
                with rasterio.Env(**gdal_env):
                    with rasterio.open(asset_href) as src:
                        win = from_bounds(*profile_bounds, transform=src.transform)
                        arr = src.read(
                            1, window=win, out_shape=profile_shape, resampling=resampling_type
                        )
                        band_arrays[b_name] = arr.astype(np.float32)
                        break
            except Exception:
                if attempt < 3:
                    time.sleep(0.5)

    # Cloud masking
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
        dilated_mask = scipy.ndimage.binary_dilation(
            cloud_mask, structure=dilation_structure, iterations=1
        )
    else:
        dilated_mask = np.zeros(stable_mask.shape, dtype=bool)

    scene_medians = {}
    valid_counts = []

    for b_name in target_bands:
        raw_band = band_arrays.get(b_name)
        if raw_band is None:
            scene_medians[b_name] = None
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

        valid_mask = stable_mask & np.isfinite(scaled) & (scaled > 0)
        valid_px = scaled[valid_mask]
        valid_counts.append(int(np.sum(valid_mask)))

        med_val = float(np.nanmedian(valid_px)) if len(valid_px) > 0 else None
        scene_medians[b_name] = round(med_val, 4) if med_val is not None else None

    valid_stable_px = int(np.median(valid_counts)) if valid_counts else 0

    return {
        "city": city_key,
        "year": year,
        "scene_id": item.id,
        "date": dt_str,
        "mgrs_tile": tile_id,
        "processing_baseline": baseline,
        "cloud_cover_pct": round(cloud_cover, 2),
        "valid_stable_pixels": valid_stable_px,
        "stable_median_red": scene_medians.get("red"),
        "stable_median_nir": scene_medians.get("nir"),
        "stable_median_swir16": scene_medians.get("swir16"),
        "stable_median_blue": scene_medians.get("blue"),
    }


def process_scene_diagnostics_for_city(
    city: str,
    years: list[int] | None = None,
    data_dir: str | Path = "data",
    config_path: str | Path | None = None,
    max_cloud_cover: float = 20.0,
    max_workers: int = 16,
) -> list[dict[str, Any]]:
    """
    Evaluates every Sentinel-2 scene in the strict Nov 1 - Feb 28 window (2018-2024, excluding 2022).
    Applies per-tile median benchmarking and drops anomalous or cloud-depleted scenes.
    """
    if years is None:
        years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

    data_path = Path(data_dir)
    config = load_city_config(city=city, config_path=config_path)
    city_key = city.lower()
    city_name = config.get("city", {}).get("name", city.capitalize())

    print("=" * 140, flush=True)
    print(
        f"[*] UrbanPulse Strict-Window Scene Diagnostics & Tile Quality Screening: {city_name}",
        flush=True,
    )
    print("    - Strict Window     : Nov 1 to Feb 28/29 (No widening)", flush=True)
    print("    - Scene Cap         : None (All valid in-window scenes evaluated)", flush=True)
    print("    - 2022 Status       : ARCHIVE-GAP YEAR (Skipped / No composite)", flush=True)
    print("    - Stable Mask Filter: Built-up [1] or Water [3] in >= 6 of 7 years", flush=True)
    print("    - Drop Rule 1       : < 50% of tile's median valid stable-pixel count", flush=True)
    print(
        "    - Drop Rule 2       : > 25% deviation from own tile's median reflectance in any band",
        flush=True,
    )
    print("=" * 140, flush=True)

    # 1. Compute stable pixels mask
    print("\n[Step 1/3] Computing multi-temporal stable pixel mask...", flush=True)
    stable_mask, profile = compute_stable_pixels_mask(
        city=city_key, years=years, data_dir=data_path
    )
    total_px = stable_mask.size
    stable_px = int(np.sum(stable_mask))
    print(
        f"[+] Stable pixels: {stable_px:,} / {total_px:,} ({stable_px/total_px*100:.2f}% of metropolitan AOI)",
        flush=True,
    )

    profile_bounds = array_bounds(profile["height"], profile["width"], profile["transform"])
    profile_shape = (profile["height"], profile["width"])

    # 2. Query all scenes across all years
    print(
        "\n[Step 2/3] Querying strict in-window scenes and processing remote assets in parallel...",
        flush=True,
    )
    tasks = []
    yearly_counts = {}

    for year in years:
        if year == 2022:
            yearly_counts[year] = {"scenes": 0, "dates": 0, "dates_list": []}
            continue

        items = query_strict_year_scenes(
            city=city_key, year=year, config=config, max_cloud_cover=max_cloud_cover
        )
        dates_list = sorted(
            list(
                {
                    (
                        it.datetime.strftime("%Y-%m-%d")
                        if it.datetime
                        else str(it.properties.get("datetime"))[:10]
                    )
                    for it in items
                }
            )
        )
        yearly_counts[year] = {
            "scenes": len(items),
            "dates": len(dates_list),
            "dates_list": dates_list,
        }

        print(
            f"  -> Year {year}: Found {len(items)} scenes across {len(dates_list)} distinct dates ({get_strict_dry_season_range(year)})",
            flush=True,
        )

        for it in items:
            tasks.append((it, year))

    print(
        f"\n[+] Executing {len(tasks)} scene evaluations using {max_workers} concurrent threads...",
        flush=True,
    )

    raw_scene_records = []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            executor.submit(
                process_single_scene,
                item=it,
                year=yr,
                city_key=city_key,
                profile_bounds=profile_bounds,
                profile_shape=profile_shape,
                stable_mask=stable_mask,
            ): it.id
            for it, yr in tasks
        }
        done_count = 0
        for f in as_completed(futures):
            try:
                res = f.result()
                raw_scene_records.append(res)
            except Exception as exc:
                print(f"[!] Scene evaluation failed: {exc}", flush=True)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                print(f"    - Processed {done_count:>3}/{len(tasks)} scenes...", flush=True)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    # Sort records deterministically by year, date, tile
    raw_scene_records.sort(
        key=lambda r: (
            r.get("year", 0),
            r.get("date", ""),
            r.get("mgrs_tile", ""),
            r.get("scene_id", ""),
        )
    )

    # 3. Compute Per-Tile Medians and Apply Dropping Rules
    print(
        "\n[Step 3/3] Calculating per-tile medians and screening for dropped scenes...", flush=True
    )
    target_bands = ["red", "nir", "swir16", "blue"]

    tile_groups: dict[str, list[dict[str, Any]]] = {}
    for r in raw_scene_records:
        tile_groups.setdefault(r["mgrs_tile"], []).append(r)

    tile_stats = {}
    for t_id, t_records in tile_groups.items():
        valid_counts = [r["valid_stable_pixels"] for r in t_records if r["valid_stable_pixels"] > 0]
        tile_med_valid = float(np.median(valid_counts)) if valid_counts else 0.0

        band_tile_medians = {}
        for b_name in target_bands:
            col = f"stable_median_{b_name}"
            vals = [r[col] for r in t_records if r[col] is not None and np.isfinite(r[col])]
            band_tile_medians[b_name] = float(np.median(vals)) if vals else 0.0

        tile_stats[t_id] = {
            "tile_median_valid_count": tile_med_valid,
            "band_medians": band_tile_medians,
        }

    print("[+] Computed Per-Tile Medians:")
    for t_id, t_info in tile_stats.items():
        print(
            f"    - Tile {t_id:<6}: Median Valid Stable Pixels = {t_info['tile_median_valid_count']:,.0f}"
        )
        for b_name in target_bands:
            print(
                f"                   Median {b_name.upper():<7} = {t_info['band_medians'][b_name]:.4f}"
            )

    # Evaluate dropping criteria
    dropped_scenes = []
    final_records = []

    for r in raw_scene_records:
        t_id = r["mgrs_tile"]
        t_info = tile_stats[t_id]
        tile_med_valid = t_info["tile_median_valid_count"]

        r["tile_median_valid_count"] = int(tile_med_valid)
        for b_name in target_bands:
            r[f"tile_median_{b_name}"] = round(t_info["band_medians"][b_name], 4)

        drop_reasons = []

        # Drop Rule 1: < 50% of tile median valid count
        if tile_med_valid > 0 and r["valid_stable_pixels"] < (0.50 * tile_med_valid):
            pct_of_med = (r["valid_stable_pixels"] / tile_med_valid) * 100.0
            drop_reasons.append(
                f"Valid stable pixels ({r['valid_stable_pixels']:,}) < 50% of tile median ({tile_med_valid:,.0f}) [{pct_of_med:.1f}%]"
            )

        # Drop Rule 2: > 25% deviation from own tile's median in any band
        band_devs = []
        for b_name in target_bands:
            val = r[f"stable_median_{b_name}"]
            b_med = t_info["band_medians"][b_name]
            if val is not None and b_med > 0:
                diff_pct = (val - b_med) / b_med
                if abs(diff_pct) > 0.25:
                    drop_reasons.append(
                        f"{b_name} ({diff_pct*100:+.1f}%) > 25% dev from tile median"
                    )
                    band_devs.append(f"{b_name} ({diff_pct*100:+.1f}%)")
                elif abs(diff_pct) > 0.15:
                    band_devs.append(f"{b_name} ({diff_pct*100:+.1f}%) [flagged >15%]")

        is_dropped = len(drop_reasons) > 0
        r["status"] = "DROPPED" if is_dropped else "KEPT"
        r["drop_reasons"] = "; ".join(drop_reasons) if drop_reasons else "None"
        r["deviations_summary"] = "; ".join(band_devs) if band_devs else "Within ±15%"

        if is_dropped:
            dropped_scenes.append(r)
        final_records.append(r)

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
        "valid_stable_pixels",
        "tile_median_valid_count",
        "stable_median_red",
        "stable_median_nir",
        "stable_median_swir16",
        "stable_median_blue",
        "tile_median_red",
        "tile_median_nir",
        "tile_median_swir16",
        "tile_median_blue",
        "status",
        "drop_reasons",
        "deviations_summary",
        "scene_id",
    ]

    for out_csv in csv_paths:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in final_records:
                writer.writerow(r)
        print(f"[+] Saved diagnostics CSV: {out_csv.resolve()}", flush=True)

    # 5. Print Results Table
    print("\n" + "=" * 145, flush=True)
    print(
        f"{'Year':<5} {'Date':<10} {'Tile':<6} {'Baseline':<9} {'Cloud%':<7} {'ValidPx':<8} {'TileMedPx':<9} {'Red':<7} {'NIR':<7} {'SWIR':<7} {'Blue':<7} {'Status':<8} {'Drop Reasons / Deviations'}",
        flush=True,
    )
    print("-" * 145, flush=True)
    for r in final_records:
        r_str = f"{r['stable_median_red']:.4f}" if r["stable_median_red"] is not None else "N/A"
        n_str = f"{r['stable_median_nir']:.4f}" if r["stable_median_nir"] is not None else "N/A"
        s_str = (
            f"{r['stable_median_swir16']:.4f}" if r["stable_median_swir16"] is not None else "N/A"
        )
        b_str = f"{r['stable_median_blue']:.4f}" if r["stable_median_blue"] is not None else "N/A"
        stat_disp = f"[*] {r['status']}" if r["status"] == "DROPPED" else "KEPT"
        reason_disp = r["drop_reasons"] if r["status"] == "DROPPED" else r["deviations_summary"]
        print(
            f"{r['year']:<5} {r['date']:<10} {r['mgrs_tile']:<6} {r['processing_baseline']:<9} {r['cloud_cover_pct']:<7.2f} {r['valid_stable_pixels']:<8} {r['tile_median_valid_count']:<9} {r_str:<7} {n_str:<7} {s_str:<7} {b_str:<7} {stat_disp:<8} {reason_disp}",
            flush=True,
        )
    print("=" * 145, flush=True)

    print(f"\n[+] Summary for {city_name}:")
    print(f"    - Total strict in-window scenes evaluated : {len(final_records)}")
    print(
        f"    - Total scenes KEPT                       : {len(final_records) - len(dropped_scenes)}"
    )
    print(f"    - Total scenes DROPPED                    : {len(dropped_scenes)}")
    print("    - Yearly In-Window Breakdown (2018-2024):")
    for y, counts in yearly_counts.items():
        if y == 2022:
            print(f"      * {y}: 0 scenes [ARCHIVE-GAP YEAR]")
        else:
            print(f"      * {y}: {counts['scenes']} scenes across {counts['dates']} distinct dates")

    return final_records


def main():
    parser = argparse.ArgumentParser(
        description="UrbanPulse - Sentinel-2 Scene Diagnostics & Quality Screening (Strict Window)"
    )
    parser.add_argument(
        "--city", type=str, default="ahmedabad", help="Target city key (e.g. ahmedabad, pune)"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Path to data directory"
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to city YAML config")
    parser.add_argument(
        "--max-cloud", type=float, default=20.0, help="Max cloud cover percentage (default: 20.0)"
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Parallel worker threads (default: 16)"
    )

    args = parser.parse_args()
    process_scene_diagnostics_for_city(
        city=args.city,
        data_dir=args.data_dir,
        config_path=args.config,
        max_cloud_cover=args.max_cloud,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
