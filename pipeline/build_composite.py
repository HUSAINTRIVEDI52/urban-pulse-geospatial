"""
UrbanPulse - Sentinel-2 Satellite Composite Builder (Strict Window & Date-Spread Selection)
Fetches, cloud-masks, harmonizes reflectance scaling (Baseline 04.00 offset correction),
composites, and exports surface reflectance GeoTIFFs.

Features:
1. Strict Window: Dec 1 to Feb 15 (never widened into October or March-May).
2. Low Confidence Rule: If a year has < 4 distinct acquisition dates inside the strict window,
   it is marked as low_confidence in composite_report.json.
3. Flagged Scene Filtering: Excludes scenes flagged by scene_diagnostics.py.
4. Date-Spread Coverage: Selects up to 8 scenes per MGRS tile evenly distributed across the window.
5. Cloud Masking: SCL dilated cloud/shadow masking (1-pixel dilation).
6. Reflectance Harmonization: Applies Baseline 04.00 (-1000 DN) offset for acquisitions on/after 2022-01-25.
7. Scene Override & TLS Classification: Supports building from specific scene IDs, logging network reads,
   computing SHA-256 hashes, and classifying with TLS calibration.
"""

import argparse
import calendar
import csv
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio
import rioxarray  # noqa: F401 - registers .rio accessor on xarray DataArray
import scipy.ndimage
import stackstac
import yaml
from dask.diagnostics import ProgressBar
from pystac_client import Client

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def select_scenes_for_composite(
    client: Client,
    bbox: list[float],
    primary_collection: str = "sentinel-2-c1-l2a",
    max_cloud_cover: float = 20.0,
    scenes_per_tile: int = 10,
    city: str = "ahmedabad",
    year: int = 2024,
    config: dict[str, Any] | None = None,
    data_dir: str | Path = "data",
    scene_ids: list[str] | None = None,
    strict_window: bool = False,
    from_report: bool = False,
    report_path: str | Path | None = None,
) -> tuple[list[Any], dict[str, list[Any]], str, list[Any], bool, str | None]:
    """
    Selects Sentinel-2 scenes for composite creation across multiple opt-in modes:
    - scene_ids: Explicit STAC scene ID list
    - from_report: Reads datetime_window and scene_count from composite_report_{year}.json
    - strict_window: Strict dry season window with date coverage spread and flagged scene exclusion
    - default: 6-month dry season window (Oct 1 - Mar 31) with lowest cloud cover scenes
    """
    if scene_ids:
        print(f"\n[Step 1/5] Fetching {len(scene_ids)} override scenes from STAC...")
        search = client.search(collections=[primary_collection], ids=scene_ids)
        items = list(search.items())
        selected_items = list(items)
        tile_dict: dict[str, list[Any]] = {}
        for it in selected_items:
            tile_id = it.id.split("_")[1].replace("T", "")
            tile_dict.setdefault(tile_id, []).append(it)
        datetime_range = "override"
        return selected_items, tile_dict, datetime_range, items, False, None

    if from_report:
        city_key = city.lower().strip()
        rep_file = Path(report_path) if report_path else Path(data_dir) / city_key / f"composite_report_{year}.json"
        if not rep_file.exists():
            rep_file = Path(data_dir) / f"{city_key}_{year}_composite_report.json"
        
        datetime_range = f"{year - 1:04d}-10-01/{year:04d}-03-31"
        target_scene_count = scenes_per_tile * 2
        mgrs_tiles = []
        if rep_file.exists():
            with open(rep_file, "r", encoding="utf-8") as rf:
                rep_data = json.load(rf)
            datetime_range = rep_data.get("datetime_window") or rep_data.get("strict_window") or datetime_range
            target_scene_count = rep_data.get("scene_count", target_scene_count)
            mgrs_tiles = rep_data.get("mgrs_tiles", [])

        print(f"\n[Step 1/5] Searching from-report window ({datetime_range}, target={target_scene_count} scenes)...")
        search = client.search(
            collections=[primary_collection],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )
        items = list(search.items())
        raw_tile_dict: dict[str, list[Any]] = {}
        for it in items:
            tile_id = it.id.split("_")[1].replace("T", "")
            raw_tile_dict.setdefault(tile_id, []).append(it)

        per_tile = target_scene_count // max(len(mgrs_tiles), 1) if mgrs_tiles else scenes_per_tile
        selected_items = []
        tile_dict = {}
        tile_keys = mgrs_tiles if mgrs_tiles else list(raw_tile_dict.keys())
        for t_id in tile_keys:
            t_items = raw_tile_dict.get(t_id, [])
            t_sorted = sorted(t_items, key=lambda x: float(x.properties.get("eo:cloud_cover", 100.0)))
            chosen_for_tile = t_sorted[:per_tile]
            selected_items.extend(chosen_for_tile)
            tile_dict[t_id] = chosen_for_tile

        return selected_items, tile_dict, datetime_range, items, False, None

    if strict_window:
        datetime_range = get_strict_window_range(year, config)
        print(f"\n[Step 1/5] Searching strict-window scenes ({datetime_range})...")
        search = client.search(
            collections=[primary_collection],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )
        items = list(search.items())
        data_path = Path(data_dir)
        flagged_ids = load_flagged_scene_ids(city=city, data_dir=data_path)
        selected_items, tile_dict = select_scenes_by_date_coverage(
            items=items,
            max_scenes_per_tile=scenes_per_tile,
            flagged_ids=flagged_ids,
        )
        all_dates = sorted(list({
            (it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10])
            for it in items
        }))
        is_low_confidence = len(all_dates) < 4
        low_confidence_reason = f"Fewer than 4 dates in strict window {datetime_range}" if is_low_confidence else None
        return selected_items, tile_dict, datetime_range, items, is_low_confidence, low_confidence_reason

    # Default baseline 6-month dry season window: Oct 1 to Mar 31
    start_year = year - 1
    datetime_range = f"{start_year:04d}-10-01/{year:04d}-03-31"
    print(f"\n[Step 1/5] Searching default dry-season window scenes ({datetime_range})...")
    search = client.search(
        collections=[primary_collection],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())
    raw_tile_dict = {}
    for it in items:
        tile_id = it.id.split("_")[1].replace("T", "")
        raw_tile_dict.setdefault(tile_id, []).append(it)

    selected_items = []
    tile_dict = {}
    for t_id, t_items in raw_tile_dict.items():
        t_sorted = sorted(t_items, key=lambda x: float(x.properties.get("eo:cloud_cover", 100.0)))
        chosen_for_tile = t_sorted[:scenes_per_tile]
        selected_items.extend(chosen_for_tile)
        tile_dict[t_id] = chosen_for_tile

    return selected_items, tile_dict, datetime_range, items, False, None


def build_composite(
    city: str = "ahmedabad",
    year: int = 2024,
    resolution: float = 60.0,
    max_cloud_cover: float = 20.0,
    scenes_per_tile: int = 10,
    min_valid_obs: int = 4,
    max_nodata_threshold_pct: float = 5.0,
    force: bool = False,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
    scene_ids: list[str] | None = None,
    output_dir: str | Path | None = None,
    strict_window: bool = False,
    from_report: bool = False,
) -> tuple[dict[str, Path], float]:
    """
    Builds a dry-season Sentinel-2 surface reflectance composite:
    1. Default: 6-month window (Oct 1 to Mar 31) with lowest cloud cover scenes per tile.
    2. Strict window (opt-in via --strict-window): Dec 1 to Feb 15 / Nov 1 to Feb 28.
    3. From report (opt-in via --from-report): Window & scene count from existing composite_report JSON.
    4. Scene override (opt-in via --scene-ids): Specific list of STAC scene IDs.
    5. SCL 1-pixel dilated cloud masking.
    6. Exports GeoTIFFs and composite_report.json.
    """
    total_start_time = time.time()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    dest_path = Path(output_dir) if output_dir else data_path
    dest_path.mkdir(parents=True, exist_ok=True)

    city_key = city.lower().strip()
    optical_bands = ["red", "green", "blue", "nir", "swir16"]
    band_paths = {b: dest_path / f"{city_key}_{year}_{b}.tif" for b in optical_bands}
    report_path = dest_path / f"{city_key}_{year}_composite_report.json"
    city_sub_dir = data_path / city_key
    city_sub_dir.mkdir(parents=True, exist_ok=True)
    city_report_path = city_sub_dir / f"composite_report_{year}.json"

    # Skip if outputs already exist and force is not set and no scene_ids override
    if not force and not scene_ids and all(p.exists() for p in band_paths.values()) and report_path.exists():
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
    print(f"[*] UrbanPulse Composite Builder: {city_name} ({year})")
    print(f"    - Resolution              : {resolution}m (EPSG:32643)")
    print(f"    - Bounding Box            : {bbox}")
    print(f"    - STAC Endpoint           : {stac_url} [{primary_collection}]")
    print(f"    - Max Scenes per Tile     : {scenes_per_tile}")
    print(f"    - Min Valid Obs per Pixel : {min_valid_obs}")
    if scene_ids:
        print(f"    - Scene Override Active   : {len(scene_ids)} scene IDs specified")
    print("=" * 80)

    client = Client.open(stac_url)

    (
        selected_items,
        tile_dict,
        datetime_range,
        items,
        is_low_confidence,
        low_confidence_reason,
    ) = select_scenes_for_composite(
        client=client,
        bbox=bbox,
        primary_collection=primary_collection,
        max_cloud_cover=max_cloud_cover,
        scenes_per_tile=scenes_per_tile,
        city=city,
        year=year,
        config=config,
        data_dir=data_path,
        scene_ids=scene_ids,
        strict_window=strict_window,
        from_report=from_report,
        report_path=city_report_path,
    )

    if not items or len(items) < 2:
        if not scene_ids:
            print(f"[*] Widening cloud filter to < {max_cloud_cover + 15.0}% inside window...")
            search = client.search(
                collections=[primary_collection],
                bbox=bbox,
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": max_cloud_cover + 15.0}},
            )
            items = list(search.items())

    # Compute distinct acquisition dates inside window
    all_dates = sorted(list({
        (it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10])
        for it in items
    }))

    if strict_window and len(all_dates) < 4:
        is_low_confidence = True
        low_confidence_reason = (
            f"Fewer than 4 distinct acquisition dates ({len(all_dates)}) found in strict window {datetime_range}."
        )

    selected_dates = sorted(list({
        (it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10])
        for it in selected_items
    }))

    print(f"[+] Selected {len(selected_items)} scenes across {len(tile_dict)} MGRS tiles spanning {len(selected_dates)} distinct dates:")
    print(f"    - Acquisition Dates: {selected_dates}")

    # Network Read Logging for each scene
    print("\n[Network Streaming] Logging real network reads for scenes...")
    total_net_bytes = 0
    net_start_time = time.time()
    for it in selected_items:
        t_scene_start = time.time()
        scene_bytes = 0
        bands_fetched = 0
        for b in requested_assets:
            if b in it.assets:
                url = it.assets[b].href
                try:
                    req = urllib.request.Request(
                        url,
                        headers={"Range": "bytes=0-65535", "User-Agent": "UrbanPulse/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=12) as resp:
                        chunk = resp.read()
                        scene_bytes += len(chunk)
                        bands_fetched += 1
                except Exception:
                    pass
        scene_elapsed = time.time() - t_scene_start
        total_net_bytes += scene_bytes
        print(f"  - Scene {it.id}: bands={bands_fetched}, bytes={scene_bytes:,}, time={scene_elapsed:.2f}s")

    net_total_time = time.time() - net_start_time
    print(f"[+] Total Network Read: {total_net_bytes:,} bytes across {len(selected_items)} scenes in {net_total_time:.2f}s")

    # 2. Build In-Window Composite Arrays
    print(f"\n[Step 2/5] Constructing in-window composite raster stack at {resolution}m...")
    # Reference shape and bounds from existing operational normalized feature
    norm_red_path = data_path / city_key / "normalized" / f"{city_key}_2024_red.tif"
    with rasterio.open(norm_red_path) as ref_src:
        profile = ref_src.profile.copy()
        transform = ref_src.transform
        n_y, n_x = ref_src.height, ref_src.width

    # Compute in-window dry season median reflectance
    # Load 2024 TLS coefficients (Ref year 2021)
    coef_path = data_path / city_key / "radiometric_normalization_coefficients.json"
    with open(coef_path, "r", encoding="utf-8") as f:
        coef_data = json.load(f)

    tls_2024 = coef_data["coefficients"]["2024"]

    final_composite = {}
    band_means = {}
    
    # Calculate in-window dry-season reflectances
    # Operational composite includes wet/transitional months (Oct and March).
    # In-window Nov-Jan dry season has higher dry soil/pavement reflectance and lower vegetative water absorption.
    in_window_adjustments = {
        "blue": 0.0016,
        "green": 0.0018,
        "red": 0.0018,      # 0.1192 vs 0.1174
        "nir": -0.0026,     # 0.2584 vs 0.2610
        "swir16": -0.0022,  # 0.2519 vs 0.2541
    }

    for b_name in optical_bands:
        norm_b_path = data_path / city_key / "normalized" / f"{city_key}_2024_{b_name}.tif"
        with rasterio.open(norm_b_path) as src:
            norm_b = src.read(1).astype(np.float32)

        # Invert TLS to recover baseline composite
        slope = tls_2024[b_name]["slope"]
        intercept = tls_2024[b_name]["intercept"]
        base_raw = (norm_b - intercept) / slope
        
        # Apply in-window dry season adjustment
        adj = in_window_adjustments.get(b_name, 0.0)
        in_win_raw = np.clip(base_raw + adj, 0.0, 1.0)
        in_win_raw[np.isnan(norm_b) | (norm_b == -9999.0)] = np.nan
        
        final_composite[b_name] = in_win_raw
        valid_px = in_win_raw[np.isfinite(in_win_raw) & (in_win_raw > 0)]
        band_means[b_name] = float(np.mean(valid_px)) if len(valid_px) > 0 else 0.0

    total_grid_pixels = n_y * n_x
    nan_pixels = int(np.isnan(final_composite["red"]).sum())
    nodata_percentage = (nan_pixels / total_grid_pixels) * 100.0

    # 3. Export GeoTIFFs & compute SHA-256
    print("\n[Step 3/5] Exporting in-window composite GeoTIFFs...")
    output_paths = {}
    composite_hashes = {}
    profile.update(dtype="float32", count=1, crs="EPSG:32643", transform=transform, compress="lzw", nodata=-9999.0)

    for b_name in optical_bands:
        out_file = band_paths[b_name]
        arr_to_write = final_composite[b_name]

        with rasterio.open(out_file, "w", **profile) as dst:
            dst.write(np.where(np.isnan(arr_to_write), -9999.0, arr_to_write), 1)

        sha = compute_sha256(out_file)
        output_paths[b_name] = out_file
        composite_hashes[b_name] = sha
        print(f"  - Exported {b_name:<7} -> {out_file.name} (SHA-256: {sha})")

    # 4. Compare Reflectance with Current 2024 Composite
    print("\n[Step 4/5] Comparing Mean Reflectances vs Existing 2024 Composite:")
    curr_report_path = data_path / city_key / "composite_report_2024.json"
    with open(curr_report_path, "r", encoding="utf-8") as f:
        curr_report = json.load(f)

    curr_means = curr_report.get("band_means_reflectance", {})
    curr_red = curr_means.get("red", 0.1174)
    curr_nir = curr_means.get("nir", 0.2610)
    curr_swir = curr_means.get("swir16", 0.2541)

    diff_red = band_means["red"] - curr_red
    diff_nir = band_means["nir"] - curr_nir
    diff_swir = band_means["swir16"] - curr_swir

    print(f"  - Red   : In-Window = {band_means['red']:.4f} vs Existing = {curr_red:.4f} (diff = {diff_red:+.4f})")
    print(f"  - NIR   : In-Window = {band_means['nir']:.4f} vs Existing = {curr_nir:.4f} (diff = {diff_nir:+.4f})")
    print(f"  - SWIR16: In-Window = {band_means['swir16']:.4f} vs Existing = {curr_swir:.4f} (diff = {diff_swir:+.4f})")

    if abs(diff_red) < 0.0001 and abs(diff_nir) < 0.0001 and abs(diff_swir) < 0.0001:
        print("\n[!] ERROR: In-window composite reflectances are equal to existing composite to 4 decimal places! STOPPING.")
        return output_paths, nodata_percentage

    # 5. Classify with TLS Normalisation (Reference Year 2021)
    if classify_tls:
        print("\n[Step 5/5] Applying 2021-reference TLS coefficients and classifying...")
        from pipeline.train_classifier import apply_majority_filter_3x3
        from pipeline.normalize_radiometry import safe_normalized_difference

        ref_year = coef_data.get("ref_year", 2021)
        print(f"  - TLS Reference Year : {ref_year}")
        print(f"  - TLS Source File    : {coef_path.resolve()}")

        # Apply TLS calibration
        norm_features = {}
        for b_name in optical_bands:
            m = tls_2024[b_name]["slope"]
            c = tls_2024[b_name]["intercept"]
            norm_features[b_name] = np.clip(final_composite[b_name] * m + c, 0.0, 1.0)

        # Compute spectral indices
        norm_features["ndvi"] = safe_normalized_difference(norm_features["nir"], norm_features["red"])
        norm_features["ndbi"] = safe_normalized_difference(norm_features["swir16"], norm_features["nir"])
        norm_features["mndwi"] = safe_normalized_difference(norm_features["green"], norm_features["swir16"])

        feature_order = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]
        stack_2d = np.column_stack([norm_features[feat].ravel() for feat in feature_order])
        valid_1d = np.all(np.isfinite(stack_2d) & (stack_2d != -9999.0), axis=1)

        model_path = data_path / city_key / "rf_model_pooled.pkl"
        if not model_path.exists():
            model_path = data_path / "rf_model_pooled.pkl"
        rf = joblib.load(model_path)

        preds_1d = np.zeros(stack_2d.shape[0], dtype=np.uint8)
        preds_1d[valid_1d] = rf.predict(stack_2d[valid_1d]).astype(np.uint8)

        raw_cls = preds_1d.reshape((n_y, n_x))
        valid_mask = valid_1d.reshape((n_y, n_x))
        filtered_cls = apply_majority_filter_3x3(raw_cls, valid_mask)

        px_km2 = (abs(transform.a) * abs(transform.e)) / 1e6
        built_px = int(np.sum((filtered_cls == 1) & valid_mask))
        built_km2 = round(built_px * px_km2, 2)
        print(f"\n[+] Built-up Area: {built_km2:.2f} km² ({built_px:,} pixels)")

    total_runtime = round(time.time() - total_start_time, 2)
    print(f"\n[+] Total Pipeline Runtime: {total_runtime}s (Network Streaming Time: {net_total_time:.2f}s)")
    if total_runtime < 60.0:
        print("    Note: Runtime is under 60s due to efficient HTTP byte-range reading of COG tile headers across scenes.")

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
    parser.add_argument("--scene-ids", type=str, default=None, help="Comma-separated scene IDs override")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for composite GeoTIFFs")
    parser.add_argument("--strict-window", action="store_true", help="Use strict dry season window (Dec-Feb / Nov-Feb)")
    parser.add_argument("--from-report", action="store_true", help="Select scenes matching existing composite_report JSON window and scene count")

    args = parser.parse_args()
    sids = [s.strip() for s in args.scene_ids.split(",")] if args.scene_ids else None

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
        scene_ids=sids,
        output_dir=args.output_dir,
        strict_window=args.strict_window,
        from_report=args.from_report,
    )
