"""
UrbanPulse - Unified Strict-Window Scene Selection, STAC Querying & Deduplication
Shared across build_composite.py, scene_diagnostics.py, and annual pipeline runners.
"""

import calendar
from pathlib import Path
from typing import Any

import numpy as np
import yaml
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


def get_strict_window_dates(year: int, config: dict[str, Any]) -> tuple[str, str, str]:
    """
    Constructs strict dry season date range from YAML config (Nov 1 to Feb 28/29).
    Never widens into October or March.

    Returns:
        (start_date, end_date, iso_datetime_range)
        e.g., for 2020: ('2019-11-01', '2020-02-29', '2019-11-01/2020-02-29')
    """
    temporal_cfg = config.get("temporal", {})
    window_cfg = temporal_cfg.get("strict_window") or temporal_cfg.get("dry_season", {})

    start_month = window_cfg.get("start_month", 11)
    end_month = window_cfg.get("end_month", 2)
    start_day_str = window_cfg.get("start_day", "11-01")
    end_day_str = window_cfg.get("end_day", "02-28")

    # Parse start month and day
    if "-" in str(start_day_str):
        s_parts = str(start_day_str).split("-")
        start_month = int(s_parts[0])
        s_day = int(s_parts[1])
    else:
        s_day = int(start_day_str)

    # Parse end month and day (account for leap years when end_month is February)
    if "-" in str(end_day_str):
        e_parts = str(end_day_str).split("-")
        end_month = int(e_parts[0])
        e_day = int(e_parts[1])
    else:
        e_day = int(end_day_str)

    if end_month == 2:
        _, max_feb_days = calendar.monthrange(year, 2)
        e_day = min(e_day, max_feb_days) if e_day < 28 else max_feb_days

    start_year = year - 1 if start_month > end_month else year
    start_date = f"{start_year:04d}-{start_month:02d}-{s_day:02d}"
    end_date = f"{year:04d}-{end_month:02d}-{e_day:02d}"
    iso_range = f"{start_date}/{end_date}"

    return start_date, end_date, iso_range


def extract_mgrs_tile(item: Any) -> str:
    """Extracts MGRS tile ID (e.g. '42QZL') from a STAC item."""
    z = str(item.properties.get("mgrs:utm_zone", "")).strip()
    b = str(item.properties.get("mgrs:latitude_band", "")).strip()
    g = str(item.properties.get("mgrs:grid_square", "")).strip()
    if z and b and g:
        return f"{z}{b}{g}"

    parts = item.id.split("_")
    if len(parts) >= 2 and len(parts[1]) == 5:
        return parts[1]

    return "UNKNOWN"


def query_strict_window_scenes(
    city_key: str,
    year: int,
    config: dict[str, Any],
    max_cloud_cover: float = 20.0,
    collection: str = "sentinel-2-c1-l2a",
    client: Client | None = None,
) -> list[Any]:
    """
    Queries Sentinel-2 scenes for the strict Nov 1 - Feb 28/29 window.
    2022 is marked as an ARCHIVE-GAP YEAR (returns empty list).
    """
    if year == 2022:
        return []

    stac_url = config.get("stac", {}).get(
        "earth_search_url", "https://earth-search.aws.element84.com/v1"
    )
    bbox = config["spatial"]["bbox"]
    _, _, datetime_range = get_strict_window_dates(year, config)

    if client is None:
        client = Client.open(stac_url)

    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())

    # Fallback to slightly higher cloud cover threshold if zero scenes found
    if not items:
        search = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover + 15.0}},
        )
        items = list(search.items())

    return items


def deduplicate_tile_date_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Deduplicates scenes having the same (mgrs_tile, date) pair.
    Keeps the scene with the highest valid_stable_pixels (or lowest cloud cover as tie-breaker).

    Returns:
        (deduplicated_records, dropped_duplicate_records)
    """
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for r in records:
        key = (r["year"], r["mgrs_tile"], r["date"])
        groups.setdefault(key, []).append(r)

    deduped = []
    dropped_dups = []

    for group in groups.values():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Sort by valid_stable_pixels descending, then cloud_cover ascending
            sorted_group = sorted(
                group,
                key=lambda x: (
                    x.get("valid_stable_pixels", 0),
                    -float(x.get("cloud_cover_pct", 100.0)),
                ),
                reverse=True,
            )
            winner = sorted_group[0]
            deduped.append(winner)
            for loser in sorted_group[1:]:
                loser_copy = dict(loser)
                loser_copy["status"] = "DROPPED_DUPLICATE"
                loser_copy["drop_reasons"] = (
                    f"Duplicate (tile {winner['mgrs_tile']}, date {winner['date']}): "
                    f"Kept {winner['scene_id']} ({winner.get('valid_stable_pixels', 0):,} valid px) "
                    f"over {loser['scene_id']} ({loser.get('valid_stable_pixels', 0):,} valid px)"
                )
                dropped_dups.append(loser_copy)

    # Sort deterministically
    deduped.sort(key=lambda r: (r["year"], r["date"], r["mgrs_tile"], r["scene_id"]))
    dropped_dups.sort(key=lambda r: (r["year"], r["date"], r["mgrs_tile"], r["scene_id"]))

    return deduped, dropped_dups


def apply_tile_quality_screening(
    records: list[dict[str, Any]],
    target_bands: list[str] | None = None,
    rule2_bands: list[str] | None = None,
    dev_threshold: float = 0.25,
    valid_ratio_threshold: float = 0.50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Computes per-tile medians and screens scenes using Rule 1 (valid pixel count) and Rule 2 (reflectance deviation).

    Args:
        records: List of scene records (pre-deduplicated).
        target_bands: All bands to calculate medians for (default: red, nir, swir16, blue).
        rule2_bands: Bands used to evaluate Rule 2 (e.g. ['red', 'nir', 'swir16', 'blue'] or ['red', 'nir', 'swir16']).
        dev_threshold: Maximum allowable fractional deviation (default: 0.25 -> 25%).
        valid_ratio_threshold: Minimum allowable ratio of tile median valid pixels (default: 0.50 -> 50%).

    Returns:
        (kept_records, dropped_records, tile_stats)
    """
    if target_bands is None:
        target_bands = ["red", "nir", "swir16", "blue"]
    if rule2_bands is None:
        rule2_bands = ["red", "nir", "swir16", "blue"]

    tile_groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        tile_groups.setdefault(r["mgrs_tile"], []).append(r)

    tile_stats = {}
    for t_id, t_records in tile_groups.items():
        valid_counts = [
            r["valid_stable_pixels"] for r in t_records if r.get("valid_stable_pixels", 0) > 0
        ]
        tile_med_valid = float(np.median(valid_counts)) if valid_counts else 0.0

        band_tile_medians = {}
        for b_name in target_bands:
            col = f"stable_median_{b_name}"
            vals = [r[col] for r in t_records if r.get(col) is not None and np.isfinite(r[col])]
            band_tile_medians[b_name] = float(np.median(vals)) if vals else 0.0

        tile_stats[t_id] = {
            "tile_median_valid_count": tile_med_valid,
            "band_medians": band_tile_medians,
        }

    kept_records = []
    dropped_records = []

    for r in records:
        t_id = r["mgrs_tile"]
        t_info = tile_stats[t_id]
        tile_med_valid = t_info["tile_median_valid_count"]

        r_copy = dict(r)
        r_copy["tile_median_valid_count"] = int(tile_med_valid)
        for b_name in target_bands:
            r_copy[f"tile_median_{b_name}"] = round(t_info["band_medians"][b_name], 4)

        drop_reasons = []

        # Rule 1: < 50% of tile median valid count
        if tile_med_valid > 0 and r_copy.get("valid_stable_pixels", 0) < (
            valid_ratio_threshold * tile_med_valid
        ):
            pct_of_med = (r_copy["valid_stable_pixels"] / tile_med_valid) * 100.0
            drop_reasons.append(
                f"Valid stable pixels ({r_copy['valid_stable_pixels']:,}) < {valid_ratio_threshold*100:.0f}% of tile median ({tile_med_valid:,.0f}) [{pct_of_med:.1f}%]"
            )

        # Rule 2: > 25% deviation from own tile's median in rule2_bands
        band_devs = []
        for b_name in target_bands:
            val = r_copy.get(f"stable_median_{b_name}")
            b_med = t_info["band_medians"].get(b_name, 0.0)
            if val is not None and b_med > 0:
                diff_pct = (val - b_med) / b_med
                if b_name in rule2_bands and abs(diff_pct) > dev_threshold:
                    drop_reasons.append(
                        f"{b_name} ({diff_pct*100:+.1f}%) > {dev_threshold*100:.0f}% dev from tile median"
                    )
                    band_devs.append(f"{b_name} ({diff_pct*100:+.1f}%)")
                elif abs(diff_pct) > 0.15:
                    band_devs.append(f"{b_name} ({diff_pct*100:+.1f}%) [flagged >15%]")

        is_dropped = len(drop_reasons) > 0
        r_copy["status"] = "DROPPED" if is_dropped else "KEPT"
        r_copy["drop_reasons"] = "; ".join(drop_reasons) if drop_reasons else "None"
        r_copy["deviations_summary"] = "; ".join(band_devs) if band_devs else "Within ±15%"

        if is_dropped:
            dropped_records.append(r_copy)
        else:
            kept_records.append(r_copy)

    return kept_records, dropped_records, tile_stats
