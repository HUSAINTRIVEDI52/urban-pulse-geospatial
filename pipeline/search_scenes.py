"""
UrbanPulse - Sentinel-2 Scene Search Module
Queries Earth Search STAC API for Sentinel-2 L2A scenes over the city AOI
filtered by dry-season temporal window and cloud cover constraints.
"""

import argparse
import calendar
from pathlib import Path
from typing import Any
import yaml
from pystac_client import Client
import pystac


def load_config(config_path: str | Path = "configs/ahmedabad.yaml") -> dict[str, Any]:
    """Loads and validates the city YAML configuration."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file.resolve()}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "spatial" not in config or "bbox" not in config["spatial"]:
        raise ValueError("Missing 'spatial.bbox' in configuration file.")

    return config


def get_dry_season_datetime(year: int, config: dict[str, Any]) -> str:
    """
    Constructs the STAC ISO 8601 datetime range string for the dry season of a given analysis year.
    Dry season Nov-Feb for year Y spans from Nov 1 of (Y-1) to the end of Feb of Y.
    """
    temporal_cfg = config.get("temporal", {})
    dry_season_cfg = temporal_cfg.get("dry_season", {})

    start_month = dry_season_cfg.get("start_month", 11)
    end_month = dry_season_cfg.get("end_month", 2)

    # Determine start date: Nov 1 of (year - 1)
    start_year = year - 1 if start_month > end_month else year
    start_date = f"{start_year:04d}-{start_month:02d}-01"

    # Determine end date: last day of end_month in `year` (handles leap years)
    _, last_day = calendar.monthrange(year, end_month)
    end_date = f"{year:04d}-{end_month:02d}-{last_day:02d}"

    return f"{start_date}/{end_date}"


def search_sentinel_scenes(
    year: int = 2024,
    config_path: str | Path = "configs/ahmedabad.yaml",
    max_cloud_cover: float = 30.0,
    custom_datetime: str | None = None,
) -> list[pystac.Item]:
    """
    Searches Earth Search STAC API for Sentinel-2 L2A scenes matching criteria.

    Args:
        year: Target analysis year (default 2024).
        config_path: Path to city config YAML.
        max_cloud_cover: Maximum allowable cloud cover percentage (default 30.0%).
        custom_datetime: Optional explicit STAC datetime string (e.g. '2024-01-01/2024-02-29').

    Returns:
        List of matching PySTAC Items.
    """
    config = load_config(config_path)
    city_name = config.get("city", {}).get("name", "Target City")
    bbox = config["spatial"]["bbox"]

    stac_cfg = config.get("stac", {})
    earth_search_url = stac_cfg.get(
        "earth_search_url", "https://earth-search.aws.element84.com/v1"
    )
    collection = stac_cfg.get("collections", {}).get("sentinel_2", "sentinel-2-l2a")

    datetime_range = custom_datetime or get_dry_season_datetime(year, config)

    print("=" * 70)
    print(f"[*] Searching STAC Catalog for {city_name} (Year: {year})")
    print(f"    - Endpoint       : {earth_search_url}")
    print(f"    - Collection     : {collection}")
    print(f"    - Bounding Box   : {bbox}")
    print(f"    - Datetime Range : {datetime_range}")
    print(f"    - Max Cloud Cover: < {max_cloud_cover}%")
    print("=" * 70)

    # Open STAC Client
    client = Client.open(earth_search_url)

    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )

    items = list(search.items())

    if not items:
        error_msg = (
            f"\n[ERROR] Zero Sentinel-2 scenes found for {city_name}!\n"
            f"Criteria used:\n"
            f"  - Datetime window : '{datetime_range}'\n"
            f"  - Max Cloud Cover : < {max_cloud_cover}%\n"
            f"  - Bounding Box    : {bbox}\n\n"
            f"[SUGGESTION] To find available scenes, please consider:\n"
            f"  1. Widening the date range (e.g. adding adjacent months/weeks with --datetime).\n"
            f"  2. Increasing the cloud cover tolerance (e.g. --max-cloud 50).\n"
            f"  3. Checking if data is available for the requested year in STAC collection '{collection}'."
        )
        raise RuntimeError(error_msg)

    # Sort items chronologically
    items.sort(key=lambda item: item.datetime or item.properties.get("datetime", ""))

    print(f"\n[+] Found {len(items)} Sentinel-2 scenes matching criteria:\n")
    print(
        f"{'#':<3} | {'Scene ID':<30} | {'Acquisition Date (UTC)':<22} | {'Cloud Cover (%)':<15}"
    )
    print("-" * 78)

    for idx, item in enumerate(items, 1):
        dt_str = item.datetime.strftime("%Y-%m-%d %H:%M:%S") if item.datetime else str(item.properties.get("datetime"))
        cloud_pct = item.properties.get("eo:cloud_cover", 0.0)
        print(f"{idx:<3} | {item.id:<30} | {dt_str:<22} | {cloud_pct:<15.2f}")

    print("-" * 78)
    print(f"Total Scenes: {len(items)}\n")

    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Sentinel-2 scenes via STAC API.")
    parser.add_argument("--year", type=int, default=2024, help="Analysis year (default: 2024)")
    parser.add_argument("--config", type=str, default="configs/ahmedabad.yaml", help="Path to config YAML")
    parser.add_argument("--max-cloud", type=float, default=30.0, help="Max cloud cover percentage (default: 30.0)")
    parser.add_argument("--datetime", type=str, default=None, help="Custom ISO-8601 datetime range (e.g. 2024-01-01/2024-02-29)")

    args = parser.parse_args()

    search_sentinel_scenes(
        year=args.year,
        config_path=args.config,
        max_cloud_cover=args.max_cloud,
        custom_datetime=args.datetime,
    )
