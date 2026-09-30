"""
UrbanPulse - Sentinel-2 Scene Search Module
Queries Earth Search STAC API for Sentinel-2 L2A scenes over the city AOI,
groups them by intersecting MGRS tiles, and selects the lowest-cloud dry season scenes.
"""

import argparse
import calendar
from pathlib import Path
from typing import Any

import pystac
import yaml
from pystac_client import Client


def load_config(config_path: str | Path = "configs/ahmedabad.yaml") -> dict[str, Any]:
    """Loads and validates the city YAML configuration."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file.resolve()}")

    with open(config_file, encoding="utf-8") as f:
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

    start_year = year - 1 if start_month > end_month else year
    start_date = f"{start_year:04d}-{start_month:02d}-01"

    _, last_day = calendar.monthrange(year, end_month)
    end_date = f"{year:04d}-{end_month:02d}-{last_day:02d}"

    return f"{start_date}/{end_date}"


def search_sentinel_scenes(
    year: int = 2024,
    config_path: str | Path = "configs/ahmedabad.yaml",
    max_cloud_cover: float = 10.0,
    scenes_per_tile: int = 4,
    custom_datetime: str | None = None,
) -> list[pystac.Item]:
    """
    Searches Earth Search STAC API for Sentinel-2 L2A scenes grouped by MGRS tile.
    """
    config = load_config(config_path)
    city_name = config.get("city", {}).get("name", "Ahmedabad")
    bbox = config["spatial"]["bbox"]

    stac_cfg = config.get("stac", {})
    earth_search_url = stac_cfg.get("earth_search_url", "https://earth-search.aws.element84.com/v1")
    collection = stac_cfg.get("collections", {}).get("sentinel_2", "sentinel-2-c1-l2a")

    datetime_range = custom_datetime or get_dry_season_datetime(year, config)

    print("=" * 78)
    print(f"[*] Searching STAC Catalog for {city_name} (Year: {year})")
    print(f"    - Endpoint          : {earth_search_url}")
    print(f"    - Collection        : {collection}")
    print(f"    - Bounding Box      : {bbox}")
    print(f"    - Datetime Window   : {datetime_range}")
    print(f"    - Max Cloud Cover   : < {max_cloud_cover}%")
    print(f"    - Top Scenes / Tile : {scenes_per_tile}")
    print("=" * 78)

    client = Client.open(earth_search_url)

    search = client.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )

    items = list(search.items())

    if not items:
        # Fallback to broader collection or cloud threshold if needed
        search = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover + 15}},
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
            f"  2. Increasing the cloud cover tolerance (e.g. --max-cloud 30).\n"
            f"  3. Checking if data is available for the requested year in STAC collection '{collection}'."
        )
        raise RuntimeError(error_msg)

    # Group items by MGRS tile
    tile_dict: dict[str, list[pystac.Item]] = {}
    for item in items:
        z = str(item.properties.get("mgrs:utm_zone", ""))
        b = str(item.properties.get("mgrs:latitude_band", ""))
        g = str(item.properties.get("mgrs:grid_square", ""))
        tile_id = f"{z}{b}{g}" if (z and b and g) else item.id.split("_")[1].replace("T", "")
        tile_dict.setdefault(tile_id, []).append(item)

    print(f"\n[+] Identified {len(tile_dict)} intersecting MGRS tiles for AOI:\n")

    selected_items: list[pystac.Item] = []
    print(
        f"{'Tile':<8} | {'Scene ID':<34} | {'Acquisition Date (UTC)':<22} | {'Cloud Cover (%)':<15}"
    )
    print("-" * 85)

    for tile_id, t_items in tile_dict.items():
        # Sort by cloud cover ascending
        t_items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
        top_tile_scenes = t_items[:scenes_per_tile]
        selected_items.extend(top_tile_scenes)

        for it in top_tile_scenes:
            dt_str = (
                it.datetime.strftime("%Y-%m-%d %H:%M:%S")
                if it.datetime
                else str(it.properties.get("datetime"))
            )
            cloud_pct = it.properties.get("eo:cloud_cover", 0.0)
            print(f"{tile_id:<8} | {it.id:<34} | {dt_str:<22} | {cloud_pct:<15.2f}")

    print("-" * 85)
    print(
        f"Total Selected Scenes: {len(selected_items)} ({scenes_per_tile} per tile across {len(tile_dict)} tiles)\n"
    )

    return selected_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search Sentinel-2 scenes grouped by MGRS tile.")
    parser.add_argument("--year", type=int, default=2024, help="Analysis year (default: 2024)")
    parser.add_argument(
        "--config", type=str, default="configs/ahmedabad.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--max-cloud", type=float, default=10.0, help="Max cloud cover percentage (default: 10.0)"
    )
    parser.add_argument(
        "--scenes-per-tile", type=int, default=4, help="Scenes per tile (default: 4)"
    )
    parser.add_argument("--datetime", type=str, default=None, help="Custom ISO-8601 datetime range")

    args = parser.parse_args()
    search_sentinel_scenes(
        year=args.year,
        config_path=args.config,
        max_cloud_cover=args.max_cloud,
        scenes_per_tile=args.scenes_per_tile,
        custom_datetime=args.datetime,
    )
