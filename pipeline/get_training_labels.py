"""
UrbanPulse - ESA WorldCover Training Label Generator
Downloads and reprojects ESA WorldCover 2021 land cover raster for any configured city AOI,
remapping standard WorldCover classes into the 5-class project schema.
"""

import argparse
import io
import sys
from pathlib import Path
from typing import Any
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import requests
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WORLDCOVER_V200_S3_BASE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
)

# WorldCover -> Project Schema
# 1: Built-up (50), 2: Vegetation (10, 20, 30, 90, 95), 3: Water (80), 4: Agriculture (40), 5: Open land (60, 70)
WORLDCOVER_REMAP = {
    50: 1,  # Built-up
    10: 2,  # Tree cover
    20: 2,  # Shrubland
    30: 2,  # Grassland
    90: 2,  # Herbaceous wetland
    95: 2,  # Mangroves
    80: 3,  # Permanent water bodies
    40: 4,  # Cropland
    60: 5,  # Bare / sparse vegetation
    70: 5,  # Snow and ice
}


def load_config(
    city: str = "ahmedabad", config_path: str | Path | None = None
) -> dict[str, Any]:
    """Loads city YAML configuration."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_worldcover_tiles_for_bbox(bbox: list[float]) -> list[str]:
    """
    Determines required 3x3 degree ESA WorldCover tile identifiers for a given bbox [minx, miny, maxx, maxy].
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    lat_start = int(np.floor(min_lat / 3.0) * 3)
    lat_end = int(np.floor(max_lat / 3.0) * 3)

    lon_start = int(np.floor(min_lon / 3.0) * 3)
    lon_end = int(np.floor(max_lon / 3.0) * 3)

    tiles = []
    for lat in range(lat_start, lat_end + 1, 3):
        for lon in range(lon_start, lon_end + 1, 3):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            tile_id = f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"
            tiles.append(tile_id)

    return tiles


def extract_worldcover_labels(
    city: str = "ahmedabad",
    config_path: str | Path | None = None,
    target_crs: str = "EPSG:32643",
    resolution: float = 20.0,
    reference_raster_path: str | Path | None = None,
    output_labels_path: str | Path | None = None,
) -> Path:
    """
    Fetches, warps, and remaps ESA WorldCover 2021 into project training labels for a city.
    """
    city_key = city.lower()
    config = load_config(city=city_key, config_path=config_path)
    city_name = config.get("city", {}).get("name", city.capitalize())
    bbox = config["spatial"]["bbox"]

    if output_labels_path is None:
        out_path = Path(f"data/{city_key}_worldcover_labels.tif")
    else:
        out_path = Path(output_labels_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"[*] Found existing training labels for {city_name}: {out_path}")
        return out_path

    print("=" * 75)
    print(f"[*] Extracting ESA WorldCover 2021 Training Labels for {city_name}")
    print(f"    - Target CRS       : {target_crs}")
    print(f"    - Resolution       : {resolution}m")
    print(f"    - Bounding Box     : {bbox}")
    print("=" * 75)

    tiles = get_worldcover_tiles_for_bbox(bbox)
    print(f"[+] Identified {len(tiles)} required WorldCover tile(s): {tiles}")

    tile_datasets = []
    for tile_id in tiles:
        tile_url = f"{WORLDCOVER_V200_S3_BASE}/ESA_WorldCover_10m_2021_v200_{tile_id}_Map.tif"
        print(f"[+] Downloading tile {tile_id} from ESA AWS S3...")
        try:
            resp = requests.get(tile_url, timeout=60)
            resp.raise_for_status()
            mem_file = io.BytesIO(resp.content)
            src = rasterio.open(mem_file)
            tile_datasets.append(src)
        except Exception as e:
            print(f"[!] Warning: Failed to download tile {tile_id} ({e}). Trying GDAL vsicurl...")
            src = rasterio.open(f"/vsicurl/{tile_url}")
            tile_datasets.append(src)

    if not tile_datasets:
        raise RuntimeError("No WorldCover tiles could be loaded.")

    if reference_raster_path and Path(reference_raster_path).exists():
        with rasterio.open(reference_raster_path) as ref:
            dst_crs = ref.crs
            dst_transform = ref.transform
            dst_width = ref.width
            dst_height = ref.height
    else:
        min_lon, min_lat, max_lon, max_lat = bbox
        dst_crs = target_crs
        dst_transform, dst_width, dst_height = calculate_default_transform(
            "EPSG:4326", dst_crs, 1000, 1000, left=min_lon, bottom=min_lat, right=max_lon, top=max_lat
        )

    destination_arr = np.zeros((dst_height, dst_width), dtype=np.uint8)

    for src in tile_datasets:
        temp_arr = np.zeros((dst_height, dst_width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=temp_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        mask = (temp_arr > 0) & (destination_arr == 0)
        destination_arr[mask] = temp_arr[mask]

    # Remap into 5 project classes
    remapped_arr = np.zeros_like(destination_arr, dtype=np.uint8)
    for raw_val, proj_val in WORLDCOVER_REMAP.items():
        remapped_arr[destination_arr == raw_val] = proj_val

    # Write output
    meta = {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": 0,
        "width": dst_width,
        "height": dst_height,
        "count": 1,
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(remapped_arr, 1)

    print(f"[+] Successfully saved {city_name} training labels: {out_path.resolve()}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract WorldCover training labels.")
    parser.add_argument("--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--crs", type=str, default="EPSG:32643", help="Target CRS (default: EPSG:32643)")
    parser.add_argument("--out", type=str, default=None, help="Output GeoTIFF path")

    args = parser.parse_args()
    extract_worldcover_labels(
        city=args.city,
        config_path=args.config,
        target_crs=args.crs,
        output_labels_path=args.out,
    )


if __name__ == "__main__":
    main()
