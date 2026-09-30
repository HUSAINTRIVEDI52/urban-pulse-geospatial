"""
UrbanPulse - Training Labels Extraction Module
Downloads and reprojects ESA WorldCover 2021 land cover raster for the Ahmedabad AOI,
resamples it to 20m in EPSG:32643 (UTM Zone 43N), remaps raw classes into 5 standardized
project categories (Built-up, Vegetation, Water, Agriculture, Open land), and exports
data/ahmedabad_worldcover_labels.tif.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ESA WorldCover 2021 -> UrbanPulse 5-Class Legend Mapping
# 1: Built-up, 2: Vegetation, 3: Water, 4: Agriculture, 5: Open land
CLASS_MAPPING = {
    50: (1, "Built-up"),
    10: (2, "Vegetation (Tree cover)"),
    20: (2, "Vegetation (Shrubland)"),
    30: (2, "Vegetation (Grassland)"),
    95: (2, "Vegetation (Mangroves)"),
    80: (3, "Water (Permanent water)"),
    90: (3, "Water (Herbaceous wetland)"),
    40: (4, "Agriculture (Cropland)"),
    60: (5, "Open land (Bare / sparse vegetation)"),
    100: (5, "Open land (Moss and lichen)"),
}

PROJECT_CLASS_NAMES = {
    1: "Built-up",
    2: "Vegetation",
    3: "Water",
    4: "Agriculture",
    5: "Open land",
}


def load_config(config_path: str | Path = "configs/ahmedabad.yaml") -> dict[str, Any]:
    """Loads city YAML configuration."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_worldcover_tiles_for_bbox(bbox: list[float]) -> list[str]:
    """
    Determines required 3x3 degree ESA WorldCover tile identifiers for a given bbox [minx, miny, maxx, maxy].
    WorldCover tiles are named by bottom-left coordinate in multiples of 3 degrees (e.g. N21E069).
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
    config_path: str | Path = "configs/ahmedabad.yaml",
    target_crs: str = "EPSG:32643",
    resolution: float = 20.0,
    reference_raster_path: str | Path | None = None,
    output_labels_path: str | Path = "data/ahmedabad_worldcover_labels.tif",
) -> Path:
    """
    Fetches, warps, and remaps ESA WorldCover 2021 into 20m project training labels.
    """
    config = load_config(config_path)
    bbox = config["spatial"]["bbox"]
    out_path = Path(output_labels_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("[*] Extracting ESA WorldCover 2021 Training Labels for Ahmedabad")
    print(f"    - Target CRS       : {target_crs}")
    print(f"    - Resolution       : {resolution}m")
    print(f"    - Bounding Box     : {bbox}")
    print(f"    - Output File      : {out_path.resolve()}")
    print("=" * 75)

    # 1. Determine Target Raster Grid (20m, EPSG:32643)
    if reference_raster_path and Path(reference_raster_path).exists():
        ref_path = Path(reference_raster_path)
        print(f"[*] Matching grid from reference raster: {ref_path.name}")
        with rasterio.open(ref_path) as ref_src:
            dest_crs = ref_src.crs
            dest_transform = ref_src.transform
            dest_shape = ref_src.shape
    else:
        print(f"[*] Calculating {resolution}m target grid in {target_crs} from AOI bounding box...")
        gdf_bbox = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326").to_crs(target_crs)
        minx, miny, maxx, maxy = gdf_bbox.total_bounds

        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))
        dest_transform = rasterio.transform.from_origin(minx, maxy, resolution, resolution)
        dest_shape = (height, width)
        dest_crs = rasterio.crs.CRS.from_string(target_crs)

    print(
        f"[*] Target Grid Shape: {dest_shape[0]} rows x {dest_shape[1]} cols ({dest_shape[0]*dest_shape[1]:,} pixels)"
    )

    # 2. Determine and Fetch WorldCover Tile COGs from AWS Open Data
    tile_ids = get_worldcover_tiles_for_bbox(bbox)
    print(f"\n[Step 1/3] Identified required ESA WorldCover 3x3 deg tiles: {tile_ids}")

    warped_raw = np.zeros(dest_shape, dtype=np.uint8)

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for tile in tile_ids:
            tile_url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
            print(f"    - Streaming & warping tile {tile} at {resolution}m...", end="", flush=True)

            try:
                with rasterio.open(tile_url) as tile_src:
                    temp_dest = np.zeros(dest_shape, dtype=np.uint8)
                    reproject(
                        source=rasterio.band(tile_src, 1),
                        destination=temp_dest,
                        src_transform=tile_src.transform,
                        src_crs=tile_src.crs,
                        dst_transform=dest_transform,
                        dst_crs=dest_crs,
                        resampling=Resampling.nearest,
                    )
                    # Merge tiles into warped raster
                    warped_raw = np.where(temp_dest > 0, temp_dest, warped_raw)
                    print(" [Done]")
            except Exception as e:
                print(f" [Error loading {tile}: {e}]")

    # 3. Remap into 5 Project Classes
    print("\n[Step 2/3] Remapping ESA WorldCover classes into 5 target project classes...")
    remapped = np.zeros(dest_shape, dtype=np.uint8)

    for src_class, (tgt_class, desc) in CLASS_MAPPING.items():
        mask = warped_raw == src_class
        count = int(np.sum(mask))
        if count > 0:
            remapped[mask] = tgt_class
            print(
                f"    - Class {src_class:<3} -> Project Class {tgt_class} ({desc:<30}): {count:>10,} pixels"
            )

    # 4. Save GeoTIFF
    print(f"\n[Step 3/3] Saving training labels GeoTIFF to: {out_path.name}...")
    profile = {
        "driver": "GTiff",
        "height": dest_shape[0],
        "width": dest_shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": dest_crs,
        "transform": dest_transform,
        "nodata": 0,
        "compress": "lzw",
        "tiled": True,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(remapped, 1)

    # 5. Compute Class Distribution Statistics
    pixel_area_km2 = (resolution * resolution) / 1e6
    valid_mask = remapped > 0
    total_valid_pixels = int(np.sum(valid_mask))

    print("\n" + "=" * 75)
    print("[*] Training Labels Class Distribution Summary (20m, EPSG:32643)")
    print("=" * 75)
    print(
        f"{'Class ID':<9} | {'Class Name':<15} | {'Pixel Count':<14} | {'Area (km^2)':<12} | {'Percentage'}"
    )
    print("-" * 75)

    for cid in range(1, 6):
        cname = PROJECT_CLASS_NAMES[cid]
        p_count = int(np.sum(remapped == cid))
        p_area = p_count * pixel_area_km2
        p_pct = (p_count / total_valid_pixels * 100.0) if total_valid_pixels > 0 else 0.0
        print(f"{cid:<9} | {cname:<15} | {p_count:>14,} | {p_area:>10.2f} km^2 | {p_pct:>6.2f}%")

    print("-" * 75)
    total_area = total_valid_pixels * pixel_area_km2
    print(
        f"{'Total':<9} | {'All Classes':<15} | {total_valid_pixels:>14,} | {total_area:>10.2f} km^2 | 100.00%"
    )
    print("=" * 75 + "\n")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ESA WorldCover training labels at 20m.")
    parser.add_argument(
        "--config", type=str, default="configs/ahmedabad.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--resolution", type=float, default=20.0, help="Target resolution in meters (default: 20.0)"
    )
    parser.add_argument(
        "--crs", type=str, default="EPSG:32643", help="Target CRS (default: EPSG:32643)"
    )
    parser.add_argument("--ref", type=str, default=None, help="Optional reference raster path")
    parser.add_argument(
        "--out",
        type=str,
        default="data/ahmedabad_worldcover_labels.tif",
        help="Output labels GeoTIFF",
    )

    args = parser.parse_args()
    extract_worldcover_labels(
        config_path=args.config,
        target_crs=args.crs,
        resolution=args.resolution,
        reference_raster_path=args.ref,
        output_labels_path=args.out,
    )
