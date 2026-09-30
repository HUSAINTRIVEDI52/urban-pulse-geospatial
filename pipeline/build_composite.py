"""
UrbanPulse - Sentinel-2 Satellite Composite Builder
Fetches, masks clouds, composites, and exports surface reflectance GeoTIFFs
using Dask parallel computation with tile-stratified scene selection,
live progress tracking, and AOI nodata validation.
"""

import argparse
import calendar
from pathlib import Path
from typing import Any
import numpy as np
import yaml
from pystac_client import Client
import stackstac
import rioxarray
import rasterio
from dask.diagnostics import ProgressBar


def load_city_config(city: str = "ahmedabad", config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads city YAML configuration."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"City configuration file not found: {cfg_file.resolve()}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dry_season_range(year: int, config: dict[str, Any]) -> str:
    """Constructs ISO 8601 dry season date range from YAML config."""
    temporal_cfg = config.get("temporal", {})
    dry_season_cfg = temporal_cfg.get("dry_season", {})

    start_month = dry_season_cfg.get("start_month", 11)
    end_month = dry_season_cfg.get("end_month", 2)

    start_year = year - 1 if start_month > end_month else year
    start_date = f"{start_year:04d}-{start_month:02d}-01"

    _, last_day = calendar.monthrange(year, end_month)
    end_date = f"{year:04d}-{end_month:02d}-{last_day:02d}"

    return f"{start_date}/{end_date}"


def build_composite(
    city: str = "ahmedabad",
    year: int = 2024,
    resolution: float = 60.0,
    max_cloud_cover: float = 10.0,
    scenes_per_tile: int = 4,
    force: bool = False,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
) -> tuple[dict[str, Path], float]:
    """
    Builds a dry-season Sentinel-2 surface reflectance median composite.

    Args:
        city: Target city name (e.g. 'ahmedabad').
        year: Target year (e.g. 2023, 2024).
        resolution: Spatial resolution in meters (default: 60.0m).
        max_cloud_cover: Maximum scene cloud cover threshold (default: 10.0%).
        scenes_per_tile: Number of least-cloudy scenes per MGRS tile (default: 4).
        force: If True, overwrites existing band GeoTIFFs.
        config_path: Path to city config YAML.
        data_dir: Destination folder for output GeoTIFFs.

    Returns:
        Tuple of (output_paths_dict, nodata_percentage).
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    city_key = city.lower()
    optical_bands = ["red", "green", "blue", "nir", "swir16"]
    band_paths = {b: data_path / f"{city_key}_{year}_{b}.tif" for b in optical_bands}

    # Skip if outputs already exist and force is not set
    if not force and all(p.exists() for p in band_paths.values()):
        print(f"[+] All composite bands for {city} ({year}) already exist in {data_path.resolve()}. Skipping download.")
        with rasterio.open(band_paths["red"]) as src:
            red_arr = src.read(1)
            nan_count = int(np.isnan(red_arr).sum() + (red_arr == -9999.0).sum())
            nodata_pct = (nan_count / red_arr.size) * 100.0
        print(f"[*] Verified existing composite: {nodata_pct:.4f}% NoData pixels.")
        return band_paths, nodata_pct

    config = load_city_config(city=city, config_path=config_path)
    city_name = config.get("city", {}).get("name", city.capitalize())
    bbox = config["spatial"]["bbox"]
    crs = config["spatial"].get("crs", "EPSG:4326")
    stac_url = config.get("stac", {}).get(
        "earth_search_url", "https://earth-search.aws.element84.com/v1"
    )
    primary_collection = config.get("stac", {}).get(
        "collections", {}
    ).get("sentinel_2", "sentinel-2-c1-l2a")

    requested_assets = optical_bands + ["scl"]

    print("=" * 78)
    print(f"[*] UrbanPulse Sentinel-2 Composite Builder: {city_name} ({year})")
    print(f"    - Resolution        : {resolution}m (EPSG:32643)")
    print(f"    - Bounding Box      : {bbox}")
    print(f"    - STAC Endpoint     : {stac_url} [{primary_collection}]")
    print(f"    - Output Directory  : {data_path.resolve()}")
    print("=" * 78)

    # 1. Search STAC
    datetime_range = get_dry_season_range(year, config)
    print(f"\n[Step 1/5] Searching dry-season scenes ({datetime_range})...")
    client = Client.open(stac_url)

    search = client.search(
        collections=[primary_collection],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())

    if not items:
        print(f"[*] Broadening cloud search to < {max_cloud_cover + 15}%...")
        search = client.search(
            collections=[primary_collection],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover + 15}},
        )
        items = list(search.items())

    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {city_name} in {year} with cloud < {max_cloud_cover+15}%.")

    # Group scenes by intersecting MGRS tile
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
        cloud_list = [round(it.properties.get("eo:cloud_cover", 0), 2) for it in chosen]
        print(f"    - Tile {tile_id:<6}: Selected {len(chosen)} scenes (Clouds: {cloud_list}%)")

    print(f"[+] Total scenes selected for composite: {len(selected_items)}")

    # 2. Build Dask Stack
    print(f"\n[Step 2/5] Constructing Dask raster stack at {resolution}m (EPSG:32643)...")
    stack = stackstac.stack(
        selected_items,
        assets=requested_assets,
        epsg=32643,
        bounds_latlon=bbox,
        resolution=resolution,
        chunksize=1024,
    )

    nbytes_mb = stack.nbytes / (1024 * 1024)
    print(f"[*] Grid Dimensions: {dict(stack.sizes)}")
    print(f"[*] Array Memory Footprint: {nbytes_mb:.2f} MB")

    # 3. Cloud Masking
    print("\n[Step 3/5] Applying Scene Classification Layer (SCL) cloud & shadow mask...")
    if "scl" in stack.band.values:
        scl = stack.sel(band="scl")
        is_clear = (
            (scl != 3) & (scl != 8) & (scl != 9) & (scl != 10) & (scl != 11) & (scl != 0)
        )
        optical_stack = stack.sel(band=optical_bands).where(is_clear)
        print("[+] Applied SCL pixel quality mask.")
    else:
        optical_stack = stack.sel(band=optical_bands)
        print("[!] SCL band not present, proceeding with unmasked optical stack.")

    # 4. Compute Median Composite
    print("\n[Step 4/5] Computing temporal median composite across tile scenes...")
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with ProgressBar(minimum=0.2):
            median_composite = optical_stack.median(dim="time").compute()

    print("\n[+] Median composite computation completed!")

    # Check NoData
    red_vals = median_composite.sel(band="red").values
    total_grid_pixels = red_vals.size
    nan_pixels = int(np.isnan(red_vals).sum())
    nodata_percentage = (nan_pixels / total_grid_pixels) * 100.0

    print("\n" + "=" * 78)
    print(f"[*] Composite Quality & Coverage Check:")
    print(f"    - Total AOI Pixels     : {total_grid_pixels:>10,}")
    print(f"    - Valid Data Pixels    : {total_grid_pixels - nan_pixels:>10,} ({(100 - nodata_percentage):.2f}%)")
    print(f"    - NoData / NaN Pixels  : {nan_pixels:>10,} ({nodata_percentage:.4f}%)")
    if nodata_percentage < 1.0:
        print(f"    - Status               : PASSED (NoData {nodata_percentage:.4f}% < 1.00%)")
    else:
        print(f"    - Status               : WARNING (NoData {nodata_percentage:.4f}% >= 1.00%)")
    print("=" * 78)

    # 5. Export GeoTIFFs
    print("\n[Step 5/5] Exporting band GeoTIFFs to disk...")
    output_paths = {}
    for band in optical_bands:
        out_file = band_paths[band]
        print(f"    - Writing {band:<7} -> {out_file.name}...", end="", flush=True)

        band_da = median_composite.sel(band=band)
        band_da.rio.write_crs("EPSG:32643", inplace=True)
        band_da.rio.to_raster(
            out_file,
            driver="GTiff",
            dtype="float32",
            compress="lzw",
            tiled=True,
        )
        output_paths[band] = out_file
        print(" [Done]")

    print("\n" + "=" * 78)
    print("[+] All band GeoTIFFs exported successfully to data/:")
    for b, p in output_paths.items():
        print(f"    - {b:<8}: {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 78)

    return output_paths, nodata_percentage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Sentinel-2 composite for UrbanPulse.")
    parser.add_argument("--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)")
    parser.add_argument("--year", type=int, default=2024, help="Target year (default: 2024)")
    parser.add_argument("--resolution", type=float, default=60.0, help="Resolution in meters (default: 60.0)")
    parser.add_argument("--max-cloud", type=float, default=10.0, help="Max cloud cover percentage (default: 10.0)")
    parser.add_argument("--scenes-per-tile", type=int, default=4, help="Scenes per MGRS tile (default: 4)")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing composite bands")
    parser.add_argument("--config", type=str, default=None, help="Path to city config YAML")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()
    build_composite(
        city=args.city,
        year=args.year,
        resolution=args.resolution,
        max_cloud_cover=args.max_cloud,
        scenes_per_tile=args.scenes_per_tile,
        force=args.force,
        config_path=args.config,
        data_dir=args.data_dir,
    )
