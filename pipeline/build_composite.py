"""
UrbanPulse - Sentinel-2 Satellite Composite Builder
Fetches, masks clouds, composites, and exports surface reflectance GeoTIFFs
using Dask parallel computation with live progress tracking.
"""

import argparse
from pathlib import Path
from typing import Any
import yaml
from pystac_client import Client
import stackstac
import rioxarray
import rasterio
from dask.diagnostics import ProgressBar


def load_config(config_path: str | Path = "configs/ahmedabad.yaml") -> dict[str, Any]:
    """Loads city YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_composite(
    year: int = 2024,
    resolution: float = 60.0,
    max_cloud_cover: float = 10.0,
    config_path: str | Path = "configs/ahmedabad.yaml",
    data_dir: str | Path = "data",
) -> dict[str, Path]:
    """
    Builds a dry-season Sentinel-2 surface reflectance median composite.

    Args:
        year: Target year (e.g. 2024).
        resolution: Spatial resolution in meters (default: 60m for fast iteration).
        max_cloud_cover: Maximum scene cloud cover threshold.
        config_path: Path to city config YAML.
        data_dir: Destination folder for output GeoTIFFs.

    Returns:
        Dictionary mapping band names to saved GeoTIFF filepaths.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    city_name = config.get("city", {}).get("name", "Ahmedabad")
    bbox = config["spatial"]["bbox"]
    stac_url = config.get("stac", {}).get(
        "earth_search_url", "https://earth-search.aws.element84.com/v1"
    )
    primary_collection = config.get("stac", {}).get(
        "collections", {}
    ).get("sentinel_2", "sentinel-2-c1-l2a")

    optical_bands = ["red", "green", "blue", "nir", "swir16"]
    requested_assets = optical_bands + ["scl"]

    print("=" * 75)
    print(f"[*] UrbanPulse Composite Builder: {city_name} ({year})")
    print(f"    - Resolution     : {resolution}m")
    print(f"    - Bounding Box   : {bbox}")
    print(f"    - Primary STAC   : {stac_url} [{primary_collection}]")
    print(f"    - Output Folder  : {data_path.resolve()}")
    print("=" * 75)

    # ---------------------------------------------------------
    # STEP 1: Search STAC Catalog & Select Cloud-Free Scenes
    # ---------------------------------------------------------
    print("\n[Step 1/5] Querying STAC catalog for dry-season scenes...")
    client = Client.open(stac_url)
    datetime_range = f"{year}-01-01/{year}-02-29"

    search = client.search(
        collections=[primary_collection],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )
    items = list(search.items())

    # Fallback to broader Nov-Feb dry season if needed
    if not items:
        datetime_range = f"{year-1}-11-01/{year}-02-29"
        print(f"[*] Broadening search window to: {datetime_range}")
        search = client.search(
            collections=[primary_collection],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover + 15}},
        )
        items = list(search.items())

    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {city_name} in {year}.")

    # Group by unique MGRS tile to get 100% spatial coverage with lowest cloud cover
    tiles: dict[str, Any] = {}
    for item in items:
        z = str(item.properties.get("mgrs:utm_zone", ""))
        b = str(item.properties.get("mgrs:latitude_band", ""))
        g = str(item.properties.get("mgrs:grid_square", ""))
        tile_id = f"{z}{b}{g}" if (z and b and g) else item.id.split("_")[1].replace("T", "")
        
        if tile_id not in tiles or item.properties.get("eo:cloud_cover", 100) < tiles[tile_id].properties.get("eo:cloud_cover", 100):
            tiles[tile_id] = item

    selected_items = list(tiles.values())
    print(f"[+] Found {len(items)} matching scenes.")
    print(f"[+] Selected {len(selected_items)} optimal cloud-free scenes covering all {len(tiles)} MGRS tiles:")
    for it in selected_items:
        z = str(it.properties.get("mgrs:utm_zone", ""))
        b = str(it.properties.get("mgrs:latitude_band", ""))
        g = str(it.properties.get("mgrs:grid_square", ""))
        tile_name = f"{z}{b}{g}" if (z and b and g) else it.id.split("_")[1]
        cc = it.properties.get("eo:cloud_cover", 0.0)
        dt = str(it.datetime)[:10] if it.datetime else ""
        print(f"    - Tile {tile_name:<6}: {it.id} ({dt}, cloud: {cc:.2f}%)")

    # ---------------------------------------------------------
    # STEP 2: Construct Dask Lazy Raster Stack
    # ---------------------------------------------------------
    print(f"\n[Step 2/5] Building lazy Dask raster stack at {resolution}m (EPSG:32643)...")
    stack = stackstac.stack(
        selected_items,
        assets=requested_assets,
        epsg=32643,
        bounds_latlon=bbox,
        resolution=resolution,
        chunksize=1024,
    )

    nbytes_mb = stack.nbytes / (1024 * 1024)
    print(f"[*] Stack Grid Dimensions : {dict(stack.sizes)}")
    print(f"[*] Uncompressed Data Size : {nbytes_mb:.2f} MB")
    print(f"[*] Chunk Shape            : {stack.data.chunksize}")

    # ---------------------------------------------------------
    # STEP 3: Apply SCL Cloud & Shadow Quality Mask
    # ---------------------------------------------------------
    print("\n[Step 3/5] Applying Scene Classification Layer (SCL) pixel mask...")
    if "scl" in stack.band.values:
        scl = stack.sel(band="scl")
        # Mask out cloud shadows (3), clouds (8, 9), cirrus (10), snow (11), nodata (0)
        is_clear = (
            (scl != 3) & (scl != 8) & (scl != 9) & (scl != 10) & (scl != 11) & (scl != 0)
        )
        optical_stack = stack.sel(band=optical_bands).where(is_clear)
        print("[+] Applied SCL cloud/shadow filtering.")
    else:
        optical_stack = stack.sel(band=optical_bands)
        print("[!] SCL band not present, proceeding with optical stack directly.")

    # ---------------------------------------------------------
    # STEP 4: Compute Median Composite with Dask ProgressBar
    # ---------------------------------------------------------
    print("\n[Step 4/5] Executing Dask parallel computation for median composite...")
    print("           Streaming chunks from open COGs with live progress:")

    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with ProgressBar(minimum=0.2):
            median_composite = optical_stack.median(dim="time").compute()

    print("\n[+] Median composite computation complete!")

    # ---------------------------------------------------------
    # STEP 5: Export Band GeoTIFFs to Disk
    # ---------------------------------------------------------
    print("\n[Step 5/5] Exporting band GeoTIFFs to data/ directory...")
    output_paths = {}
    for band in optical_bands:
        out_file = data_path / f"ahmedabad_{year}_{band}.tif"
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

    print("\n" + "=" * 75)
    print("[+] All 5 surface reflectance bands exported successfully:")
    for b, p in output_paths.items():
        print(f"    - {b:<8}: {p.name} ({p.stat().st_size / (1024*1024):.2f} MB)")
    print("=" * 75)

    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Sentinel-2 composite for UrbanPulse.")
    parser.add_argument("year", type=int, nargs="?", default=2024, help="Target year (default: 2024)")
    parser.add_argument("--resolution", type=float, default=60.0, help="Spatial resolution in meters (default: 60.0)")
    parser.add_argument("--max-cloud", type=float, default=10.0, help="Max cloud cover percentage (default: 10.0)")
    parser.add_argument("--config", type=str, default="configs/ahmedabad.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()
    build_composite(
        year=args.year,
        resolution=args.resolution,
        max_cloud_cover=args.max_cloud,
        config_path=args.config,
        data_dir=args.data_dir,
    )
