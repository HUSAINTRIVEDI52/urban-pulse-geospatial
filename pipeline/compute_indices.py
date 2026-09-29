"""
UrbanPulse - Spectral Indices Computation Module
Calculates Normalized Difference Vegetation Index (NDVI), Normalized Difference
Built-up Index (NDBI), and Modified Normalized Difference Water Index (MNDWI)
from Sentinel-2 optical bands.
"""

import argparse
from pathlib import Path
from typing import Any
import numpy as np
import rasterio
import yaml
from pystac_client import Client
import stackstac
import rioxarray


def load_config(config_path: str | Path = "configs/ahmedabad.yaml") -> dict[str, Any]:
    """Loads city YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_required_bands(
    year: int = 2024,
    required_bands: list[str] | None = None,
    config_path: str | Path = "configs/ahmedabad.yaml",
    data_dir: str | Path = "data",
    resolution: float = 30.0,
) -> dict[str, Path]:
    """
    Ensures all required band GeoTIFFs (red, green, nir, swir16) exist locally.
    Downloads and composites any missing bands from Earth Search STAC.
    """
    if required_bands is None:
        required_bands = ["red", "green", "nir", "swir16"]

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    band_paths = {b: data_path / f"ahmedabad_{year}_{b}.tif" for b in required_bands}
    missing_bands = [b for b, p in band_paths.items() if not p.exists()]

    if not missing_bands:
        print(f"[+] All required band GeoTIFFs exist in {data_path.resolve()}")
        return band_paths

    print(f"[*] Missing band GeoTIFFs: {missing_bands}. Fetching from STAC catalog...")
    config = load_config(config_path)
    bbox = config["spatial"]["bbox"]
    stac_url = config.get("stac", {}).get(
        "earth_search_url", "https://earth-search.aws.element84.com/v1"
    )

    client = Client.open(stac_url)
    datetime_range = f"{year}-01-01/{year}-02-29"
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": 10}},
    )
    items = list(search.items())

    if not items:
        datetime_range = f"{year-1}-11-01/{year}-02-29"
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": 25}},
        )
        items = list(search.items())

    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for year {year} in STAC.")

    tiles: dict[str, Any] = {}
    for item in items:
        tile_id = item.id.split("_")[1]
        if tile_id not in tiles:
            tiles[tile_id] = item
        else:
            if item.properties.get("eo:cloud_cover", 100) < tiles[tile_id].properties.get("eo:cloud_cover", 100):
                tiles[tile_id] = item

    selected_items = list(tiles.values())
    print(f"[+] Selected {len(selected_items)} optimal cloud-free scenes across {len(tiles)} MGRS tiles covering 100% of AOI.")

    print(f"[+] Stacking {len(selected_items)} scenes for bands {missing_bands} at {resolution}m...")
    stack = stackstac.stack(
        selected_items,
        assets=missing_bands,
        epsg=32643,
        bounds_latlon=bbox,
        resolution=resolution,
        chunksize=1024,
    )

    median_ds = stack.median(dim="time").compute()

    for band in missing_bands:
        band_da = median_ds.sel(band=band)
        out_file = band_paths[band]
        band_da.rio.write_crs("EPSG:32643", inplace=True)
        band_da.rio.to_raster(out_file, driver="GTiff", dtype="float32")
        print(f"[+] Saved GeoTIFF: {out_file.name}")

    return band_paths


def safe_normalized_difference(
    band_a: np.ndarray, band_b: np.ndarray, min_denom: float = 1e-5
) -> np.ndarray:
    """
    Computes (band_a - band_b) / (band_a + band_b) safely, preventing division by zero.
    Returns NaN for invalid or zero-denominator pixels, with values clipped to [-1, 1].
    """
    numerator = band_a - band_b
    denominator = band_a + band_b

    # Valid mask where denominator magnitude is above epsilon and both bands are finite
    valid_mask = (np.abs(denominator) > min_denom) & np.isfinite(band_a) & np.isfinite(band_b)

    index = np.full_like(band_a, fill_value=np.nan, dtype=np.float32)
    np.divide(numerator, denominator, out=index, where=valid_mask)
    np.clip(index, -1.0, 1.0, out=index, where=valid_mask)

    return index


def save_geotiff(
    data: np.ndarray, output_path: Path, profile: dict[str, Any], nodata_val: float = -9999.0
) -> None:
    """Saves a 2D numpy array as a single-band GeoTIFF with spatial metadata."""
    prof = profile.copy()
    prof.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "float32",
            "nodata": nodata_val,
            "compress": "lzw",
        }
    )

    # Replace NaNs with nodata value for GeoTIFF export
    out_arr = np.where(np.isnan(data), nodata_val, data).astype(np.float32)

    with rasterio.open(output_path, "w", **prof) as dst:
        dst.write(out_arr, 1)


def compute_indices(
    year: int = 2024,
    config_path: str | Path = "configs/ahmedabad.yaml",
    data_dir: str | Path = "data",
) -> dict[str, dict[str, float]]:
    """
    Loads optical bands, calculates NDVI, NDBI, MNDWI, exports GeoTIFFs, and prints stats.
    """
    data_path = Path(data_dir)
    band_paths = ensure_required_bands(
        year=year,
        required_bands=["red", "green", "nir", "swir16"],
        config_path=config_path,
        data_dir=data_path,
    )

    print(f"\n[*] Loading raster bands for Ahmedabad ({year})...")
    with rasterio.open(band_paths["red"]) as src:
        red = src.read(1).astype(np.float32)
        profile = src.profile

    with rasterio.open(band_paths["green"]) as src:
        green = src.read(1).astype(np.float32)

    with rasterio.open(band_paths["nir"]) as src:
        nir = src.read(1).astype(np.float32)

    with rasterio.open(band_paths["swir16"]) as src:
        swir16 = src.read(1).astype(np.float32)

    # Compute Indices
    print("[*] Computing spectral indices...")
    ndvi = safe_normalized_difference(nir, red)
    ndbi = safe_normalized_difference(swir16, nir)
    mndwi = safe_normalized_difference(green, swir16)

    indices = {
        "NDVI": (ndvi, data_path / f"ahmedabad_{year}_ndvi.tif"),
        "NDBI": (ndbi, data_path / f"ahmedabad_{year}_ndbi.tif"),
        "MNDWI": (mndwi, data_path / f"ahmedabad_{year}_mndwi.tif"),
    }

    stats_summary = {}

    print("\n" + "=" * 75)
    print(f"[*] Spectral Indices Summary for Ahmedabad ({year})")
    print("=" * 75)
    print(
        f"{'Index':<8} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std Dev':<10} | {'Output File'}"
    )
    print("-" * 75)

    for name, (arr, out_path) in indices.items():
        save_geotiff(arr, out_path, profile)

        valid = arr[np.isfinite(arr)]
        if len(valid) > 0:
            val_min = float(np.nanmin(valid))
            val_max = float(np.nanmax(valid))
            val_mean = float(np.nanmean(valid))
            val_std = float(np.nanstd(valid))
        else:
            val_min = val_max = val_mean = val_std = 0.0

        stats_summary[name] = {
            "min": val_min,
            "max": val_max,
            "mean": val_mean,
            "std": val_std,
            "file": str(out_path.name),
        }

        print(
            f"{name:<8} | {val_min:<10.4f} | {val_max:<10.4f} | {val_mean:<10.4f} | {val_std:<10.4f} | {out_path.name}"
        )

    print("=" * 75 + "\n")
    return stats_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute spectral indices (NDVI, NDBI, MNDWI).")
    parser.add_argument("year", type=int, nargs="?", default=2024, help="Target year (default: 2024)")
    parser.add_argument("--config", type=str, default="configs/ahmedabad.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for data files")

    args = parser.parse_args()
    compute_indices(year=args.year, config_path=args.config, data_dir=args.data_dir)
