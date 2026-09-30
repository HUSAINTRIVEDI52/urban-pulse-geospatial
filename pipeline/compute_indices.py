"""
UrbanPulse - Spectral Indices Computation Module
Calculates Normalized Difference Vegetation Index (NDVI), Normalized Difference
Built-up Index (NDBI), and Modified Normalized Difference Water Index (MNDWI)
from Sentinel-2 optical bands.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_composite import build_composite


def load_config(config_path: str | Path = "configs/ahmedabad.yaml") -> dict[str, Any]:
    """Loads city YAML configuration."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_required_bands(
    year: int = 2024,
    required_bands: list[str] | None = None,
    config_path: str | Path = "configs/ahmedabad.yaml",
    data_dir: str | Path = "data",
) -> dict[str, Path]:
    """
    Ensures all required band GeoTIFFs exist locally.
    Triggers build_composite if missing.
    """
    if required_bands is None:
        required_bands = ["red", "green", "blue", "nir", "swir16"]

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    band_paths = {b: data_path / f"ahmedabad_{year}_{b}.tif" for b in required_bands}
    missing_bands = [b for b, p in band_paths.items() if not p.exists()]

    if not missing_bands:
        print(f"[+] All required band GeoTIFFs exist in {data_path.resolve()}")
        return band_paths

    print(f"[*] Missing band GeoTIFFs: {missing_bands}. Triggering build_composite...")
    build_composite(year=year, config_path=config_path, data_dir=data_dir)
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
            "tiled": True,
        }
    )

    out_arr = np.where(np.isnan(data), nodata_val, data).astype(np.float32)

    with rasterio.open(output_path, "w", **prof) as dst:
        dst.write(out_arr, 1)


def compute_indices(
    city: str = "ahmedabad",
    year: int = 2024,
    force: bool = False,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
) -> dict[str, dict[str, float]]:
    """
    Loads optical bands for a given city and year, calculates NDVI, NDBI, MNDWI,
    exports GeoTIFFs, and prints stats. Skips if outputs already exist unless force=True.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    index_paths = {
        "NDVI": data_path / f"{city_key}_{year}_ndvi.tif",
        "NDBI": data_path / f"{city_key}_{year}_ndbi.tif",
        "MNDWI": data_path / f"{city_key}_{year}_mndwi.tif",
    }

    # Check if all indices already exist
    if not force and all(p.exists() for p in index_paths.values()):
        print(
            f"[+] All spectral indices for {city} ({year}) already exist in {data_path.resolve()}. Skipping computation."
        )
        stats_summary = {}
        for name, out_path in index_paths.items():
            with rasterio.open(out_path) as src:
                arr = src.read(1)
                valid = arr[np.isfinite(arr) & (arr != -9999.0)]
                stats_summary[name] = {
                    "min": float(np.nanmin(valid)) if len(valid) > 0 else 0.0,
                    "max": float(np.nanmax(valid)) if len(valid) > 0 else 0.0,
                    "mean": float(np.nanmean(valid)) if len(valid) > 0 else 0.0,
                    "std": float(np.nanstd(valid)) if len(valid) > 0 else 0.0,
                    "file": str(out_path.name),
                }
        return stats_summary

    band_names = ["red", "green", "nir", "swir16"]
    band_paths = {b: data_path / f"{city_key}_{year}_{b}.tif" for b in band_names}
    missing_bands = [b for b, p in band_paths.items() if not p.exists()]
    if missing_bands:
        print(
            f"[*] Missing required bands for {city} ({year}): {missing_bands}. Triggering build_composite..."
        )
        build_composite(
            city=city, year=year, config_path=config_path, data_dir=data_path, force=force
        )

    print(f"\n[*] Loading raster bands for {city.capitalize()} ({year})...")
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
    print("[*] Computing spectral indices (NDVI, NDBI, MNDWI)...")
    ndvi = safe_normalized_difference(nir, red)
    ndbi = safe_normalized_difference(swir16, nir)
    mndwi = safe_normalized_difference(green, swir16)

    indices = {
        "NDVI": (ndvi, index_paths["NDVI"]),
        "NDBI": (ndbi, index_paths["NDBI"]),
        "MNDWI": (mndwi, index_paths["MNDWI"]),
    }

    stats_summary = {}

    print("\n" + "=" * 78)
    print(f"[*] Spectral Indices Summary for {city.capitalize()} ({year})")
    print("=" * 78)
    print(
        f"{'Index':<8} | {'Min':<10} | {'Max':<10} | {'Mean':<10} | {'Std Dev':<10} | {'Output File'}"
    )
    print("-" * 78)

    for name, (arr, out_path) in indices.items():
        save_geotiff(arr, out_path, profile)

        valid = arr[np.isfinite(arr) & (arr != -9999.0)]
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

    print("=" * 78 + "\n")
    return stats_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute spectral indices (NDVI, NDBI, MNDWI).")
    parser.add_argument(
        "--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)"
    )
    parser.add_argument("--year", type=int, default=2024, help="Target year (default: 2024)")
    parser.add_argument("--force", action="store_true", help="Force recalculate indices")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for data files")

    args = parser.parse_args()
    compute_indices(
        city=args.city,
        year=args.year,
        force=args.force,
        config_path=args.config,
        data_dir=args.data_dir,
    )
