"""
UrbanPulse - True Color RGB & NDVI Preview Generator
Loads Ahmedabad Sentinel-2 surface reflectance bands from data/,
generates true-color RGB with contrast stretching, computes NDVI,
and exports high-resolution preview PNG images.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import rasterio
import yaml
from pipeline.build_composite import build_composite


def load_config(config_path: str | Path = "configs/ahmedabad.yaml") -> dict[str, Any]:
    """Loads city YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_preview_bands(
    year: int = 2024,
    config_path: str | Path = "configs/ahmedabad.yaml",
    data_dir: str | Path = "data",
) -> dict[str, Path]:
    """Ensures required optical bands exist in data/."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    band_names = ["red", "green", "blue", "nir"]
    band_paths = {b: data_path / f"ahmedabad_{year}_{b}.tif" for b in band_names}

    if all(p.exists() for p in band_paths.values()):
        print(f"[+] All required band GeoTIFFs found in {data_path.resolve()}")
        return band_paths

    print(f"[*] Missing band GeoTIFFs for {year}. Triggering build_composite...")
    build_composite(year=year, config_path=config_path, data_dir=data_dir)
    return band_paths


def contrast_stretch(band: np.ndarray, lower_p: float = 2.0, upper_p: float = 98.0) -> np.ndarray:
    """Applies robust percentile contrast stretching (2nd to 98th percentile)."""
    valid = band[np.isfinite(band) & (band > 0)]
    if len(valid) == 0:
        return np.zeros_like(band)

    p_low, p_high = np.percentile(valid, (lower_p, upper_p))
    if p_high <= p_low:
        p_high = p_low + 1e-4

    stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
    # Maintain NaNs/nodata as 0 for clean rendering
    stretched[~np.isfinite(band)] = 0.0
    return stretched


def generate_previews(
    year: int = 2024,
    config_path: str | Path = "configs/ahmedabad.yaml",
    data_dir: str | Path = "data",
) -> tuple[Path, Path]:
    """
    Loads band files, creates contrast-stretched RGB composite & NDVI map, and saves PNGs.
    """
    band_paths = ensure_preview_bands(year=year, config_path=config_path, data_dir=data_dir)
    data_path = Path(data_dir)

    print(f"\n[*] Loading band data with rasterio for Ahmedabad ({year})...")
    with rasterio.open(band_paths["red"]) as src_r:
        red = src_r.read(1).astype(np.float32)

    with rasterio.open(band_paths["green"]) as src_g:
        green = src_g.read(1).astype(np.float32)

    with rasterio.open(band_paths["blue"]) as src_b:
        blue = src_b.read(1).astype(np.float32)

    with rasterio.open(band_paths["nir"]) as src_n:
        nir = src_n.read(1).astype(np.float32)

    # 1. True-Color RGB Composite
    print("[*] Creating True-Color RGB composite with contrast stretching...")
    r_stretched = contrast_stretch(red, 2, 98)
    g_stretched = contrast_stretch(green, 2, 98)
    b_stretched = contrast_stretch(blue, 2, 98)

    rgb = np.dstack((r_stretched, g_stretched, b_stretched))

    rgb_out_path = data_path / f"preview_{year}_rgb.png"
    plt.figure(figsize=(12, 10), dpi=200)
    plt.imshow(rgb)
    plt.title(f"Ahmedabad {year} - Sentinel-2 True Color (RGB)", fontsize=14, pad=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(rgb_out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[+] Saved RGB preview: {rgb_out_path.resolve()}")

    # 2. NDVI Calculation
    print("[*] Computing Normalized Difference Vegetation Index (NDVI)...")
    denominator = nir + red
    valid_mask = (np.abs(denominator) > 1e-5) & np.isfinite(nir) & np.isfinite(red)
    ndvi = np.full_like(nir, fill_value=np.nan, dtype=np.float32)
    np.divide(nir - red, denominator, out=ndvi, where=valid_mask)
    np.clip(ndvi, -1.0, 1.0, out=ndvi, where=valid_mask)

    # Green-to-brown Earth colormap
    ndvi_out_path = data_path / f"preview_{year}_ndvi.png"
    plt.figure(figsize=(12, 10), dpi=200)

    # Brown (urban/bare soil) -> Khaki -> Light Green -> Dark Green (dense vegetation)
    colors = ["#7f3b08", "#b35806", "#e08214", "#fdb863", "#fee0b6", "#d9ef8b", "#91cf60", "#1a9850", "#006837"]
    earth_cmap = LinearSegmentedColormap.from_list("ndvi_earth", colors, N=256)
    earth_cmap.set_bad(color="#111111")

    im = plt.imshow(ndvi, cmap=earth_cmap, vmin=-0.2, vmax=0.7)
    cbar = plt.colorbar(im, fraction=0.036, pad=0.03)
    cbar.set_label("NDVI (Normalized Difference Vegetation Index)", fontsize=11)

    plt.title(f"Ahmedabad {year} - NDVI Vegetation & Built-up Distribution", fontsize=14, pad=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(ndvi_out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[+] Saved NDVI preview: {ndvi_out_path.resolve()}")

    print("\n" + "=" * 60)
    print("[+] Preview Generation Complete!")
    print(f"   - RGB Preview  : {rgb_out_path.name}")
    print(f"   - NDVI Preview : {ndvi_out_path.name}")
    print("=" * 60)

    return rgb_out_path, ndvi_out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RGB and NDVI preview maps.")
    parser.add_argument("year", type=int, nargs="?", default=2024, help="Target year (default: 2024)")
    parser.add_argument("--config", type=str, default="configs/ahmedabad.yaml", help="Path to config YAML")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()
    generate_previews(year=args.year, config_path=args.config, data_dir=args.data_dir)
