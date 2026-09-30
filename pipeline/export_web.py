"""
UrbanPulse - Web App Data Exporter Module
Generates optimized, web-ready spatial overlays and statistical JSON feeds in web/data/{city}/:
1. Reprojected (EPSG:4326) transparent PNG overlays for each classified year.
2. Reprojected change detection transparent PNG overlay.
3. City metadata JSON (bounding box, centre, year catalog, color palette).
4. Analytical stats JSON (time-series area, sprawl entropy metrics, ring profiles, transition matrix).
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image
from rasterio.warp import Resampling, calculate_default_transform, reproject

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Standard Palette: RGBA with clean alpha channel
CLASS_RGBA = {
    0: (0, 0, 0, 0),  # NoData: Fully transparent
    1: (228, 26, 28, 240),  # Built-up: Bright Red
    2: (35, 139, 69, 240),  # Vegetation: Lush Green
    3: (31, 120, 180, 240),  # Water: Cerulean Blue
    4: (255, 217, 47, 240),  # Agriculture: Warm Golden Yellow
    5: (210, 180, 140, 240),  # Open land: Sand Tan
}

CLASS_HEX = {
    1: {"name": "Built-up", "color": "#e41a1c", "rgb": [228, 26, 28]},
    2: {"name": "Vegetation", "color": "#238b45", "rgb": [35, 139, 69]},
    3: {"name": "Water", "color": "#1f78b4", "rgb": [31, 120, 180]},
    4: {"name": "Agriculture", "color": "#ffd92f", "rgb": [255, 217, 47]},
    5: {"name": "Open land", "color": "#d2b48c", "rgb": [210, 180, 140]},
}

CHANGE_RGBA = {
    21: (46, 125, 50, 255),  # Veg -> Built-up (Dark Forest Green)
    41: (234, 88, 12, 255),  # Agri -> Built-up (Vibrant Orange)
    51: (139, 92, 246, 255),  # Open land -> Built-up (Purple)
    12: (15, 23, 42, 255),  # Built-up loss -> Veg (Dark Slate / Black)
    13: (15, 23, 42, 255),  # Built-up loss -> Water
    14: (15, 23, 42, 255),  # Built-up loss -> Agri
    15: (15, 23, 42, 255),  # Built-up loss -> Open land
}


def load_city_config(
    city: str = "ahmedabad", config_path: str | Path | None = None
) -> dict[str, Any]:
    """Loads city YAML configuration."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"City configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def reproject_raster_to_wgs84(
    src_path: Path, dst_crs: str = "EPSG:4326"
) -> tuple[np.ndarray, list[float]]:
    """
    Reprojects raster array to WGS84 (EPSG:4326) using Nearest-Neighbour resampling.
    Returns (reprojected_2d_array, [west, south, east, north]).
    """
    with rasterio.open(src_path) as src:
        src_crs = src.crs
        src_transform = src.transform
        src_data = src.read(1)

        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds
        )

        dst_data = np.zeros((dst_height, dst_width), dtype=src_data.dtype)

        reproject(
            source=src_data,
            destination=dst_data,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )

        # Bounds: [west, south, east, north]
        bounds = rasterio.transform.array_bounds(dst_height, dst_width, dst_transform)
        bounds_wgs84 = [round(float(b), 6) for b in bounds]

        return dst_data, bounds_wgs84


def save_classified_overlay_png(data_2d: np.ndarray, output_png_path: Path) -> Path:
    """
    Converts 2D categorical raster into RGBA image with transparent NoData.
    """
    h, w = data_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for cid, color in CLASS_RGBA.items():
        mask = data_2d == cid
        rgba[mask] = color

    img = Image.fromarray(rgba, mode="RGBA")
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_png_path, format="PNG", optimize=True)
    return output_png_path


def save_change_overlay_png(data_2d: np.ndarray, output_png_path: Path) -> Path:
    """
    Converts 2D change code raster into RGBA image (transparent outside changed pixels).
    """
    h, w = data_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    for code, color in CHANGE_RGBA.items():
        mask = data_2d == code
        rgba[mask] = color

    img = Image.fromarray(rgba, mode="RGBA")
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_png_path, format="PNG", optimize=True)
    return output_png_path


def export_web_data(
    city: str = "ahmedabad",
    data_dir: str | Path = "data",
    web_dir: str | Path = "web/data",
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Exports all web visualization assets into web/data/{city}/.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    web_dest_dir = Path(web_dir) / city_key
    web_dest_dir.mkdir(parents=True, exist_ok=True)

    config = load_city_config(city=city, config_path=config_path)
    city_name = config.get("city", {}).get("name", city.capitalize())

    print("=" * 80)
    print(f" URBANPULSE WEB ASSET EXPORTER: {city_name.upper()}")
    print(f" Source Directory : {data_path.resolve()}")
    print(f" Target Web Dir   : {web_dest_dir.resolve()}")
    print("=" * 80)

    # 1. Discover all classified rasters
    pattern = re.compile(rf"^{re.escape(city_key)}_(\d{{4}})_classified\.tif$")
    year_raster_map: dict[int, Path] = {}
    for f in data_path.glob(f"{city_key}_*_classified.tif"):
        match = pattern.match(f.name)
        if match:
            yr = int(match.group(1))
            year_raster_map[yr] = f

    sorted_years = sorted(year_raster_map.keys())
    if not sorted_years:
        raise FileNotFoundError(
            f"No classified rasters found for {city_key} in {data_path.resolve()}"
        )

    print(f"[+] Found {len(sorted_years)} classified years: {sorted_years}")

    # 2. Export each classified year as transparent WGS84 PNG
    exported_bounds = None
    print("\n[Step 1/4] Reprojecting and exporting annual PNG overlays (EPSG:4326)...")
    for yr in sorted_years:
        src_tif = year_raster_map[yr]
        dst_png = web_dest_dir / f"{yr}.png"

        reproj_arr, bounds_wgs84 = reproject_raster_to_wgs84(src_tif, dst_crs="EPSG:4326")
        if exported_bounds is None:
            exported_bounds = bounds_wgs84

        save_classified_overlay_png(reproj_arr, dst_png)
        size_kb = dst_png.stat().st_size / 1024.0
        print(f"    - Year {yr}: -> {dst_png.name} ({size_kb:.1f} KB)")

    # 3. Export Change Map PNG overlay
    print("\n[Step 2/4] Reprojecting and exporting change detection overlay...")
    change_tifs = list(data_path.glob(f"{city_key}_change_*_*.tif"))
    exported_change_maps = []
    for c_tif in change_tifs:
        parts = c_tif.stem.split("_")
        if len(parts) >= 4:
            s_yr, e_yr = parts[-2], parts[-1]
            dst_change_png = web_dest_dir / f"change_{s_yr}_{e_yr}.png"
            reproj_change_arr, _ = reproject_raster_to_wgs84(c_tif, dst_crs="EPSG:4326")
            save_change_overlay_png(reproj_change_arr, dst_change_png)
            size_kb = dst_change_png.stat().st_size / 1024.0
            print(f"    - Change ({s_yr}->{e_yr}): -> {dst_change_png.name} ({size_kb:.1f} KB)")
            exported_change_maps.append((s_yr, e_yr, dst_change_png))

    # 4. Generate meta.json
    print("\n[Step 3/4] Compiling metadata into meta.json...")
    spatial_cfg = config.get("spatial", {})
    city_cfg = config.get("city", {})

    center_lat = spatial_cfg.get("center_lat") or city_cfg.get("latitude", 23.0225)
    center_lon = spatial_cfg.get("center_lon") or city_cfg.get("longitude", 72.5714)
    bbox_config = spatial_cfg.get("bbox", exported_bounds)

    meta_payload = {
        "city": city_name,
        "state": city_cfg.get("state", "Gujarat"),
        "country": city_cfg.get("country", "India"),
        "center": [float(center_lat), float(center_lon)],
        "bounds": exported_bounds if exported_bounds else bbox_config,
        "years": sorted_years,
        "classes": CLASS_HEX,
        "change_legend": {
            "veg_to_built": {"label": "Vegetation -> Built-up", "color": "#2e7d32"},
            "agri_to_built": {"label": "Agriculture -> Built-up", "color": "#ea580c"},
            "open_to_built": {"label": "Open land -> Built-up", "color": "#8b5cf6"},
            "built_loss": {"label": "Built-up Loss -> Other", "color": "#0f172a"},
        },
    }

    meta_file = web_dest_dir / "meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"[+] Saved {meta_file.name} ({meta_file.stat().st_size / 1024:.1f} KB)")

    # 5. Generate stats.json combining all tabular datasets
    print("\n[Step 4/4] Compiling multi-year statistics into stats.json...")

    # Class Areas
    class_areas_list = []
    class_areas_csv = data_path / f"{city_key}_class_areas.csv"
    if class_areas_csv.exists():
        df_areas = pd.read_csv(class_areas_csv)
        for _, r in df_areas.iterrows():
            class_areas_list.append(
                {
                    "year": int(r["Year"]),
                    "built_up_km2": float(r["Built-up"]),
                    "vegetation_km2": float(r["Vegetation"]),
                    "water_km2": float(r["Water"]),
                    "agriculture_km2": float(r["Agriculture"]),
                    "open_land_km2": float(r["Open land"]),
                    "total_area_km2": float(r["Total_Area_km2"]),
                }
            )

    # Metrics
    metrics_list = []
    metrics_csv = data_path / f"{city_key}_metrics.csv"
    if metrics_csv.exists():
        df_metrics = pd.read_csv(metrics_csv)
        for _, r in df_metrics.iterrows():
            metrics_list.append(
                {
                    "year": int(r["year"]),
                    "builtup_km2": float(r["builtup_km2"]),
                    "annual_growth_pct": float(r["annual_growth_pct"]),
                    "cagr_pct": float(r["cagr_from_start_pct"]),
                    "shannon_entropy": float(r["shannon_entropy"]),
                    "core_builtup_km2": float(r["core_builtup_0_6km_km2"]),
                    "core_share_pct": float(r["core_share_0_6km_pct"]),
                    "periphery_builtup_km2": float(r["periphery_builtup_gt_12km_km2"]),
                    "periphery_share_pct": float(r["periphery_share_gt_12km_pct"]),
                }
            )

    # Rings
    rings_dict: dict[str, list[dict[str, Any]]] = {}
    rings_csv = data_path / f"{city_key}_rings.csv"
    if rings_csv.exists():
        df_rings = pd.read_csv(rings_csv)
        for yr in sorted_years:
            sub = df_rings[df_rings["year"] == yr].sort_values("ring_start_km")
            rings_dict[str(yr)] = [
                {
                    "ring_start_km": float(r["ring_start_km"]),
                    "ring_end_km": float(r["ring_end_km"]),
                    "ring_label": f"{int(r['ring_start_km'])}-{int(r['ring_end_km'])} km",
                    "builtup_km2": float(r["builtup_km2"]),
                    "valid_km2": float(r["valid_km2"]),
                    "builtup_pct": float(r["builtup_pct"]),
                }
                for _, r in sub.iterrows()
            ]

    # Transitions
    transitions_dict = {}
    for trans_csv in data_path.glob(f"{city_key}_transition_*_*.csv"):
        parts = trans_csv.stem.split("_")
        if len(parts) >= 4:
            s_yr, e_yr = parts[-2], parts[-1]
            df_trans = pd.read_csv(trans_csv, index_col=0)
            transitions_dict[f"{s_yr}_{e_yr}"] = df_trans.to_dict()

    stats_payload = {
        "city": city_name,
        "years": sorted_years,
        "class_areas": class_areas_list,
        "metrics": metrics_list,
        "rings": rings_dict,
        "transitions": transitions_dict,
    }

    stats_file = web_dest_dir / "stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)
    print(f"[+] Saved {stats_file.name} ({stats_file.stat().st_size / 1024:.1f} KB)")

    # 6. Summary of all exported files
    print("\n" + "=" * 80)
    print(f"[*] EXPORTED WEB ASSETS SUMMARY ({web_dest_dir.resolve()}):")
    print("=" * 80)
    total_bytes = 0
    all_exported_files = sorted(list(web_dest_dir.glob("*")), key=lambda p: p.name)
    for p in all_exported_files:
        st_size = p.stat().st_size
        total_bytes += st_size
        if st_size > 1024 * 1024:
            size_str = f"{st_size / (1024*1024):>6.2f} MB"
        else:
            size_str = f"{st_size / 1024:>6.1f} KB"
        print(f"  - {p.name:<32} : {size_str}")
    print("-" * 80)
    print(
        f"  Total Web Bundle Size : {total_bytes / (1024*1024):.2f} MB ({len(all_exported_files)} files)"
    )
    print("=" * 80 + "\n")

    return {
        "web_dir": web_dest_dir,
        "files": all_exported_files,
        "meta": meta_payload,
        "stats": stats_payload,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export web application assets, transparent PNG overlays, and JSON datasets."
    )
    parser.add_argument(
        "--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory with data rasters and CSVs"
    )
    parser.add_argument("--web-dir", type=str, default="web/data", help="Target web data directory")
    parser.add_argument("--config", type=str, default=None, help="Custom city YAML config path")

    args = parser.parse_args()
    export_web_data(
        city=args.city,
        data_dir=args.data_dir,
        web_dir=args.web_dir,
        config_path=args.config,
    )
