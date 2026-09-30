"""
UrbanPulse - Multi-Year Composite & Radiometric Diagnostic Module
Analyzes Ahmedabad annual satellite composites (2018-2024), queries STAC
scene metadata, evaluates nodata %, cloud cover, and calculates mean spectral
indices (NDVI, NDBI, MNDWI, Red, NIR) over stable Built-up & Water pixels.
Generates side-by-side true-color RGB previews for 2018, 2019, and 2020.
"""

import argparse
import calendar
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from pystac_client import Client


def load_config(city: str = "ahmedabad", config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads city YAML configuration file."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dry_season_range(year: int, config: dict[str, Any]) -> str:
    """Constructs ISO 8601 dry season date range from YAML config (Nov to Feb)."""
    temporal_cfg = config.get("temporal", {})
    dry_season_cfg = temporal_cfg.get("dry_season", {})

    start_month = dry_season_cfg.get("start_month", 11)
    end_month = dry_season_cfg.get("end_month", 2)

    start_year = year - 1 if start_month > end_month else year
    start_date = f"{start_year:04d}-{start_month:02d}-01"

    _, last_day = calendar.monthrange(year, end_month)
    end_date = f"{year:04d}-{end_month:02d}-{last_day:02d}"

    return f"{start_date}/{end_date}"


def query_stac_scenes_for_year(
    year: int,
    config: dict[str, Any],
    max_cloud_cover: float = 10.0,
    scenes_per_tile: int = 4,
) -> tuple[int, list[str], float]:
    """
    Queries STAC catalog for Sentinel-2 L2A scenes used in the dry-season composite.

    Returns:
        (scene_count, scene_dates_list, mean_cloud_cover)
    """
    bbox = config["spatial"]["bbox"]
    stac_cfg = config.get("stac", {})
    stac_url = stac_cfg.get("earth_search_url", "https://earth-search.aws.element84.com/v1")
    collection = stac_cfg.get("collections", {}).get("sentinel_2", "sentinel-2-c1-l2a")

    datetime_range = get_dry_season_range(year, config)

    try:
        client = Client.open(stac_url)
        search = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=datetime_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        )
        items = list(search.items())

        if not items:
            search = client.search(
                collections=[collection],
                bbox=bbox,
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": max_cloud_cover + 15.0}},
            )
            items = list(search.items())

        if not items:
            return 0, [], 0.0

        # Group by MGRS tile and pick lowest cloud cover per tile
        tile_dict: dict[str, list[Any]] = {}
        for item in items:
            z = str(item.properties.get("mgrs:utm_zone", ""))
            b = str(item.properties.get("mgrs:latitude_band", ""))
            g = str(item.properties.get("mgrs:grid_square", ""))
            tile_id = f"{z}{b}{g}" if (z and b and g) else item.id.split("_")[1].replace("T", "")
            tile_dict.setdefault(tile_id, []).append(item)

        selected_items = []
        for _, t_items in tile_dict.items():
            t_items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
            selected_items.extend(t_items[:scenes_per_tile])

        dates = sorted(
            list(
                {
                    (
                        it.datetime.strftime("%Y-%m-%d")
                        if it.datetime
                        else str(it.properties.get("datetime"))[:10]
                    )
                    for it in selected_items
                }
            )
        )
        clouds = [it.properties.get("eo:cloud_cover", 0.0) for it in selected_items]
        mean_cloud = float(np.mean(clouds)) if clouds else 0.0

        return len(selected_items), dates, mean_cloud

    except Exception as err:
        print(f"[!] Warning: STAC query failed for {year}: {err}")
        return 0, [], 0.0


def contrast_stretch(band: np.ndarray, lower_p: float = 2.0, upper_p: float = 98.0) -> np.ndarray:
    """Applies percentile contrast stretch (2nd to 98th percentile)."""
    valid = band[np.isfinite(band) & (band > 0)]
    if len(valid) == 0:
        return np.zeros_like(band)

    p_low, p_high = np.percentile(valid, (lower_p, upper_p))
    if p_high <= p_low:
        p_high = p_low + 1e-4

    stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
    stretched[~np.isfinite(band)] = 0.0
    return stretched


def generate_rgb_comparison_preview(
    city: str = "ahmedabad",
    years: tuple[int, int, int] = (2018, 2019, 2020),
    data_dir: str | Path = "data",
    output_path: str | Path | None = None,
) -> Path:
    """
    Creates and saves a side-by-side RGB true-color preview for 2018, 2019, and 2020.
    """
    data_path = Path(data_dir)
    out_file = (
        Path(output_path)
        if output_path
        else data_path / f"{city.lower()}_rgb_comparison_2018_2019_2020.png"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    fig.patch.set_facecolor("#090d16")

    for idx, year in enumerate(years):
        ax = axes[idx]
        ax.set_facecolor("#0b1120")

        red_p = data_path / f"{city}_{year}_red.tif"
        green_p = data_path / f"{city}_{year}_green.tif"
        blue_p = data_path / f"{city}_{year}_blue.tif"

        if not (red_p.exists() and green_p.exists() and blue_p.exists()):
            ax.text(
                0.5,
                0.5,
                f"Missing bands for {year}",
                color="white",
                ha="center",
                va="center",
            )
            ax.axis("off")
            continue

        with rasterio.open(red_p) as src_r:
            red = src_r.read(1).astype(np.float32)
        with rasterio.open(green_p) as src_g:
            green = src_g.read(1).astype(np.float32)
        with rasterio.open(blue_p) as src_b:
            blue = src_b.read(1).astype(np.float32)

        r_s = contrast_stretch(red, 2, 98)
        g_s = contrast_stretch(green, 2, 98)
        b_s = contrast_stretch(blue, 2, 98)

        rgb = np.dstack((r_s, g_s, b_s))
        ax.imshow(rgb)
        ax.set_title(
            f"{city.capitalize()} - {year} Dry Season RGB",
            color="#f8fafc",
            fontsize=13,
            fontweight="bold",
            pad=10,
        )
        ax.axis("off")

    plt.suptitle(
        f"UrbanPulse Radiometric Comparison (True Color RGB) - {city.capitalize()}",
        color="#38bdf8",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print(f"\n[+] Side-by-side RGB comparison saved to: {out_file.resolve()}")
    return out_file


def diagnose_years(
    city: str = "ahmedabad",
    start_year: int = 2018,
    end_year: int = 2024,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
    preview_output: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Performs full multi-year diagnostic across all years:
    - Number of scenes used & scene dates
    - Nodata %
    - Mean cloud cover
    - Mean NDVI, NDBI, MNDWI, Red, NIR over pixels that are Water or Built-up in ALL years
    """
    config = load_config(city=city, config_path=config_path)
    data_path = Path(data_dir)
    years = list(range(start_year, end_year + 1))

    print("=" * 115)
    print(
        f"[*] UrbanPulse Multi-Year Diagnostic & Radiometry Analysis: {city.upper()} ({start_year} - {end_year})"
    )
    print("=" * 115)

    # 1. Load classified maps and compute stable mask (Water or Built-up in ALL years)
    classes_dict = {}
    for y in years:
        class_file = data_path / f"{city}_{y}_classified.tif"
        if not class_file.exists():
            raise FileNotFoundError(f"Missing classified map: {class_file.resolve()}")
        with rasterio.open(class_file) as src:
            classes_dict[y] = src.read(1)

    # Stable mask: pixel is Built-up (1) or Water (3) in ALL years
    first_year = years[0]
    stable_mask = np.ones_like(classes_dict[first_year], dtype=bool)
    for y in years:
        c_arr = classes_dict[y]
        stable_mask &= (c_arr == 1) | (c_arr == 3)

    stable_pixel_count = int(np.sum(stable_mask))
    total_pixels = stable_mask.size
    print(f"[+] Computed Stable Pixels Mask (Built-up [1] or Water [3] in ALL {len(years)} years):")
    print(
        f"    - Stable Pixels Count : {stable_pixel_count:,} / {total_pixels:,} ({stable_pixel_count / total_pixels * 100:.2f}%)"
    )

    # 2. Collect statistics per year
    diagnostics = []

    for y in years:
        # A. Query STAC for scene info
        scene_count, scene_dates, mean_cloud = query_stac_scenes_for_year(year=y, config=config)
        dates_str = ", ".join(scene_dates) if scene_dates else "N/A"

        # B. Load band GeoTIFFs
        red_p = data_path / f"{city}_{y}_red.tif"
        nir_p = data_path / f"{city}_{y}_nir.tif"
        ndvi_p = data_path / f"{city}_{y}_ndvi.tif"
        ndbi_p = data_path / f"{city}_{y}_ndbi.tif"
        mndwi_p = data_path / f"{city}_{y}_mndwi.tif"

        with rasterio.open(red_p) as src_r:
            red = src_r.read(1).astype(np.float32)
        with rasterio.open(nir_p) as src_n:
            nir = src_n.read(1).astype(np.float32)
        with rasterio.open(ndvi_p) as src_vi:
            ndvi = src_vi.read(1).astype(np.float32)
        with rasterio.open(ndbi_p) as src_bi:
            ndbi = src_bi.read(1).astype(np.float32)
        with rasterio.open(mndwi_p) as src_wi:
            mndwi = src_wi.read(1).astype(np.float32)

        # C. Nodata calculation
        nodata_mask = np.isnan(red) | (red == -9999.0) | (red <= 0.0)
        nodata_pct = float(np.sum(nodata_mask) / red.size * 100.0)

        # D. Statistics over stable pixels
        valid_stable = (
            stable_mask
            & ~nodata_mask
            & np.isfinite(ndvi)
            & np.isfinite(ndbi)
            & np.isfinite(mndwi)
            & np.isfinite(red)
            & np.isfinite(nir)
        )

        mean_ndvi = float(np.mean(ndvi[valid_stable])) if np.any(valid_stable) else 0.0
        mean_ndbi = float(np.mean(ndbi[valid_stable])) if np.any(valid_stable) else 0.0
        mean_mndwi = float(np.mean(mndwi[valid_stable])) if np.any(valid_stable) else 0.0
        mean_red = float(np.mean(red[valid_stable])) if np.any(valid_stable) else 0.0
        mean_nir = float(np.mean(nir[valid_stable])) if np.any(valid_stable) else 0.0

        diagnostics.append(
            {
                "year": y,
                "scenes_used": scene_count,
                "scene_dates": dates_str,
                "nodata_pct": nodata_pct,
                "mean_cloud_pct": mean_cloud,
                "mean_ndvi": mean_ndvi,
                "mean_ndbi": mean_ndbi,
                "mean_mndwi": mean_mndwi,
                "mean_red": mean_red,
                "mean_nir": mean_nir,
            }
        )

    # 3. Print formatted table
    print("\n" + "=" * 125)
    print(
        f"{'Year':<5} | {'Scenes':<6} | {'Nodata %':<8} | {'Cloud %':<7} | {'Mean NDVI':<9} | {'Mean NDBI':<9} | {'Mean MNDWI':<10} | {'Mean Red':<8} | {'Mean NIR':<8} | {'Scene Acquisition Dates'}"
    )
    print("-" * 125)

    for row in diagnostics:
        print(
            f"{row['year']:<5} | "
            f"{row['scenes_used']:<6} | "
            f"{row['nodata_pct']:<7.2f}% | "
            f"{row['mean_cloud_pct']:<6.2f}% | "
            f"{row['mean_ndvi']:<9.4f} | "
            f"{row['mean_ndbi']:<9.4f} | "
            f"{row['mean_mndwi']:<10.4f} | "
            f"{row['mean_red']:<8.4f} | "
            f"{row['mean_nir']:<8.4f} | "
            f"{row['scene_dates']}"
        )

    print("=" * 125)

    # 4. Generate side-by-side RGB comparison for 2018, 2019, 2020
    generate_rgb_comparison_preview(
        city=city,
        years=(2018, 2019, 2020),
        data_dir=data_dir,
        output_path=preview_output,
    )

    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose annual Sentinel-2 composites and radiometry over stable pixels."
    )
    parser.add_argument(
        "--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)"
    )
    parser.add_argument("--start-year", type=int, default=2018, help="Start year (default: 2018)")
    parser.add_argument("--end-year", type=int, default=2024, help="End year (default: 2024)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ahmedabad.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Data directory (default: data)"
    )
    parser.add_argument(
        "--preview-out",
        type=str,
        default=None,
        help="Output path for 2018-2020 RGB comparison preview PNG",
    )
    args = parser.parse_args()

    diagnose_years(
        city=args.city,
        start_year=args.start_year,
        end_year=args.end_year,
        config_path=args.config,
        data_dir=args.data_dir,
        preview_output=args.preview_out,
    )


if __name__ == "__main__":
    main()
