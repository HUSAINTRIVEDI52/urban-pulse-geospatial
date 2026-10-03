"""
UrbanPulse - Land Cover Change Detection & Urban Expansion Analytics
Computes pixel-level transition matrices, change trajectories, gross/net built-up gains,
filters spatial noise with connected-component analysis, and renders change maps.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.train_classifier import PROJECT_CLASS_NAMES


def compute_transition_matrix(
    start_arr: np.ndarray,
    end_arr: np.ndarray,
    num_classes: int = 5,
    pixel_area_km2: float = 1.0,
    nodata_val: int = 0,
) -> np.ndarray:
    """
    Computes a transition matrix (rows = from_class, columns = to_class) in km2.
    Excludes any pixels where start or end raster equals nodata_val.
    """
    valid_mask = (
        (start_arr != nodata_val)
        & (end_arr != nodata_val)
        & np.isfinite(start_arr)
        & np.isfinite(end_arr)
    )
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)

    for i in range(1, num_classes + 1):
        for j in range(1, num_classes + 1):
            px_count = np.sum(valid_mask & (start_arr == i) & (end_arr == j))
            matrix[i - 1, j - 1] = px_count * pixel_area_km2

    return matrix


def detect_changes(
    city: str = "ahmedabad",
    start_year: int = 2018,
    end_year: int = 2024,
    min_patch_size: int = 3,
    data_dir: str | Path = "data",
    use_raw: bool = False,
    output_png_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Performs land cover change detection between start_year and end_year.

    Args:
        city: Target city name (e.g. 'ahmedabad').
        start_year: Baseline year.
        end_year: Comparison/target year.
        min_patch_size: Minimum connected component pixel count to retain change.
        data_dir: Directory containing classified GeoTIFFs.
        use_raw: If True, uses raw classified rasters instead of clean/ directory.
        output_png_path: Optional custom destination for the change visualization map.

    Returns:
        Dictionary of summary statistics, transition matrix DataFrame, and output file paths.
    """
    city_key = city.lower()
    data_path = Path(data_dir)

    def resolve_classified_path(yr: int) -> Path:
        clean_candidates = [
            data_path / city_key / "clean" / f"{city_key}_{yr}_classified.tif",
            data_path / "clean" / f"{city_key}_{yr}_classified.tif",
            data_path / city_key / "clean" / f"{yr}_classified.tif",
        ]
        raw_candidates = [
            data_path / f"{city_key}_{yr}_classified.tif",
            data_path / city_key / f"{city_key}_{yr}_classified.tif",
            data_path / city_key / f"{yr}_classified.tif",
        ]
        if not use_raw:
            for c in clean_candidates:
                if c.exists():
                    return c
        for r in raw_candidates:
            if r.exists():
                return r
        # Fallback to default expected path for clear error reporting
        return raw_candidates[0] if use_raw else clean_candidates[0]

    start_raster_path = resolve_classified_path(start_year)
    end_raster_path = resolve_classified_path(end_year)

    print("=" * 80)
    mode_str = "RAW" if use_raw else "CLEAN (Temporally Consistent)"
    print(
        f" URBANPULSE LAND COVER CHANGE DETECTION [{mode_str}]: {city.upper()} ({start_year} -> {end_year})"
    )
    print(f" Start Raster : {start_raster_path.resolve()}")
    print(f" End Raster   : {end_raster_path.resolve()}")
    print(f" Min Patch    : {min_patch_size} pixels")
    print("=" * 80)

    # 1. Verify existence of rasters
    if not start_raster_path.exists():
        raise FileNotFoundError(
            f"Classified raster for {start_year} not found at {start_raster_path.resolve()}."
        )
    if not end_raster_path.exists():
        raise FileNotFoundError(
            f"Classified raster for {end_year} not found at {end_raster_path.resolve()}."
        )

    # 2. Load rasters and validate CRS, shape, transform
    with rasterio.open(start_raster_path) as src_start:
        start_arr = src_start.read(1)
        start_profile = src_start.profile.copy()
        start_shape = src_start.shape
        start_crs = src_start.crs
        start_transform = src_start.transform

    with rasterio.open(end_raster_path) as src_end:
        end_arr = src_end.read(1)
        end_shape = src_end.shape
        end_crs = src_end.crs
        end_transform = src_end.transform

    if start_shape != end_shape:
        raise ValueError(
            f"Raster shape mismatch: {start_year} has shape {start_shape} while "
            f"{end_year} has shape {end_shape}."
        )
    if start_crs != end_crs:
        raise ValueError(
            f"Raster CRS mismatch: {start_year} is {start_crs} while {end_year} is {end_crs}."
        )
    if start_transform != end_transform:
        raise ValueError(
            f"Raster geotransform mismatch: {start_year} transform != {end_year} transform."
        )

    # 3. Mask pixels that are nodata (0) in either year
    valid_mask = (start_arr > 0) & (end_arr > 0) & np.isfinite(start_arr) & np.isfinite(end_arr)
    pixel_res_x = abs(start_transform.a)
    pixel_res_y = abs(start_transform.e)
    pixel_area_km2 = (pixel_res_x * pixel_res_y) / 1e6

    total_valid_pixels = int(np.sum(valid_mask))
    total_aoi_km2 = total_valid_pixels * pixel_area_km2

    print(
        f"\n[+] Aligned Rasters Verified: {start_shape[0]} x {start_shape[1]} pixels ({pixel_res_x:.1f}m resolution)"
    )
    print(f"[+] Total Valid Analyzed Area: {total_aoi_km2:.2f} km² ({total_valid_pixels:,} pixels)")

    # 4. Filter small change clusters using connected-component labelling
    if min_patch_size > 1:
        raw_changes = valid_mask & (start_arr != end_arr)

        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labeled_arr, num_features = ndimage.label(raw_changes, structure=structure)

        if num_features > 0:
            component_sizes = np.bincount(labeled_arr.ravel())
            small_comp_mask = np.isin(
                labeled_arr,
                np.where(
                    (component_sizes < min_patch_size) & (np.arange(len(component_sizes)) > 0)
                )[0],
            )
            # Revert isolated small change patches back to start class
            filtered_end_arr = np.where(small_comp_mask, start_arr, end_arr)
            removed_pixels = int(np.sum(small_comp_mask))
            print(
                f"[+] Filtered {removed_pixels:,} isolated change pixels (< {min_patch_size} connected pixels) "
                f"across {num_features:,} raw change components."
            )
        else:
            filtered_end_arr = end_arr.copy()
    else:
        filtered_end_arr = end_arr.copy()

    # 5. Compute transition matrix (rows = start class, cols = end class) in km2
    class_ids = [1, 2, 3, 4, 5]
    class_names = [PROJECT_CLASS_NAMES[cid] for cid in class_ids]
    transition_matrix_km2 = compute_transition_matrix(
        start_arr=start_arr,
        end_arr=filtered_end_arr,
        num_classes=5,
        pixel_area_km2=pixel_area_km2,
        nodata_val=0,
    )

    # Transition DataFrame
    df_trans = pd.DataFrame(
        transition_matrix_km2,
        index=[f"{c} ({start_year})" for c in class_names],
        columns=[f"{c} ({end_year})" for c in class_names],
    )

    csv_out_path = data_path / f"{city_key}_transition_{start_year}_{end_year}.csv"
    df_trans.to_csv(csv_out_path)

    print("\n" + "=" * 80)
    print(f"[*] TRANSITION MATRIX: {start_year} -> {end_year} (Area in km²)")
    print("=" * 80)
    col_from_to = "From \\ To"
    header_str = (
        f"{col_from_to:<22} | "
        + " | ".join([f"{c[:10]:>10}" for c in class_names])
        + " | {'Total':>10}"
    )
    print(header_str)
    print("-" * len(header_str))
    for i, c_start in enumerate(class_names):
        row_str = " | ".join([f"{transition_matrix_km2[i, j]:>10.2f}" for j in range(5)])
        row_total = transition_matrix_km2[i, :].sum()
        print(f"{c_start:<22} | {row_str} | {row_total:>10.2f}")
    print("-" * len(header_str))
    col_totals = " | ".join([f"{transition_matrix_km2[:, j].sum():>10.2f}" for j in range(5)])
    print(f"{'Total':<22} | {col_totals} | {transition_matrix_km2.sum():>10.2f}")
    print("=" * 80 + "\n")

    # 6. Save change raster: pixel = from_class * 10 + to_class
    change_raster_path = data_path / f"{city_key}_change_{start_year}_{end_year}.tif"
    change_arr = np.zeros_like(start_arr, dtype=np.uint8)
    change_arr[valid_mask] = (start_arr[valid_mask] * 10 + filtered_end_arr[valid_mask]).astype(
        np.uint8
    )

    start_profile.pop("blockxsize", None)
    start_profile.pop("blockysize", None)
    start_profile.pop("tiled", None)
    start_profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "lzw",
        }
    )
    with rasterio.open(change_raster_path, "w", **start_profile) as dst:
        dst.write(change_arr, 1)

    print(f"[+] Saved change raster to: {change_raster_path.name}")

    # 7. Summary metrics calculation
    start_builtup_km2 = float(np.sum(valid_mask & (start_arr == 1)) * pixel_area_km2)
    end_builtup_km2 = float(np.sum(valid_mask & (filtered_end_arr == 1)) * pixel_area_km2)

    gross_gain_km2 = float(
        np.sum(valid_mask & (start_arr != 1) & (filtered_end_arr == 1)) * pixel_area_km2
    )
    gross_loss_km2 = float(
        np.sum(valid_mask & (start_arr == 1) & (filtered_end_arr != 1)) * pixel_area_km2
    )
    net_change_km2 = end_builtup_km2 - start_builtup_km2
    pct_change = (net_change_km2 / start_builtup_km2 * 100.0) if start_builtup_km2 > 0 else 0.0

    # Sources of Built-up Gain
    gain_veg_km2 = float(
        np.sum(valid_mask & (start_arr == 2) & (filtered_end_arr == 1)) * pixel_area_km2
    )
    gain_agri_km2 = float(
        np.sum(valid_mask & (start_arr == 4) & (filtered_end_arr == 1)) * pixel_area_km2
    )
    gain_open_km2 = float(
        np.sum(valid_mask & (start_arr == 5) & (filtered_end_arr == 1)) * pixel_area_km2
    )
    gain_water_km2 = float(
        np.sum(valid_mask & (start_arr == 3) & (filtered_end_arr == 1)) * pixel_area_km2
    )

    share_veg = (gain_veg_km2 / gross_gain_km2 * 100.0) if gross_gain_km2 > 0 else 0.0
    share_agri = (gain_agri_km2 / gross_gain_km2 * 100.0) if gross_gain_km2 > 0 else 0.0
    share_open = (gain_open_km2 / gross_gain_km2 * 100.0) if gross_gain_km2 > 0 else 0.0
    share_water = (gain_water_km2 / gross_gain_km2 * 100.0) if gross_gain_km2 > 0 else 0.0

    print("=" * 80)
    print(f"[*] URBAN EXPANSION SUMMARY: {city.capitalize()} ({start_year} -> {end_year})")
    print("=" * 80)
    print(f"  - Total Built-up ({start_year})           : {start_builtup_km2:>8.2f} km²")
    print(f"  - Total Built-up ({end_year})             : {end_builtup_km2:>8.2f} km²")
    print(
        f"  - Gross Built-up Gain              : +{gross_gain_km2:>7.2f} km² (Non-built-up -> Built-up)"
    )
    print(
        f"  - Gross Built-up Loss              : -{gross_loss_km2:>7.2f} km² (Built-up -> Non-built-up)"
    )
    print(
        f"  - Net Built-up Change              : {net_change_km2:>+8.2f} km² ({pct_change:>+6.2f}%)"
    )
    print("-" * 80)
    print("  - Sources of New Built-up Land:")
    print(
        f"      * Agriculture -> Built-up      : {gain_agri_km2:>7.2f} km² ({share_agri:>5.1f}% of total gain)"
    )
    print(
        f"      * Vegetation  -> Built-up      : {gain_veg_km2:>7.2f} km² ({share_veg:>5.1f}% of total gain)"
    )
    print(
        f"      * Open Land   -> Built-up      : {gain_open_km2:>7.2f} km² ({share_open:>5.1f}% of total gain)"
    )
    if gain_water_km2 > 0:
        print(
            f"      * Water       -> Built-up      : {gain_water_km2:>7.2f} km² ({share_water:>5.1f}% of total gain)"
        )
    print("=" * 80 + "\n")

    # 8. Generate Map: Vegetation->Built-up, Agriculture->Built-up, Open land->Built-up, Built-up->Loss
    map_path = (
        Path(output_png_path)
        if output_png_path
        else data_path / f"change_{start_year}_{end_year}.png"
    )

    # Categories:
    # 0: Background / NoData / Unchanged / other changes (Light Grey)
    # 1: Vegetation -> Built-up (21) -> Dark green / Forest-to-Red (#2e7d32)
    # 2: Agriculture -> Built-up (41) -> Orange (#ea580c)
    # 3: Open land -> Built-up (51) -> Purple (#8b5cf6)
    # 4: Built-up -> Loss (12, 13, 14, 15) -> Black (#111827)
    vis_arr = np.zeros_like(change_arr, dtype=np.uint8)

    # Set light grey for all valid analyzed pixels
    vis_arr[valid_mask] = 0

    # Categorical assignments
    vis_arr[valid_mask & (change_arr == 21)] = 1  # Veg -> Built
    vis_arr[valid_mask & (change_arr == 41)] = 2  # Agri -> Built
    vis_arr[valid_mask & (change_arr == 51)] = 3  # Open -> Built
    vis_arr[valid_mask & np.isin(change_arr, [12, 13, 14, 15])] = 4  # Built -> Loss

    # Colors: [Unchanged / Other, Veg->Built, Agri->Built, Open->Built, Built->Loss]
    cmap_colors = ["#e2e8f0", "#2e7d32", "#ea580c", "#8b5cf6", "#0f172a"]
    cmap = ListedColormap(cmap_colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 14), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#cbd5e1")

    ax.imshow(vis_arr, cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_title(
        f"{city.capitalize()} Urban Land Cover Change ({start_year} – {end_year})\n"
        f"Gross Built-up Gain: +{gross_gain_km2:.1f} km² | Net Built-up Growth: {net_change_km2:+.1f} km² ({pct_change:+.1f}%)",
        fontsize=13,
        fontweight="bold",
        pad=14,
        color="#0f172a",
    )
    ax.axis("off")

    # Legend Patches
    legend_patches = [
        mpatches.Patch(
            facecolor="#2e7d32",
            edgecolor="#1b5e20",
            label=f"Vegetation -> Built-up (+{gain_veg_km2:.1f} km², {share_veg:.1f}%)",
        ),
        mpatches.Patch(
            facecolor="#ea580c",
            edgecolor="#c2410c",
            label=f"Agriculture -> Built-up (+{gain_agri_km2:.1f} km², {share_agri:.1f}%)",
        ),
        mpatches.Patch(
            facecolor="#8b5cf6",
            edgecolor="#6d28d9",
            label=f"Open land -> Built-up (+{gain_open_km2:.1f} km², {share_open:.1f}%)",
        ),
        mpatches.Patch(
            facecolor="#0f172a",
            edgecolor="#000000",
            label=f"Built-up Loss -> Other (-{gross_loss_km2:.1f} km²)",
        ),
        mpatches.Patch(facecolor="#e2e8f0", edgecolor="#94a3b8", label="Unchanged / Other Classes"),
    ]

    ax.legend(
        handles=legend_patches,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        title="Urban Growth Trajectory",
        title_fontsize=11,
        fontsize=9.5,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cbd5e1",
        framealpha=0.95,
        fancybox=True,
        shadow=True,
    )

    plt.tight_layout()
    plt.savefig(map_path, bbox_inches="tight", dpi=200, facecolor=fig.get_facecolor())
    plt.close()

    print(f"[+] Saved change map to: {map_path.resolve()}")

    return {
        "city": city,
        "start_year": start_year,
        "end_year": end_year,
        "start_builtup_km2": start_builtup_km2,
        "end_builtup_km2": end_builtup_km2,
        "gross_gain_km2": gross_gain_km2,
        "gross_loss_km2": gross_loss_km2,
        "net_change_km2": net_change_km2,
        "pct_change": pct_change,
        "share_veg": share_veg,
        "share_agri": share_agri,
        "share_open": share_open,
        "transition_matrix_df": df_trans,
        "change_raster": change_raster_path,
        "change_map_png": map_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UrbanPulse Land Cover Change Detection & Urban Expansion Analytics."
    )
    parser.add_argument(
        "--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)"
    )
    parser.add_argument("--start", type=int, default=2018, help="Start year (default: 2018)")
    parser.add_argument("--end", type=int, default=2024, help="End year (default: 2024)")
    parser.add_argument(
        "--min-patch", type=int, default=3, help="Minimum patch size in pixels (default: 3)"
    )
    parser.add_argument(
        "--raw", action="store_true", help="Use raw uncleaned classifications instead of clean/"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for data rasters")
    parser.add_argument("--out-map", type=str, default=None, help="Custom path for output PNG map")

    args = parser.parse_args()
    detect_changes(
        city=args.city,
        start_year=args.start,
        end_year=args.end,
        min_patch_size=args.min_patch,
        data_dir=args.data_dir,
        use_raw=args.raw,
        output_png_path=args.out_map,
    )
