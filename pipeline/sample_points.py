"""
UrbanPulse - Stratified Spatial Block Point Sampling Module
Generates balanced training and testing ground truth points from ESA WorldCover labels,
excluding mixed boundary pixels via morphological erosion, and partitioning samples
spatially (70% train / 30% test) to prevent spatial autocorrelation leakage.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import yaml
from scipy.ndimage import binary_erosion
from shapely.geometry import Point
from sklearn.model_selection import GroupShuffleSplit

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def sample_training_points(
    labels_raster_path: str | Path = "data/ahmedabad_worldcover_labels.tif",
    samples_per_class: int = 300,
    train_ratio: float = 0.70,
    n_spatial_blocks: int = 8,
    random_seed: int = 42,
    output_train_path: str | Path = "data/train_points.geojson",
    output_test_path: str | Path = "data/test_points.geojson",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Samples pure interior pixels per class and performs spatial block train/test split.

    Args:
        labels_raster_path: Path to remapped WorldCover labels GeoTIFF.
        samples_per_class: Maximum number of sample points per class.
        train_ratio: Proportion of spatial blocks allocated to training set (default: 0.70).
        n_spatial_blocks: Number of spatial grid divisions along X and Y axes.
        random_seed: Reproducibility seed for sampling and spatial split.
        output_train_path: Destination path for train_points.geojson.
        output_test_path: Destination path for test_points.geojson.

    Returns:
        Tuple of (train_gdf, test_gdf).
    """
    labels_path = Path(labels_raster_path)
    train_path = Path(output_train_path)
    test_path = Path(output_test_path)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels raster not found: {labels_path.resolve()}. "
            "Please run pipeline/get_training_labels.py first."
        )

    print("=" * 75)
    print("[*] UrbanPulse Ground Truth Point Sampler")
    print(f"    - Input Labels Raster : {labels_path.resolve()}")
    print(f"    - Target per Class    : up to {samples_per_class} points")
    print(
        f"    - Split Strategy      : Spatial Block Split ({train_ratio*100:.0f}% Train / {(1-train_ratio)*100:.0f}% Test)"
    )
    print(f"    - Spatial Grid Size   : {n_spatial_blocks} x {n_spatial_blocks} blocks")
    print("=" * 75)

    # 1. Load Labels Raster
    print("\n[Step 1/4] Loading labels raster...")
    with rasterio.open(labels_path) as src:
        labels = src.read(1)
        transform = src.transform
        raster_crs = src.crs

    print(f"[+] Loaded raster shape: {labels.shape} in {raster_crs}")

    # 2. Stratified Sampling with Boundary Exclusion
    print("\n[Step 2/4] Applying morphological erosion (3x3) to exclude mixed boundary pixels...")
    rng = np.random.default_rng(random_seed)
    erosion_struct = np.ones((3, 3), dtype=bool)

    sampled_records = []

    for cid in range(1, 6):
        cname = PROJECT_CLASS_NAMES[cid]
        class_mask = labels == cid
        total_pixels = int(np.sum(class_mask))

        # Binary erosion excludes any pixel adjacent to a different class or nodata
        pure_mask = binary_erosion(class_mask, structure=erosion_struct)
        pure_pixels = int(np.sum(pure_mask))

        rows, cols = np.where(pure_mask)
        n_available = len(rows)

        if n_available == 0:
            print(f"[!] Warning: No pure interior pixels found for Class {cid} ({cname})")
            continue

        n_sample = min(samples_per_class, n_available)
        chosen_indices = rng.choice(n_available, size=n_sample, replace=False)

        sample_rows = rows[chosen_indices]
        sample_cols = cols[chosen_indices]
        xs, ys = rasterio.transform.xy(transform, sample_rows, sample_cols)

        for x, y in zip(xs, ys):
            sampled_records.append(
                {
                    "class_id": int(cid),
                    "class_name": cname,
                    "x": float(x),
                    "y": float(y),
                }
            )

        pct_pure = (pure_pixels / total_pixels * 100.0) if total_pixels > 0 else 0.0
        print(
            f"    - Class {cid} ({cname:<12}): {total_pixels:>10,} total px | {pure_pixels:>10,} pure px ({pct_pure:>5.1f}%) | sampled {n_sample:>4} points"
        )

    gdf_utm = gpd.GeoDataFrame(
        sampled_records,
        geometry=[Point(r["x"], r["y"]) for r in sampled_records],
        crs=raster_crs,
    )
    print(f"\n[+] Collected {len(gdf_utm)} total pure ground truth points.")

    # 3. Spatial Block Partitioning
    print(
        "\n[Step 3/4] Partitioning points into spatial blocks to eliminate spatial data leakage..."
    )
    minx, miny, maxx, maxy = gdf_utm.total_bounds
    dx = (maxx - minx + 1e-6) / n_spatial_blocks
    dy = (maxy - miny + 1e-6) / n_spatial_blocks

    block_x = np.clip(np.floor((gdf_utm["x"] - minx) / dx), 0, n_spatial_blocks - 1).astype(int)
    block_y = np.clip(np.floor((gdf_utm["y"] - miny) / dy), 0, n_spatial_blocks - 1).astype(int)
    gdf_utm["block_id"] = block_y * n_spatial_blocks + block_x

    n_active_blocks = gdf_utm["block_id"].nunique()
    print(f"[*] Points distributed across {n_active_blocks} active spatial blocks.")

    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_seed)
    train_idx, test_idx = next(gss.split(gdf_utm, groups=gdf_utm["block_id"]))

    train_gdf_utm = gdf_utm.iloc[train_idx].copy()
    test_gdf_utm = gdf_utm.iloc[test_idx].copy()

    train_gdf_utm["split"] = "train"
    test_gdf_utm["split"] = "test"

    # Transform to WGS84 (EPSG:4326) for standard GeoJSON storage
    train_gdf = train_gdf_utm.to_crs("EPSG:4326")[["class_id", "class_name", "split", "geometry"]]
    test_gdf = test_gdf_utm.to_crs("EPSG:4326")[["class_id", "class_name", "split", "geometry"]]

    # 4. Save GeoJSON Files
    print("\n[Step 4/4] Saving GeoJSON files...")
    train_gdf.to_file(train_path, driver="GeoJSON")
    print(f"[+] Saved training dataset : {train_path.resolve()} ({len(train_gdf)} points)")

    test_gdf.to_file(test_path, driver="GeoJSON")
    print(f"[+] Saved testing dataset  : {test_path.resolve()} ({len(test_gdf)} points)")

    # 5. Print Distribution Table
    print("\n" + "=" * 75)
    print("[*] Sample Points Distribution Summary (Train vs Test)")
    print("=" * 75)
    print(
        f"{'Class ID':<9} | {'Class Name':<15} | {'Train Pts':<11} | {'Test Pts':<10} | {'Total Pts':<11} | {'Train %'}"
    )
    print("-" * 75)

    for cid in range(1, 6):
        cname = PROJECT_CLASS_NAMES[cid]
        tr_c = int(np.sum(train_gdf["class_id"] == cid))
        te_c = int(np.sum(test_gdf["class_id"] == cid))
        tot_c = tr_c + te_c
        tr_pct = (tr_c / tot_c * 100.0) if tot_c > 0 else 0.0
        print(
            f"{cid:<9} | {cname:<15} | {tr_c:>11,} | {te_c:>10,} | {tot_c:>11,} | {tr_pct:>6.1f}%"
        )

    print("-" * 75)
    tot_tr = len(train_gdf)
    tot_te = len(test_gdf)
    total_pts = tot_tr + tot_te
    overall_tr_pct = (tot_tr / total_pts * 100.0) if total_pts > 0 else 0.0
    print(
        f"{'Total':<9} | {'All Classes':<15} | {tot_tr:>11,} | {tot_te:>10,} | {total_pts:>11,} | {overall_tr_pct:>6.1f}%"
    )
    print("=" * 75 + "\n")

    return train_gdf, test_gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified spatial block point sampling.")
    parser.add_argument(
        "--labels", type=str, default="data/ahmedabad_worldcover_labels.tif", help="Labels raster"
    )
    parser.add_argument(
        "--samples", type=int, default=300, help="Max sample points per class (default: 300)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70, help="Train split ratio (default: 0.70)"
    )
    parser.add_argument(
        "--blocks", type=int, default=8, help="Spatial grid block divisions (default: 8)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--train-out", type=str, default="data/train_points.geojson", help="Train GeoJSON output"
    )
    parser.add_argument(
        "--test-out", type=str, default="data/test_points.geojson", help="Test GeoJSON output"
    )

    args = parser.parse_args()
    sample_training_points(
        labels_raster_path=args.labels,
        samples_per_class=args.samples,
        train_ratio=args.train_ratio,
        n_spatial_blocks=args.blocks,
        random_seed=args.seed,
        output_train_path=args.train_out,
        output_test_path=args.test_out,
    )
