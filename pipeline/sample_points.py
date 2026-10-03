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
import pandas as pd
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


def load_config(city: str = "ahmedabad", config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads city YAML configuration."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_training_points(
    city: str = "ahmedabad",
    config_path: str | Path | None = None,
    labels_raster_path: str | Path | None = None,
    samples_per_class: int = 300,
    train_ratio: float = 0.70,
    n_spatial_blocks: int = 8,
    random_seed: int = 42,
    train_years: list[int] | None = None,
    data_dir: str | Path = "data",
    output_train_path: str | Path | None = None,
    output_test_path: str | Path | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Samples pure interior pixels per class from ESA WorldCover labels, partitions
    them into spatial blocks to avoid spatial autocorrelation, extracts 8 spectral
    features across multiple composite years (train_years), and pools them with a 'year' column.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    city_subpath = data_path / city_key
    city_subpath.mkdir(parents=True, exist_ok=True)

    if train_years is None:
        cfg = load_config(city=city_key, config_path=config_path)
        ay = cfg.get("temporal", {}).get("analysis_years", {})
        sy = ay.get("start_year", 2020)
        ey = ay.get("end_year", 2024)
        my = (sy + ey) // 2
        train_years = [sy, my, ey] if sy != ey else [sy]

    if labels_raster_path is None:
        labels_path = data_path / f"{city_key}_worldcover_labels.tif"
    else:
        labels_path = Path(labels_raster_path)

    if output_train_path is None:
        train_path = data_path / f"{city_key}_train_points.geojson"
    else:
        train_path = Path(output_train_path)

    if output_test_path is None:
        test_path = data_path / f"{city_key}_test_points.geojson"
    else:
        test_path = Path(output_test_path)

    train_pooled_path = data_path / f"{city_key}_train_points_pooled.geojson"
    test_pooled_path = data_path / f"{city_key}_test_points_pooled.geojson"

    if not labels_path.exists():
        from pipeline.get_training_labels import extract_worldcover_labels

        extract_worldcover_labels(city=city_key, output_labels_path=labels_path)

    print("=" * 78)
    print(f"[*] UrbanPulse Multi-Year Point Sampler: {city.capitalize()}")
    print(f"    - Input Labels Raster : {labels_path.resolve()}")
    print(f"    - Target per Class    : up to {samples_per_class} points")
    print(
        f"    - Split Strategy      : Spatial Block Split ({train_ratio*100:.0f}% Train / {(1-train_ratio)*100:.0f}% Test)"
    )
    print(f"    - Spatial Grid Size   : {n_spatial_blocks} x {n_spatial_blocks} blocks")
    print(f"    - Training Years      : {train_years}")
    print("=" * 78)

    # 1. Load Labels Raster
    print("\n[Step 1/5] Loading labels raster...")
    with rasterio.open(labels_path) as src:
        labels = src.read(1)
        transform = src.transform
        raster_crs = src.crs

    print(f"[+] Loaded raster shape: {labels.shape} in {raster_crs}")

    # 2. Stratified Sampling with Boundary Exclusion
    print("\n[Step 2/5] Applying morphological erosion (3x3) to exclude mixed boundary pixels...")
    rng = np.random.default_rng(random_seed)
    structure_3x3 = np.ones((3, 3), dtype=bool)

    sampled_records = []
    class_counts = {}

    for cid in range(1, 6):
        class_name = PROJECT_CLASS_NAMES[cid]
        class_mask = labels == cid
        n_raw_pixels = int(np.sum(class_mask))

        if n_raw_pixels == 0:
            print(f"[!] Warning: Class {cid} ({class_name}) has 0 pixels in the scene.")
            continue

        eroded_mask = binary_erosion(class_mask, structure=structure_3x3)
        n_eroded_pixels = int(np.sum(eroded_mask))

        chosen_mask = eroded_mask if n_eroded_pixels >= 50 else class_mask

        rows, cols = np.where(chosen_mask)
        n_available = len(rows)

        n_samples = min(samples_per_class, n_available)
        if n_samples == 0:
            continue

        selected_indices = rng.choice(n_available, size=n_samples, replace=False)
        sample_rows = rows[selected_indices]
        sample_cols = cols[selected_indices]

        xs, ys = rasterio.transform.xy(transform, sample_rows, sample_cols)

        for i, (x, y) in enumerate(zip(xs, ys)):
            sampled_records.append(
                {
                    "class_id": cid,
                    "class_name": class_name,
                    "geometry": Point(x, y),
                    "raster_row": int(sample_rows[i]),
                    "raster_col": int(sample_cols[i]),
                }
            )

        class_counts[class_name] = n_samples
        print(
            f"    - Class {cid} ({class_name:<12}): Sampled {n_samples:>4} points (from {n_available:>6} interior pixels)"
        )

    if not sampled_records:
        raise ValueError("Failed to extract any valid sample points from the labels raster.")

    gdf = gpd.GeoDataFrame(sampled_records, crs=raster_crs)

    # 3. Spatial Block Partitioning
    print(
        "\n[Step 3/5] Partitioning points into spatial blocks to avoid spatial autocorrelation..."
    )
    minx, miny, maxx, maxy = gdf.total_bounds
    x_bins = np.linspace(minx, maxx, n_spatial_blocks + 1)
    y_bins = np.linspace(miny, maxy, n_spatial_blocks + 1)

    block_x = np.clip(np.digitize(gdf.geometry.x, x_bins) - 1, 0, n_spatial_blocks - 1)
    block_y = np.clip(np.digitize(gdf.geometry.y, y_bins) - 1, 0, n_spatial_blocks - 1)
    gdf["block_id"] = block_x * n_spatial_blocks + block_y

    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_seed)
    train_idx, test_idx = next(gss.split(gdf, groups=gdf["block_id"]))

    train_gdf = gdf.iloc[train_idx].copy().reset_index(drop=True)
    test_gdf = gdf.iloc[test_idx].copy().reset_index(drop=True)

    print(f"[+] Base Unique Sample Locations: {len(gdf)}")
    print(f"    - Training Set: {len(train_gdf)} points ({len(train_gdf)/len(gdf)*100:.1f}%)")
    print(f"    - Testing Set : {len(test_gdf)} points ({len(test_gdf)/len(gdf)*100:.1f}%)")

    # 4. Multi-Year Feature Extraction & Pooling
    print(f"\n[Step 4/5] Extracting 8-band spectral features for years: {train_years}...")
    feature_names = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]

    def extract_yearly_features(points_gdf: gpd.GeoDataFrame, yr: int) -> gpd.GeoDataFrame:
        feat_paths = [data_path / f"{city_key}_{yr}_{feat}.tif" for feat in feature_names]
        if not all(p.exists() for p in feat_paths):
            print(
                f"    [!] Warning: Missing some feature rasters for {city_key} {yr}. Skipping extraction for {yr}."
            )
            return gpd.GeoDataFrame(
                columns=list(points_gdf.columns) + feature_names + ["year"], crs=points_gdf.crs
            )

        pts_proj = points_gdf.to_crs(raster_crs)
        coords = [(geom.x, geom.y) for geom in pts_proj.geometry]

        band_data = {}
        for f_name, f_path in zip(feature_names, feat_paths):
            with rasterio.open(f_path) as src:
                sampled = [val[0] for val in src.sample(coords)]
                band_data[f_name] = sampled

        yearly_df = points_gdf.copy()
        yearly_df["year"] = yr
        for f_name in feature_names:
            yearly_df[f_name] = np.array(band_data[f_name], dtype=np.float32)

        # Filter out nodata / non-finite samples
        feat_matrix = yearly_df[feature_names].values
        valid_mask = np.all(np.isfinite(feat_matrix) & (feat_matrix != -9999.0), axis=1)
        valid_df = yearly_df[valid_mask].copy().reset_index(drop=True)
        return valid_df

    train_pooled_list = []
    test_pooled_list = []

    for yr in train_years:
        tr_yr = extract_yearly_features(train_gdf, yr)
        te_yr = extract_yearly_features(test_gdf, yr)
        if len(tr_yr) > 0:
            train_pooled_list.append(tr_yr)
            print(
                f"    - Year {yr}: Extracted {len(tr_yr)} valid train points, {len(te_yr)} valid test points"
            )
        if len(te_yr) > 0:
            test_pooled_list.append(te_yr)

    if train_pooled_list:
        train_pooled_gdf = gpd.GeoDataFrame(
            pd.concat(train_pooled_list, ignore_index=True), crs=train_gdf.crs
        )
    else:
        train_pooled_gdf = train_gdf.copy()

    if test_pooled_list:
        test_pooled_gdf = gpd.GeoDataFrame(
            pd.concat(test_pooled_list, ignore_index=True), crs=test_gdf.crs
        )
    else:
        test_pooled_gdf = test_gdf.copy()

    print(f"\n[+] Total Pooled Training Set : {len(train_pooled_gdf)} samples")
    print(f"[+] Total Pooled Testing Set  : {len(test_pooled_gdf)} samples")

    # 5. Save GeoJSON outputs
    print("\n[Step 5/5] Saving datasets to disk...")
    train_wgs84 = train_gdf.to_crs("EPSG:4326")
    test_wgs84 = test_gdf.to_crs("EPSG:4326")
    train_wgs84.to_file(train_path, driver="GeoJSON")
    test_wgs84.to_file(test_path, driver="GeoJSON")
    (city_subpath / "train_points.geojson").write_text(
        train_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (city_subpath / "test_points.geojson").write_text(
        test_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    if len(train_pooled_gdf) > 0:
        train_pooled_wgs84 = train_pooled_gdf.to_crs("EPSG:4326")
        test_pooled_wgs84 = test_pooled_gdf.to_crs("EPSG:4326")
        train_pooled_wgs84.to_file(train_pooled_path, driver="GeoJSON")
        test_pooled_wgs84.to_file(test_pooled_path, driver="GeoJSON")
        (city_subpath / "train_points_pooled.geojson").write_text(
            train_pooled_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (city_subpath / "test_points_pooled.geojson").write_text(
            test_pooled_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    print(f"[+] Saved Training Points: {train_path.resolve()}")
    print(f"[+] Saved Testing Points : {test_path.resolve()}")
    print(f"[+] Saved Pooled Training Points: {train_pooled_path.resolve()}")
    print(f"[+] Saved Pooled Testing Points : {test_pooled_path.resolve()}")

    return train_pooled_gdf if len(train_pooled_gdf) > 0 else train_gdf, (
        test_pooled_gdf if len(test_pooled_gdf) > 0 else test_gdf
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stratified spatial block point sampling with multi-year pooling."
    )
    parser.add_argument(
        "--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)"
    )
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        default=[2018, 2021, 2024],
        help="Training years to pool (default: 2018 2021 2024)",
    )
    parser.add_argument(
        "--labels", type=str, default=None, help="Path to WorldCover labels GeoTIFF"
    )
    parser.add_argument(
        "--samples-per-class", type=int, default=300, help="Target samples per class (default: 300)"
    )
    parser.add_argument(
        "--train-out", type=str, default=None, help="Output train points GeoJSON path"
    )
    parser.add_argument(
        "--test-out", type=str, default=None, help="Output test points GeoJSON path"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()
    sample_training_points(
        city=args.city,
        labels_raster_path=args.labels,
        samples_per_class=args.samples_per_class,
        train_years=args.train_years,
        data_dir=args.data_dir,
        output_train_path=args.train_out,
        output_test_path=args.test_out,
    )


if __name__ == "__main__":
    main()
