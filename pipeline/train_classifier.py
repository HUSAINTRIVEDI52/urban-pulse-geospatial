"""
UrbanPulse - Land Cover Random Forest Classifier & Mapping Module
Loads multi-spectral bands & indices, extracts spectral signatures for ground truth points,
trains a Random Forest classifier (200 trees), evaluates on spatial-block test points,
saves model to data/rf_model_{year}.pkl, classifies full AOI raster to
data/ahmedabad_{year}_classified.tif, and prints accuracy metrics and area statistics.
"""

import argparse
import sys
from pathlib import Path
from typing import Any
import numpy as np
import rasterio
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)
import joblib
import yaml

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

FEATURE_NAMES = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]


def load_config(config_path: str | Path = "configs/ahmedabad.yaml") -> dict[str, Any]:
    """Loads city YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_features_at_points(
    gdf: gpd.GeoDataFrame, feature_files: list[Path], raster_crs: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts multi-spectral feature values at point coordinates.
    Filters out any invalid/nodata samples.
    """
    gdf_proj = gdf.to_crs(raster_crs)
    coords = [(geom.x, geom.y) for geom in gdf_proj.geometry]

    feats = []
    for fpath in feature_files:
        with rasterio.open(fpath) as src:
            sampled_vals = [val[0] for val in src.sample(coords)]
            feats.append(sampled_vals)

    X = np.array(feats, dtype=np.float32).T
    y = gdf["class_id"].values.astype(np.int64)

    # Valid mask: all feature channels must be finite and not equal to nodata (-9999.0)
    valid_mask = np.all(np.isfinite(X) & (X != -9999.0), axis=1)

    return X[valid_mask], y[valid_mask], valid_mask


def train_and_evaluate_classifier(
    year: int = 2024,
    n_trees: int = 200,
    random_state: int = 42,
    data_dir: str | Path = "data",
    train_geojson: str | Path = "data/train_points.geojson",
    test_geojson: str | Path = "data/test_points.geojson",
    output_model_path: str | Path = "data/rf_model_2024.pkl",
    output_classified_path: str | Path = "data/ahmedabad_2024_classified.tif",
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """
    Trains Random Forest, evaluates on spatial test split, classifies full raster, and exports GeoTIFF.
    """
    data_path = Path(data_dir)
    train_path = Path(train_geojson)
    test_path = Path(test_geojson)
    model_path = Path(output_model_path)
    classified_path = Path(output_classified_path)

    print("=" * 78)
    print(f"[*] UrbanPulse Land Cover Classifier (Year: {year})")
    print(f"    - Algorithm         : Random Forest ({n_trees} estimators)")
    print(f"    - Input Features    : {FEATURE_NAMES}")
    print(f"    - Train Points      : {train_path.resolve()}")
    print(f"    - Test Points       : {test_path.resolve()}")
    print(f"    - Model Output      : {model_path.resolve()}")
    print(f"    - Classified Raster : {classified_path.resolve()}")
    print("=" * 78)

    # 1. Verify Feature Rasters
    feature_files = [data_path / f"ahmedabad_{year}_{feat}.tif" for feat in FEATURE_NAMES]
    missing_features = [f.name for f in feature_files if not f.exists()]
    if missing_features:
        raise FileNotFoundError(
            f"Missing required feature rasters in {data_path}: {missing_features}. "
            "Please run pipeline/build_composite.py and pipeline/compute_indices.py first."
        )

    # Read Reference Spatial Metadata
    with rasterio.open(feature_files[0]) as ref_src:
        profile = ref_src.profile.copy()
        raster_crs = ref_src.crs
        raster_transform = ref_src.transform
        raster_shape = ref_src.shape

    # 2. Extract Training and Testing Feature Matrices
    print("\n[Step 1/5] Extracting 8-channel spectral features at sample point locations...")
    train_gdf = gpd.read_file(train_path)
    test_gdf = gpd.read_file(test_path)

    X_train, y_train, _ = extract_features_at_points(train_gdf, feature_files, raster_crs)
    X_test, y_test, _ = extract_features_at_points(test_gdf, feature_files, raster_crs)

    print(f"[+] Clean Training Samples : {len(X_train):>5} points across 8 features")
    print(f"[+] Clean Testing Samples  : {len(X_test):>5} points across 8 features")

    # 3. Train Random Forest Classifier
    print(f"\n[Step 2/5] Training Random Forest model ({n_trees} trees, all CPU cores)...")
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    print("[+] Model training completed successfully.")

    # Save Model to Disk
    joblib.dump(rf, model_path)
    print(f"[+] Saved trained model to: {model_path.name}")

    # 4. Evaluate Model on Spatial Block Test Set
    print("\n[Step 3/5] Evaluating model performance on independent spatial test blocks...")
    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    target_names = [PROJECT_CLASS_NAMES[cid] for cid in range(1, 6)]
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])

    print("\n" + "=" * 78)
    print("[*] Model Evaluation Metrics (Spatial Block Validation)")
    print("=" * 78)
    print(f"Overall Accuracy : {acc * 100:.2f}%")
    print(f"Cohen's Kappa    : {kappa:.4f}")
    print("\nClassification Report:\n")
    print(report)

    print("Confusion Matrix (Rows: Ground Truth, Columns: Predicted):")
    header_str = f"{'True \\ Pred':<14} | " + " | ".join([f"{PROJECT_CLASS_NAMES[c][:8]:<8}" for c in range(1, 6)])
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    for i, cid in enumerate(range(1, 6)):
        row_vals = " | ".join([f"{cm[i, j]:>8}" for j in range(5)])
        print(f"{PROJECT_CLASS_NAMES[cid]:<14} | {row_vals}")
    print("-" * len(header_str))

    # Feature Importance
    print("\nFeature Importances:")
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank}. {FEATURE_NAMES[idx]:<8}: {importances[idx]*100:>5.2f}%")

    # 5. Classify Full 2D Feature Stack
    print(f"\n[Step 4/5] Classifying full raster stack for Ahmedabad ({raster_shape[0]} x {raster_shape[1]} pixels)...")
    feature_arrays = []
    for fpath in feature_files:
        with rasterio.open(fpath) as src:
            feature_arrays.append(src.read(1).astype(np.float32))

    # Stack into 2D feature matrix (H*W, 8)
    stack_2d = np.column_stack([arr.ravel() for arr in feature_arrays])

    # Valid pixel mask where all features are finite
    valid_raster_mask = np.all(np.isfinite(stack_2d) & (stack_2d != -9999.0), axis=1)
    print(f"[*] Total Grid Pixels: {len(stack_2d):,}")
    print(f"[*] Valid AOI Pixels : {int(np.sum(valid_raster_mask)):,} ({np.sum(valid_raster_mask)/len(stack_2d)*100:.1f}%)")

    # Predict valid pixels
    classified_flat = np.zeros(len(stack_2d), dtype=np.uint8)
    if np.any(valid_raster_mask):
        classified_flat[valid_raster_mask] = rf.predict(stack_2d[valid_raster_mask])

    classified_map = classified_flat.reshape(raster_shape)

    # 6. Save Classified GeoTIFF
    print(f"\n[Step 5/5] Saving classified GeoTIFF map to: {classified_path.name}...")
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "lzw",
            "tiled": True,
        }
    )

    with rasterio.open(classified_path, "w", **profile) as dst:
        dst.write(classified_map, 1)

    # 7. Area Summary Statistics
    pixel_res_x = abs(raster_transform.a)
    pixel_res_y = abs(raster_transform.e)
    pixel_area_km2 = (pixel_res_x * pixel_res_y) / 1e6

    valid_classified = classified_map > 0
    total_valid_px = int(np.sum(valid_classified))

    print("\n" + "=" * 78)
    print(f"[*] Ahmedabad {year} Final Land Cover Area Summary")
    print("=" * 78)
    print(
        f"{'Class ID':<9} | {'Class Name':<15} | {'Pixel Count':<14} | {'Area (km^2)':<12} | {'Percentage'}"
    )
    print("-" * 78)

    area_summary = {}
    for cid in range(1, 6):
        cname = PROJECT_CLASS_NAMES[cid]
        px_count = int(np.sum(classified_map == cid))
        area_km2 = px_count * pixel_area_km2
        pct = (px_count / total_valid_px * 100.0) if total_valid_px > 0 else 0.0
        area_summary[cname] = {
            "class_id": cid,
            "pixel_count": px_count,
            "area_km2": area_km2,
            "percentage": pct,
        }
        print(f"{cid:<9} | {cname:<15} | {px_count:>14,} | {area_km2:>10.2f} km^2 | {pct:>6.2f}%")

    print("-" * 78)
    total_area_km2 = total_valid_px * pixel_area_km2
    print(
        f"{'Total':<9} | {'All Classes':<15} | {total_valid_px:>14,} | {total_area_km2:>10.2f} km^2 | 100.00%"
    )
    print("=" * 78 + "\n")

    return rf, area_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest classifier on spectral features.")
    parser.add_argument("year", type=int, nargs="?", default=2024, help="Analysis year (default: 2024)")
    parser.add_argument("--trees", type=int, default=200, help="Number of trees in Random Forest (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with raster and geojson data")

    args = parser.parse_args()
    train_and_evaluate_classifier(
        year=args.year,
        n_trees=args.trees,
        random_state=args.seed,
        data_dir=args.data_dir,
    )
