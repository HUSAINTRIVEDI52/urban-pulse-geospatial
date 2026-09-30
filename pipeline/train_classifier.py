"""
UrbanPulse - Land Cover Random Forest Classifier & Mapping Module
Loads multi-spectral bands & indices, extracts spectral signatures for ground truth points,
trains a Random Forest classifier (200 trees), evaluates on spatial-block test points,
applies a 3x3 majority (mode) filter to eliminate salt-and-pepper noise,
masks nodata pixels, and exports classified map and area statistics.
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

    valid_mask = np.all(np.isfinite(X) & (X != -9999.0), axis=1)
    return X[valid_mask], y[valid_mask], valid_mask


def apply_majority_filter_3x3(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Applies an ultra-fast vectorized 3x3 majority (mode) filter to 2D classified labels
    to eliminate isolated single-pixel salt-and-pepper artifacts while preserving nodata boundaries.
    """
    h, w = arr.shape
    padded = np.pad(arr, 1, mode="edge")

    # Stack 9 neighborhood slices
    neighbors = np.stack(
        [
            padded[0:h, 0:w],
            padded[0:h, 1 : w + 1],
            padded[0:h, 2 : w + 2],
            padded[1 : h + 1, 0:w],
            padded[1 : h + 1, 1 : w + 1],
            padded[1 : h + 1, 2 : w + 2],
            padded[2 : h + 2, 0:w],
            padded[2 : h + 2, 1 : w + 1],
            padded[2 : h + 2, 2 : w + 2],
        ],
        axis=0,
    )

    # Compute vote counts for classes 1 through 5
    class_votes = np.stack([(neighbors == c).sum(axis=0) for c in range(1, 6)], axis=0)
    majority_class = (np.argmax(class_votes, axis=0) + 1).astype(np.uint8)

    # Preserve nodata background mask
    filtered_map = np.where(valid_mask, majority_class, 0).astype(np.uint8)
    return filtered_map


def classify_raster(
    city: str = "ahmedabad",
    year: int = 2024,
    model_path: str | Path = "data/rf_model_2024.pkl",
    force: bool = False,
    data_dir: str | Path = "data",
) -> tuple[Path, dict[str, Any]]:
    """
    Classifies the full 8-channel feature stack for a given city and year using a trained
    Random Forest model, applies a 3x3 majority (mode) filter to reduce noise, masks nodata,
    exports the classified GeoTIFF, and calculates area statistics.

    Skips classification if the output GeoTIFF already exists unless force=True.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    model_file = Path(model_path)
    classified_path = data_path / f"{city_key}_{year}_classified.tif"

    if not model_file.exists():
        raise FileNotFoundError(f"Trained model not found at {model_file.resolve()}. Run train_and_evaluate_classifier first.")

    # 1. Feature file paths
    feature_files = [data_path / f"{city_key}_{year}_{feat}.tif" for feat in FEATURE_NAMES]

    # If already exists and not force, read and print area summary
    if not force and classified_path.exists():
        print(f"[+] Classified map already exists at: {classified_path.resolve()}. Skipping inference.")
        with rasterio.open(classified_path) as src:
            classified_arr = src.read(1)
            transform = src.transform
            pixel_res_x = abs(transform.a)
            pixel_res_y = abs(transform.e)
            pixel_area_km2 = (pixel_res_x * pixel_res_y) / 1e6

        valid_classified = classified_arr > 0
        total_valid_px = int(np.sum(valid_classified))

        print("\n" + "=" * 78)
        print(f"[*] {city.capitalize()} {year} Land Cover Area Summary (NoData Excluded)")
        print("=" * 78)
        print(
            f"{'Class ID':<9} | {'Class Name':<15} | {'Pixel Count':<14} | {'Area (km^2)':<12} | {'Percentage'}"
        )
        print("-" * 78)

        area_summary = {}
        for cid in range(1, 6):
            cname = PROJECT_CLASS_NAMES[cid]
            px_count = int(np.sum(classified_arr == cid))
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
        return classified_path, area_summary

    # Check for missing feature files
    missing_features = [f.name for f in feature_files if not f.exists()]
    if missing_features:
        raise FileNotFoundError(
            f"Missing required feature rasters in {data_path}: {missing_features}. "
            "Please run composite and spectral indices steps first."
        )

    print(f"\n[*] Loading Random Forest model from {model_file.name}...")
    rf: RandomForestClassifier = joblib.load(model_file)

    with rasterio.open(feature_files[0]) as ref_src:
        profile = ref_src.profile.copy()
        raster_transform = ref_src.transform
        raster_shape = ref_src.shape

    print(f"[*] Classifying full raster stack for {city.capitalize()} ({year}) [{raster_shape[0]} x {raster_shape[1]} pixels]...")
    feature_arrays = []
    for fpath in feature_files:
        with rasterio.open(fpath) as src:
            feature_arrays.append(src.read(1).astype(np.float32))

    stack_2d = np.column_stack([arr.ravel() for arr in feature_arrays])
    valid_raster_mask_1d = np.all(np.isfinite(stack_2d) & (stack_2d != -9999.0), axis=1)
    valid_raster_mask_2d = valid_raster_mask_1d.reshape(raster_shape)

    print(f"[*] Total Grid Pixels : {len(stack_2d):,}")
    print(f"[*] Valid AOI Pixels  : {int(np.sum(valid_raster_mask_1d)):,} ({np.sum(valid_raster_mask_1d)/len(stack_2d)*100:.1f}%)")

    # Raw Pixel-Level Prediction
    raw_classified_1d = np.zeros(len(stack_2d), dtype=np.uint8)
    if np.any(valid_raster_mask_1d):
        raw_classified_1d[valid_raster_mask_1d] = rf.predict(stack_2d[valid_raster_mask_1d])

    raw_classified_2d = raw_classified_1d.reshape(raster_shape)

    # Apply 3x3 Majority (Mode) Filter
    print("[*] Applying 3x3 majority (mode) filter to reduce salt-and-pepper noise...")
    smooth_classified_2d = apply_majority_filter_3x3(raw_classified_2d, valid_raster_mask_2d)
    print("[+] Applied 3x3 spatial smoothing filter.")

    # Save Classified GeoTIFF
    print(f"[*] Saving final classified GeoTIFF to: {classified_path.name}...")
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
        dst.write(smooth_classified_2d, 1)

    # Area Summary Statistics (Excluding NoData)
    pixel_res_x = abs(raster_transform.a)
    pixel_res_y = abs(raster_transform.e)
    pixel_area_km2 = (pixel_res_x * pixel_res_y) / 1e6

    valid_classified = smooth_classified_2d > 0
    total_valid_px = int(np.sum(valid_classified))

    print("\n" + "=" * 78)
    print(f"[*] {city.capitalize()} {year} Final Land Cover Area Summary (NoData Excluded)")
    print("=" * 78)
    print(
        f"{'Class ID':<9} | {'Class Name':<15} | {'Pixel Count':<14} | {'Area (km^2)':<12} | {'Percentage'}"
    )
    print("-" * 78)

    area_summary = {}
    for cid in range(1, 6):
        cname = PROJECT_CLASS_NAMES[cid]
        px_count = int(np.sum(smooth_classified_2d == cid))
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

    return classified_path, area_summary


def train_and_evaluate_classifier(
    city: str = "ahmedabad",
    year: int = 2024,
    n_trees: int = 200,
    random_state: int = 42,
    force: bool = False,
    data_dir: str | Path = "data",
    train_geojson: str | Path = "data/train_points.geojson",
    test_geojson: str | Path = "data/test_points.geojson",
    output_model_path: str | Path = "data/rf_model_2024.pkl",
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """
    Trains Random Forest on sample points, evaluates on spatial test split,
    and runs full raster classification.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    train_path = Path(train_geojson)
    test_path = Path(test_geojson)
    model_path = Path(output_model_path)

    print("=" * 78)
    print(f"[*] UrbanPulse Land Cover Classifier (Training on {city.capitalize()} {year})")
    print(f"    - Model Algorithm    : Random Forest ({n_trees} estimators)")
    print(f"    - Input Features     : {FEATURE_NAMES}")
    print(f"    - Train Dataset      : {train_path.resolve()}")
    print(f"    - Test Dataset       : {test_path.resolve()}")
    print(f"    - Model Output       : {model_path.resolve()}")
    print("=" * 78)

    # 1. Verify Feature Rasters
    feature_files = [data_path / f"{city_key}_{year}_{feat}.tif" for feat in FEATURE_NAMES]
    missing_features = [f.name for f in feature_files if not f.exists()]
    if missing_features:
        raise FileNotFoundError(
            f"Missing required feature rasters in {data_path}: {missing_features}. "
            "Please run pipeline/build_composite.py and pipeline/compute_indices.py first."
        )

    with rasterio.open(feature_files[0]) as ref_src:
        raster_crs = ref_src.crs

    # 2. Extract Training and Testing Feature Matrices
    print("\n[Step 1/4] Extracting 8-channel spectral features at sample point locations...")
    train_gdf = gpd.read_file(train_path)
    test_gdf = gpd.read_file(test_path)

    X_train, y_train, _ = extract_features_at_points(train_gdf, feature_files, raster_crs)
    X_test, y_test, _ = extract_features_at_points(test_gdf, feature_files, raster_crs)

    print(f"[+] Clean Training Samples : {len(X_train):>5} points across 8 features")
    print(f"[+] Clean Testing Samples  : {len(X_test):>5} points across 8 features")

    # 3. Train Random Forest Classifier
    print(f"\n[Step 2/4] Training Random Forest model ({n_trees} trees, all CPU cores)...")
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
    print("\n[Step 3/4] Evaluating model performance on independent spatial test blocks...")
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

    # 5. Classify Full Raster
    print(f"\n[Step 4/4] Executing full raster classification for {city} ({year})...")
    _, area_summary = classify_raster(
        city=city,
        year=year,
        model_path=model_path,
        force=force,
        data_dir=data_dir,
    )

    return rf, area_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest classifier on spectral features.")
    parser.add_argument("--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)")
    parser.add_argument("--year", type=int, default=2024, help="Analysis year (default: 2024)")
    parser.add_argument("--trees", type=int, default=200, help="Number of trees in Random Forest (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--force", action="store_true", help="Force recalculate outputs")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with raster and geojson data")

    args = parser.parse_args()
    train_and_evaluate_classifier(
        city=args.city,
        year=args.year,
        n_trees=args.trees,
        random_state=args.seed,
        force=args.force,
        data_dir=args.data_dir,
    )

