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

import geopandas as gpd
import joblib
import numpy as np
import rasterio
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)

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


def load_config(city: str = "ahmedabad", config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads city YAML configuration."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, encoding="utf-8") as f:
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


def train_and_evaluate_classifier(
    city: str = "ahmedabad",
    train_years: list[int] | None = None,
    n_trees: int = 200,
    random_state: int = 42,
    force: bool = False,
    data_dir: str | Path = "data",
    train_geojson: str | Path | None = None,
    test_geojson: str | Path | None = None,
    output_model_path: str | Path | None = None,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """
    Trains a unified Random Forest classifier on pooled multi-year data (200 trees, class_weight='balanced'),
    evaluates on the pooled test set and per-year on held-out blocks, prints metrics & confusion matrices,
    and saves the model to data/{city}/rf_model_pooled.pkl.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    city_subpath = data_path / city_key
    city_subpath.mkdir(parents=True, exist_ok=True)

    if train_years is None:
        cfg = load_config(city=city_key)
        ay = cfg.get("temporal", {}).get("analysis_years", {})
        sy = ay.get("start_year", 2020)
        ey = ay.get("end_year", 2024)
        my = (sy + ey) // 2
        train_years = [sy, my, ey] if sy != ey else [sy]

    if output_model_path is None:
        model_path = city_subpath / "rf_model_pooled.pkl"
    else:
        model_path = Path(output_model_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_pooled_file = data_path / f"{city_key}_train_points_pooled.geojson"
    test_pooled_file = data_path / f"{city_key}_test_points_pooled.geojson"

    # Check if pooled geojson files exist, else sample/extract
    if not train_pooled_file.exists() or not test_pooled_file.exists() or force:
        from pipeline.sample_points import sample_training_points

        sample_training_points(
            city=city_key,
            train_years=train_years,
            data_dir=data_path,
        )

    print("=" * 78)
    print(f"[*] UrbanPulse Pooled Multi-Year Classifier (Training: {city.capitalize()} {train_years})")
    print(f"    - Model Algorithm    : Random Forest ({n_trees} estimators, class_weight='balanced')")
    print(f"    - Input Features     : {FEATURE_NAMES}")
    print(f"    - Pooled Train Data  : {train_pooled_file.resolve()}")
    print(f"    - Pooled Test Data   : {test_pooled_file.resolve()}")
    print(f"    - Model Target Path  : {model_path.resolve()}")
    print("=" * 78)

    train_gdf = gpd.read_file(train_pooled_file)
    test_gdf = gpd.read_file(test_pooled_file)

    # If feature columns are already present in pooled GeoJSON
    has_all_feats = all(f in train_gdf.columns for f in FEATURE_NAMES) and "year" in train_gdf.columns
    if has_all_feats:
        X_train = train_gdf[FEATURE_NAMES].values.astype(np.float32)
        y_train = train_gdf["class_id"].values.astype(np.int64)
        train_years_arr = train_gdf["year"].values

        X_test = test_gdf[FEATURE_NAMES].values.astype(np.float32)
        y_test = test_gdf["class_id"].values.astype(np.int64)
        test_years_arr = test_gdf["year"].values
    else:
        # Fallback extract from rasters
        train_X_list, train_y_list, train_yr_list = [], [], []
        test_X_list, test_y_list, test_yr_list = [], [], []

        for yr in train_years:
            feat_files = [data_path / f"{city_key}_{yr}_{feat}.tif" for feat in FEATURE_NAMES]
            if not all(f.exists() for f in feat_files):
                continue
            with rasterio.open(feat_files[0]) as ref_src:
                raster_crs = ref_src.crs

            X_tr, y_tr, _ = extract_features_at_points(train_gdf, feat_files, raster_crs)
            X_te, y_te, _ = extract_features_at_points(test_gdf, feat_files, raster_crs)

            train_X_list.append(X_tr)
            train_y_list.append(y_tr)
            train_yr_list.append(np.full(len(y_tr), yr))

            test_X_list.append(X_te)
            test_y_list.append(y_te)
            test_yr_list.append(np.full(len(y_te), yr))

        X_train = np.vstack(train_X_list)
        y_train = np.concatenate(train_y_list)
        train_years_arr = np.concatenate(train_yr_list)

        X_test = np.vstack(test_X_list)
        y_test = np.concatenate(test_y_list)
        test_years_arr = np.concatenate(test_yr_list)

    print(f"\n[+] Total Pooled Training Samples : {len(X_train):>5} points across {len(np.unique(train_years_arr))} years")
    print(f"[+] Total Pooled Testing Samples  : {len(X_test):>5} points across {len(np.unique(test_years_arr))} years")

    # 1. Train Random Forest Classifier on Pooled Data
    print(f"\n[Step 1/3] Training Random Forest model on pooled data ({n_trees} trees, class_weight='balanced')...")
    rf = RandomForestClassifier(
        n_estimators=n_trees,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    print("[+] Pooled Random Forest training completed successfully.")

    # Save model to target path and aliases
    joblib.dump(rf, model_path)
    joblib.dump(rf, data_path / f"{city_key}_rf_model_pooled.pkl")
    joblib.dump(rf, data_path / "rf_model_pooled.pkl")
    print(f"[+] Saved trained pooled model to: {model_path.resolve()}")

    # 2. Evaluate on Pooled Test Data
    print("\n[Step 2/3] Evaluating model performance on POOLED spatial block test set...")
    y_pred_pooled = rf.predict(X_test)
    acc_pooled = accuracy_score(y_test, y_pred_pooled)
    kappa_pooled = cohen_kappa_score(y_test, y_pred_pooled)
    report_pooled = classification_report(
        y_test,
        y_pred_pooled,
        labels=list(PROJECT_CLASS_NAMES.keys()),
        target_names=list(PROJECT_CLASS_NAMES.values()),
        digits=4,
        zero_division=0,
    )
    cm_pooled = confusion_matrix(y_test, y_pred_pooled, labels=list(range(1, 6)))

    print("\n" + "=" * 78)
    print(f"[*] POOLED EVALUATION METRICS ({city.capitalize()} - Test Blocks Across All Years)")
    print("=" * 78)
    print(f"Overall Accuracy : {acc_pooled * 100:.2f}%")
    print(f"Cohen's Kappa    : {kappa_pooled:.4f}")
    print("\nClassification Report (Pooled):\n")
    print(report_pooled)

    print("Confusion Matrix (Pooled Test Samples):")
    col_header = "True \\ Pred"
    header_str = f"{col_header:<14} | " + " | ".join(
        [f"{PROJECT_CLASS_NAMES[c][:8]:<8}" for c in range(1, 6)]
    )
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    for i, cid in enumerate(range(1, 6)):
        row_vals = " | ".join([f"{cm_pooled[i, j]:>8}" for j in range(5)])
        print(f"{PROJECT_CLASS_NAMES[cid]:<14} | {row_vals}")
    print("-" * len(header_str))

    # 3. Evaluate PER YEAR on Held-Out Blocks
    print("\n[Step 3/3] Evaluating model performance PER YEAR on held-out blocks...")
    per_year_metrics = {}

    for yr in sorted(list(np.unique(test_years_arr))):
        mask_yr = test_years_arr == yr
        X_test_yr = X_test[mask_yr]
        y_test_yr = y_test[mask_yr]

        if len(y_test_yr) == 0:
            continue

        y_pred_yr = rf.predict(X_test_yr)
        acc_yr = accuracy_score(y_test_yr, y_pred_yr)
        kappa_yr = cohen_kappa_score(y_test_yr, y_pred_yr)
        cm_yr = confusion_matrix(y_test_yr, y_pred_yr, labels=list(range(1, 6)))
        rep_dict = classification_report(
            y_test_yr,
            y_pred_yr,
            labels=list(PROJECT_CLASS_NAMES.keys()),
            target_names=list(PROJECT_CLASS_NAMES.values()),
            output_dict=True,
            zero_division=0,
        )

        per_year_metrics[int(yr)] = {
            "accuracy": float(acc_yr),
            "kappa": float(kappa_yr),
            "confusion_matrix": cm_yr.tolist(),
            "f1_scores": {
                cname: float(rep_dict.get(cname, {}).get("f1-score", 0.0))
                for cname in PROJECT_CLASS_NAMES.values()
            },
        }

        print("\n" + "-" * 78)
        print(f"[*] EVALUATION METRICS FOR YEAR: {yr} (Held-Out Spatial Blocks)")
        print("-" * 78)
        print(f"  - Overall Accuracy : {acc_yr * 100:.2f}%")
        print(f"  - Cohen's Kappa    : {kappa_yr:.4f}")
        print("  - Per-Class F1 Scores:")
        for cid, cname in PROJECT_CLASS_NAMES.items():
            f1 = rep_dict.get(cname, {}).get("f1-score", 0.0)
            prec = rep_dict.get(cname, {}).get("precision", 0.0)
            rec = rep_dict.get(cname, {}).get("recall", 0.0)
            print(f"      * {cname:<12}: F1 = {f1:.4f} (Precision = {prec:.4f}, Recall = {rec:.4f})")

        print(f"\n  Confusion Matrix ({yr}):")
        print(f"  {header_str}")
        print(f"  {'-' * len(header_str)}")
        for i, cid in enumerate(range(1, 6)):
            row_vals = " | ".join([f"{cm_yr[i, j]:>8}" for j in range(5)])
            print(f"  {PROJECT_CLASS_NAMES[cid]:<14} | {row_vals}")
        print(f"  {'-' * len(header_str)}")

    metrics_result = {
        "pooled_accuracy": acc_pooled,
        "pooled_kappa": kappa_pooled,
        "pooled_report": report_pooled,
        "pooled_confusion_matrix": cm_pooled,
        "per_year_metrics": per_year_metrics,
    }

    return rf, metrics_result


def classify_raster(
    city: str = "ahmedabad",
    year: int = 2024,
    model_path: str | Path | None = None,
    force: bool = False,
    data_dir: str | Path = "data",
) -> tuple[Path, dict[str, Any]]:
    """
    Classifies the full 8-channel feature stack for a given city and year using the trained
    pooled Random Forest model, applies a 3x3 majority filter, masks nodata,
    exports the classified GeoTIFF, and calculates area statistics.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    city_subpath = data_path / city_key
    city_subpath.mkdir(parents=True, exist_ok=True)

    classified_path = data_path / f"{city_key}_{year}_classified.tif"

    if classified_path.exists() and not force:
        print(
            f"[*] Classified raster already exists for {city.capitalize()} {year}: {classified_path.name}"
        )
        with rasterio.open(classified_path) as src:
            data = src.read(1)
            pixel_res_x = abs(src.transform.a)
            pixel_res_y = abs(src.transform.e)
            pixel_area_km2 = (pixel_res_x * pixel_res_y) / 1e6

        valid_classified = data > 0
        total_valid_px = int(np.sum(valid_classified))

        area_summary = {}
        for cid in range(1, 6):
            cname = PROJECT_CLASS_NAMES[cid]
            px_count = int(np.sum(data == cid))
            area_km2 = px_count * pixel_area_km2
            pct = (px_count / total_valid_px * 100.0) if total_valid_px > 0 else 0.0
            area_summary[cname] = {
                "class_id": cid,
                "pixel_count": px_count,
                "area_km2": area_km2,
                "percentage": pct,
            }
        return classified_path, area_summary

    # Locate Pooled or Custom Model
    resolved_model_path = None
    if model_path is not None and Path(model_path).exists():
        resolved_model_path = Path(model_path)
    else:
        candidates = [
            city_subpath / "rf_model_pooled.pkl",
            data_path / f"{city_key}_rf_model_pooled.pkl",
            data_path / "rf_model_pooled.pkl",
            city_subpath / f"rf_model_{year}.pkl",
            data_path / f"{city_key}_rf_model_{year}.pkl",
            data_path / f"{city_key}_rf_model.pkl",
            data_path / "rf_model_2024.pkl" if city_key == "ahmedabad" else None,
        ]
        for c in candidates:
            if c is not None and c.exists():
                resolved_model_path = c
                break

    if resolved_model_path is None or not resolved_model_path.exists():
        print(
            f"[*] No pre-trained pooled model found for {city.capitalize()}. Training fresh pooled model..."
        )
        rf, _ = train_and_evaluate_classifier(
            city=city_key,
            train_years=[2018, 2021, 2024],
            data_dir=data_path,
        )
    else:
        print(f"[*] Loading trained Random Forest model: {resolved_model_path.resolve()}...")
        rf = joblib.load(resolved_model_path)

    # 1. Load 8-channel features
    feature_files = [data_path / f"{city_key}_{year}_{feat}.tif" for feat in FEATURE_NAMES]
    for f in feature_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing required feature file: {f}")

    print(
        f"[*] Reading and assembling 8 spectral feature rasters for {city.capitalize()} ({year})..."
    )
    feature_arrays = []
    with rasterio.open(feature_files[0]) as ref_src:
        profile = ref_src.profile.copy()
        raster_transform = ref_src.transform
        raster_shape = (ref_src.height, ref_src.width)

    for fpath in feature_files:
        with rasterio.open(fpath) as src:
            feature_arrays.append(src.read(1).astype(np.float32))

    stack_2d = np.column_stack([arr.ravel() for arr in feature_arrays])
    valid_raster_mask_1d = np.all(np.isfinite(stack_2d) & (stack_2d != -9999.0), axis=1)
    valid_raster_mask_2d = valid_raster_mask_1d.reshape(raster_shape)

    print(f"[*] Total Grid Pixels : {len(stack_2d):,}")
    print(
        f"[*] Valid AOI Pixels  : {int(np.sum(valid_raster_mask_1d)):,} ({np.sum(valid_raster_mask_1d)/len(stack_2d)*100:.1f}%)"
    )

    # Prediction
    raw_classified_1d = np.zeros(len(stack_2d), dtype=np.uint8)
    if np.any(valid_raster_mask_1d):
        raw_classified_1d[valid_raster_mask_1d] = rf.predict(stack_2d[valid_raster_mask_1d])

    raw_classified_2d = raw_classified_1d.reshape(raster_shape)

    # Majority filter
    print("[*] Applying 3x3 majority (mode) filter to reduce salt-and-pepper noise...")
    smooth_classified_2d = apply_majority_filter_3x3(raw_classified_2d, valid_raster_mask_2d)
    print("[+] Applied 3x3 spatial smoothing filter.")

    # Save Classified GeoTIFF
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.pop("tiled", None)
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "lzw",
        }
    )

    with rasterio.open(classified_path, "w", **profile) as dst:
        dst.write(smooth_classified_2d, 1)

    city_classified_path = city_subpath / f"{city_key}_{year}_classified.tif"
    with rasterio.open(city_classified_path, "w", **profile) as dst:
        dst.write(smooth_classified_2d, 1)

    # Area Summary Statistics
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


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate pooled multi-year Random Forest model.")
    parser.add_argument(
        "--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)"
    )
    parser.add_argument(
        "--train-years",
        type=int,
        nargs="+",
        default=None,
        help="Training years to pool (default: derived from config analysis_years)",
    )
    parser.add_argument("--n-trees", type=int, default=200, help="Number of trees (default: 200)")
    parser.add_argument("--model-out", type=str, default=None, help="Output model path")
    parser.add_argument("--force", action="store_true", help="Force retraining")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")

    args = parser.parse_args()
    train_and_evaluate_classifier(
        city=args.city,
        train_years=args.train_years,
        n_trees=args.n_trees,
        output_model_path=args.model_out,
        force=args.force,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
