import sys
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import rasterio

PROJECT_ROOT = Path("f:/gis-project/UrbanPulse")
DATA_DIR = PROJECT_ROOT / "data"

PROJECT_CLASS_NAMES = {
    1: "Built-up",
    2: "Vegetation",
    3: "Water",
    4: "Agriculture",
    5: "Open land",
}
FEATURE_NAMES = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]

for city in ["ahmedabad", "pune"]:
    print("=" * 80)
    print(f"CITY: {city.upper()} - LEAVE-ONE-YEAR-OUT (BEFORE NORMALISATION)")
    print("=" * 80)
    
    tr_p = DATA_DIR / city / "train_points_pooled.geojson"
    te_p = DATA_DIR / city / "test_points_pooled.geojson"
    gdf_tr = gpd.read_file(tr_p)
    gdf_te = gpd.read_file(te_p)
    
    # We can evaluate on test points of heldout year, or all points of heldout year. Let's check both or test_points.
    # To strictly preserve spatial blocks: train on train_points of (years != holdout), test on test_points of holdout (or all points of holdout)
    all_pts = pd.concat([gdf_tr, gdf_te], ignore_index=True)
    
    ref_tif = DATA_DIR / city / f"{city}_2024_classified.tif"
    with rasterio.open(ref_tif) as src:
        total_aoi_km2 = (src.height * src.width * src.res[0] * src.res[1]) / 1e6
        px_km2 = (src.res[0] * src.res[1]) / 1e6
    
    for holdout_yr in [2018, 2021, 2024]:
        print(f"\n>>> Hold-out Year: {holdout_yr}")
        # Train on remaining years
        train_df = gdf_tr[gdf_tr["year"] != holdout_yr]
        test_df = gdf_te[gdf_te["year"] == holdout_yr]
        
        X_train = train_df[FEATURE_NAMES].values
        y_train = train_df["class_id"].values
        X_test = test_df[FEATURE_NAMES].values
        y_test = test_df["class_id"].values
        
        rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])
        print("Confusion Matrix (Labels: 1=Built-up, 2=Vegetation, 3=Water, 4=Agriculture, 5=Open land):")
        print(f"{'True \\ Pred':<14} | " + " | ".join([f"{PROJECT_CLASS_NAMES[c][:8]:<8}" for c in range(1, 6)]))
        print("-" * 65)
        for i, cid in enumerate(range(1, 6)):
            row_str = " | ".join([f"{cm[i, j]:>8}" for j in range(5)])
            print(f"{PROJECT_CLASS_NAMES[cid]:<14} | {row_str}")
        print("-" * 65)
        
        # False positives for Built-up: column 0 (pred == 1) for rows i != 0 (true != 1)
        fp_builtup = {}
        for i, cid in enumerate(range(2, 6)): # classes 2, 3, 4, 5
            fp_count = cm[cid-1, 0]
            if fp_count > 0:
                fp_builtup[PROJECT_CLASS_NAMES[cid]] = fp_count
        print(f"False Positives for Built-up (Predicted as Built-up, but True class was): {fp_builtup} (Total FP = {sum(fp_builtup.values())})")
        
        y_true_bin = (y_test == 1).astype(int)
        y_pred_bin = (y_pred == 1).astype(int)
        
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        
        # Area bias
        pred_prop = np.mean(y_pred_bin)
        true_prop = np.mean(y_true_bin)
        area_bias_km2 = (pred_prop - true_prop) * total_aoi_km2
        
        print(f"Built-up Metrics: Precision = {prec:.4f}, Recall = {rec:.4f}, F1 = {f1:.4f}, Area Bias = {area_bias_km2:+.2f} km²")
