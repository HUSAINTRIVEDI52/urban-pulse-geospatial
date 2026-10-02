import sys
from pathlib import Path
import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.ndimage import binary_erosion

PROJECT_ROOT = Path("f:/gis-project/UrbanPulse")
data_dir = PROJECT_ROOT / "data"

for city in ["ahmedabad", "pune"]:
    print(f"\n--- Checking {city} ---")
    clean_dir = data_dir / city / "clean"
    years = list(range(2018, 2025))
    clean_rasters = []
    for y in years:
        p = clean_dir / f"{city}_{y}_classified.tif"
        if not p.exists():
            p = data_dir / f"{city}_{y}_classified.tif"
        with rasterio.open(p) as src:
            clean_rasters.append(src.read(1))
    
    stack = np.stack(clean_rasters, axis=0) # (7, H, W)
    
    # Check pixels where class is Water (3) in >= 6 years or Built-up (1) in >= 6 years
    water_counts = np.sum(stack == 3, axis=0)
    built_counts = np.sum(stack == 1, axis=0)
    
    water_pif_mask = water_counts >= 6
    built_pif_mask = built_counts >= 6
    combined_pif_mask = water_pif_mask | built_pif_mask
    
    # Morphological erosion to stay away from edges
    struct_3x3 = np.ones((3, 3), dtype=bool)
    water_eroded = binary_erosion(water_pif_mask, structure=struct_3x3)
    built_eroded = binary_erosion(built_pif_mask, structure=struct_3x3)
    pif_eroded = water_eroded | built_eroded
    
    print(f"Total grid shape: {stack[0].shape}")
    print(f"Water PIFs raw: {np.sum(water_pif_mask):,}, eroded: {np.sum(water_eroded):,}")
    print(f"Built-up PIFs raw: {np.sum(built_pif_mask):,}, eroded: {np.sum(built_eroded):,}")
    print(f"Total PIF eroded pixels: {np.sum(pif_eroded):,}")
    
    # Stable points across all 7 years (same class in all 7 years)
    stable_all_7 = (stack == stack[0:1]).all(axis=0) & (stack[0] > 0)
    print(f"Pixels perfectly stable across all 7 years: {np.sum(stable_all_7):,} ({np.sum(stable_all_7)/np.sum(stack[0]>0)*100:.1f}%)")
    for cid, cname in {1: "Built-up", 2: "Vegetation", 3: "Water", 4: "Agriculture", 5: "Open land"}.items():
        cnt = np.sum(stable_all_7 & (stack[0] == cid))
        print(f"  Class {cid} ({cname}): {cnt:,} stable pixels")
