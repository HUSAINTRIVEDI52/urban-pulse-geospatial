import json
from pathlib import Path
import numpy as np
import rasterio
from scipy.ndimage import binary_erosion
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path("f:/gis-project/UrbanPulse")
DATA_DIR = PROJECT_ROOT / "data"

BANDS = ["blue", "green", "red", "nir", "swir16"]
YEARS = list(range(2018, 2025))
REF_YEAR = 2021

for city in ["ahmedabad", "pune"]:
    print("\n" + "=" * 90)
    print(f"TESTING RADIOMETRIC NORMALIZATION FOR {city.upper()} (REF YEAR = {REF_YEAR})")
    print("=" * 90)
    
    clean_dir = DATA_DIR / city / "clean"
    clean_rasters = []
    for y in YEARS:
        p = clean_dir / f"{city}_{y}_classified.tif"
        if not p.exists():
            p = DATA_DIR / f"{city}_{y}_classified.tif"
        with rasterio.open(p) as src:
            clean_rasters.append(src.read(1))
    
    stack = np.stack(clean_rasters, axis=0) # (7, H, W)
    
    # Identify PIFs: Water (3) in >=6 of 7 years OR Built-up (1) in >=6 of 7 years
    water_pif = np.sum(stack == 3, axis=0) >= 6
    built_pif = np.sum(stack == 1, axis=0) >= 6
    
    struct_3x3 = np.ones((3, 3), dtype=bool)
    water_pif_eroded = binary_erosion(water_pif, structure=struct_3x3)
    built_pif_eroded = binary_erosion(built_pif, structure=struct_3x3)
    pif_mask = water_pif_eroded | built_pif_eroded
    
    print(f"PIF mask count: {np.sum(pif_mask):,} pixels (Water: {np.sum(water_pif_eroded):,}, Built-up: {np.sum(built_pif_eroded):,})")
    
    # Load reference year 2021 bands
    ref_bands = {}
    for b in BANDS:
        p = DATA_DIR / f"{city}_{REF_YEAR}_{b}.tif"
        with rasterio.open(p) as src:
            ref_bands[b] = src.read(1).astype(np.float32)
            profile = src.profile.copy()
            
    coeffs = {}
    
    for y in YEARS:
        coeffs[y] = {}
        print(f"\n--- Year {y} vs Reference {REF_YEAR} ---")
        print(f"{'Band':<8} | {'Slope (m)':<10} | {'Intercept (c)':<14} | {'R²':<8} | {'Before Mean':<12} | {'After Mean':<12} | {'Ref (2021) Mean'}")
        print("-" * 90)
        
        for b in BANDS:
            p = DATA_DIR / f"{city}_{y}_{b}.tif"
            with rasterio.open(p) as src:
                curr_arr = src.read(1).astype(np.float32)
                
            x_vals = curr_arr[pif_mask]
            y_vals = ref_bands[b][pif_mask]
            
            valid = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals != -9999.0) & (y_vals != -9999.0)
            x_clean = x_vals[valid].reshape(-1, 1)
            y_clean = y_vals[valid]
            
            if y == REF_YEAR:
                slope, intercept, r2 = 1.0, 0.0, 1.0
            else:
                reg = LinearRegression().fit(x_clean, y_clean)
                slope = float(reg.coef_[0])
                intercept = float(reg.intercept_)
                r2 = float(reg.score(x_clean, y_clean))
                
            norm_pif_vals = slope * x_clean.ravel() + intercept
            
            before_mean = float(np.mean(x_clean))
            after_mean = float(np.mean(norm_pif_vals))
            ref_mean = float(np.mean(y_clean))
            
            coeffs[y][b] = {
                "slope": round(slope, 6),
                "intercept": round(intercept, 6),
                "r2": round(r2, 4),
                "before_mean": round(before_mean, 6),
                "after_mean": round(after_mean, 6),
                "ref_mean": round(ref_mean, 6),
            }
            
            print(f"{b:<8} | {slope:<10.4f} | {intercept:<14.4f} | {r2:<8.4f} | {before_mean:<12.4f} | {after_mean:<12.4f} | {ref_mean:<12.4f}")
