from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path("f:/gis-project/UrbanPulse")
DATA_DIR = PROJECT_ROOT / "data"

for city in ["ahmedabad", "pune"]:
    print(f"\n==================== {city.upper()} ====================")
    # Check band rasters and classified rasters
    files_to_check = [
        DATA_DIR / f"{city}_2021_blue.tif",
        DATA_DIR / f"{city}_2021_classified.tif",
        DATA_DIR / f"{city}_worldcover_labels.tif",
        DATA_DIR / city / f"{city}_worldcover_labels.tif"
    ]
    for p in files_to_check:
        if p.exists():
            with rasterio.open(p) as src:
                arr = src.read(1)
                res_x, res_y = abs(src.transform.a), abs(src.transform.e)
                px_area_m2 = res_x * res_y
                px_area_km2 = px_area_m2 / 1e6
                h, w = src.height, src.width
                total_px = h * w
                valid_px = int(np.sum((arr != src.nodata) & (arr > 0) & np.isfinite(arr))) if src.nodata is not None else int(np.sum(arr > 0))
                total_grid_area_km2 = total_px * px_area_km2
                valid_area_km2 = valid_px * px_area_km2
                print(f"File: {p.name}")
                print(f"  Shape: ({h}, {w}) = {total_px:,} pixels")
                print(f"  Transform: {src.transform}")
                print(f"  Resolution: {res_x:.4f} m x {res_y:.4f} m | Pixel Area: {px_area_m2:.2f} m² = {px_area_km2:.8f} km²")
                print(f"  CRS: {src.crs}")
                print(f"  Total Grid Area: {total_grid_area_km2:.4f} km²")
                print(f"  Valid Pixels (>0): {valid_px:,} ({valid_area_km2:.4f} km²)")
