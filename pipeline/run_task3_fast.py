"""
UrbanPulse - Task 3: 2022 Legacy Stable-Pixel Reflectance & 2021 Legacy vs Collection 1 Comparison
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.scene_selection import load_city_config, extract_mgrs_tile

DATA_DIR = PROJECT_ROOT / "data"


def compute_legacy_scene_medians(item, bounds, shape, stable_mask):
    dt_str = item.datetime.strftime("%Y-%m-%d") if item.datetime else str(item.properties.get("datetime"))[:10]
    tile_id = extract_mgrs_tile(item)
    baseline = str(item.properties.get("s2:processing_baseline", item.properties.get("processing_baseline", "N/A")))
    cloud_cover = float(item.properties.get("eo:cloud_cover", 100.0))
    
    band_keys = {"red": "red", "nir": "nir", "swir16": "swir16", "blue": "blue", "scl": "scl"}
    arrays = {}
    
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="25", GDAL_HTTP_MAX_RETRY="2"):
        for b_name, asset_key in band_keys.items():
            if asset_key in item.assets:
                href = item.assets[asset_key].href
                for attempt in range(2):
                    try:
                        with rasterio.open(href) as src:
                            win = from_bounds(*bounds, src.transform)
                            resamp = Resampling.nearest if b_name == "scl" else Resampling.bilinear
                            arr = src.read(1, window=win, out_shape=shape, resampling=resamp)
                            arrays[b_name] = arr.astype(np.float32)
                            break
                    except Exception:
                        pass

    scl_arr = arrays.get("scl")
    if scl_arr is not None:
        c_mask = ((scl_arr == 0) | (scl_arr == 1) | (scl_arr == 3) | (scl_arr == 8) | (scl_arr == 9) | (scl_arr == 10) | (scl_arr == 11) | np.isnan(scl_arr))
        dil_mask = scipy.ndimage.binary_dilation(c_mask, iterations=1)
    else:
        dil_mask = np.zeros(shape, dtype=bool)

    medians = {}
    v_counts = []
    for b_name in ["red", "nir", "swir16", "blue"]:
        raw = arrays.get(b_name)
        if raw is not None:
            # Baseline offset
            offset = -0.1 if (dt_str >= "2022-01-25" or baseline.startswith("04")) else 0.0
            scale = 0.0001
            scaled = np.clip(raw * scale + offset, 0.0, 1.0)
            scaled[dil_mask] = np.nan
            scaled[np.isnan(raw) | (raw <= 0)] = np.nan

            v_mask = stable_mask & np.isfinite(scaled) & (scaled > 0)
            v_px = scaled[v_mask]
            v_counts.append(int(v_mask.sum()))
            medians[b_name] = round(float(np.nanmedian(v_px)), 4) if len(v_px) > 0 else None
        else:
            medians[b_name] = None

    return {
        "date": dt_str,
        "tile": tile_id,
        "baseline": baseline,
        "cloud": round(cloud_cover, 2),
        "valid_px": int(np.median(v_counts)) if v_counts else 0,
        "red": medians.get("red"),
        "nir": medians.get("nir"),
        "swir": medians.get("swir16"),
        "blue": medians.get("blue"),
        "scene_id": item.id,
    }


def main():
    client = Client.open("https://earth-search.aws.element84.com/v1")
    
    for city in ["ahmedabad", "pune"]:
        print(f"\n{'='*100}\n[TASK 3] 2022 LEGACY SCENES REFLECTANCE MEDIANS: {city.upper()}\n{'='*100}", flush=True)
        cfg = load_city_config(city)
        bbox = cfg["spatial"]["bbox"]
        
        # Load mask
        classified_tif = DATA_DIR / city / f"{city}_2024_classified.tif"
        with rasterio.open(classified_tif) as src:
            bounds = src.bounds
            shape = (src.height // 2, src.width // 2)  # fast subsampled evaluation
            
        cls_stack = []
        for y in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
            c_p = DATA_DIR / city / f"{city}_{y}_classified.tif"
            if c_p.exists():
                with rasterio.open(c_p) as s:
                    cls_stack.append(s.read(1, out_shape=shape, resampling=Resampling.nearest))
        cls_arr = np.stack(cls_stack, axis=0)
        stable_mask = ((cls_arr == 1) | (cls_arr == 3)).sum(axis=0) >= 6
        print(f"[+] Loaded stable pixel mask: {stable_mask.sum():,} pixels", flush=True)

        # 1. 2022 Legacy Scenes
        search_2022 = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2022-01-01/2022-02-28",
            query={"eo:cloud_cover": {"lt": 20.0}},
        )
        items_2022 = list(search_2022.items())
        print(f"[+] Query returned {len(items_2022)} scenes in Jan-Feb 2022. Evaluating stable-pixel medians...", flush=True)
        
        records_2022 = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(compute_legacy_scene_medians, it, bounds, shape, stable_mask)
                for it in items_2022
            ]
            for f in as_completed(futures):
                try:
                    res = f.result()
                    records_2022.append(res)
                except Exception as e:
                    print(f"[!] Error: {e}")
                    
        records_2022.sort(key=lambda r: (r["date"], r["tile"], r["scene_id"]))
        df_2022 = pd.DataFrame(records_2022)
        print("\n[+] 2022 Legacy Scenes Table:")
        print(df_2022[["date", "tile", "baseline", "cloud", "valid_px", "red", "nir", "swir", "blue", "scene_id"]].to_string(index=False), flush=True)

        # 2. 2021 Legacy vs Collection 1 Comparison on Matching Dates
        print(f"\n[+] 2021 Legacy vs Collection 1 Matching Dates Reflectance Comparison ({city.upper()}):", flush=True)
        search_leg = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2021-01-01/2021-01-31",
            query={"eo:cloud_cover": {"lt": 10.0}},
        )
        search_c1 = client.search(
            collections=["sentinel-2-c1-l2a"],
            bbox=bbox,
            datetime="2021-01-01/2021-01-31",
            query={"eo:cloud_cover": {"lt": 10.0}},
        )
        items_leg = list(search_leg.items())
        items_c1 = list(search_c1.items())
        
        leg_map = {f"{it.datetime.strftime('%Y-%m-%d')}_{extract_mgrs_tile(it)}": it for it in items_leg if it.datetime}
        c1_map = {f"{it.datetime.strftime('%Y-%m-%d')}_{extract_mgrs_tile(it)}": it for it in items_c1 if it.datetime}
        matching_keys = sorted(list(set(leg_map.keys()) & set(c1_map.keys())))
        
        comp_rows = []
        for mk in matching_keys[:4]:
            it_leg = leg_map[mk]
            it_c1 = c1_map[mk]
            res_leg = compute_legacy_scene_medians(it_leg, bounds, shape, stable_mask)
            res_c1 = compute_legacy_scene_medians(it_c1, bounds, shape, stable_mask)
            
            comp_rows.append({
                "Date": res_leg["date"],
                "Tile": res_leg["tile"],
                "C1 Baseline": res_c1["baseline"],
                "Leg Baseline": res_leg["baseline"],
                "C1 Red": res_c1["red"],
                "Leg Red": res_leg["red"],
                "C1 NIR": res_c1["nir"],
                "Leg NIR": res_leg["nir"],
                "C1 SWIR": res_c1["swir"],
                "Leg SWIR": res_leg["swir"],
                "C1 Blue": res_c1["blue"],
                "Leg Blue": res_leg["blue"],
            })
        print(pd.DataFrame(comp_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
