"""
UrbanPulse - Unified Execution & Diagnostics Engine for Tasks 1, 2, 3, 4
"""

import calendar
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

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
import joblib

from pipeline.scene_selection import (
    load_city_config,
    get_strict_window_dates,
    query_strict_window_scenes,
    extract_mgrs_tile,
    deduplicate_tile_date_records,
    apply_tile_quality_screening,
)

DATA_DIR = PROJECT_ROOT / "data"


# ==============================================================================
# TASK 1: Prove Window Dates & Diff composite_scenes.csv vs scene_diagnostics.csv
# ==============================================================================
def run_task1():
    print("=" * 100, flush=True)
    print("TASK 1: PROVE 2020 STRICT WINDOW & DIFF COMPOSITE_SCENES.CSV VS SCENE_DIAGNOSTICS.CSV", flush=True)
    print("=" * 100, flush=True)
    
    for city in ["ahmedabad", "pune"]:
        cfg = load_city_config(city)
        s_date, e_date, iso_range = get_strict_window_dates(2020, cfg)
        print(f"\n[+] {city.upper()} 2020 Parsed Strict Window:")
        print(f"    - Start Date : {s_date} (Nov 1 of prior year)")
        print(f"    - End Date   : {e_date} (Feb 29 of leap year 2020)")
        print(f"    - ISO Range  : {iso_range}")
        
        diag_csv = DATA_DIR / city / "scene_diagnostics.csv"
        diag_df = pd.read_csv(diag_csv)
        
        # Build composite_scenes.csv
        diag_records = diag_df.to_dict("records")
        deduped_records, dropped_dups = deduplicate_tile_date_records(diag_records)
        kept_records, dropped_quality, tile_stats = apply_tile_quality_screening(deduped_records)
        
        comp_csv = DATA_DIR / city / "composite_scenes.csv"
        all_comp_rows = []
        for r in kept_records:
            rc = dict(r)
            rc["selected_for_composite"] = "YES"
            all_comp_rows.append(rc)
        for r in dropped_quality:
            rc = dict(r)
            rc["selected_for_composite"] = "NO"
            all_comp_rows.append(rc)
        for r in dropped_dups:
            rc = dict(r)
            rc["selected_for_composite"] = "NO"
            all_comp_rows.append(rc)
            
        comp_df = pd.DataFrame(all_comp_rows)
        comp_df.sort_values(by=["year", "date", "mgrs_tile", "scene_id"], inplace=True)
        comp_df.to_csv(comp_csv, index=False)
        print(f"[+] Saved {comp_csv.resolve()} ({len(comp_df)} total records, {len(kept_records)} used in composite)")
        
        # Diff against scene_diagnostics.csv
        diag_scenes = set(diag_df["scene_id"])
        comp_scenes = set(comp_df["scene_id"])
        missing_in_comp = diag_scenes - comp_scenes
        extra_in_comp = comp_scenes - diag_scenes
        
        print(f"[+] Diff Results for {city.upper()}:")
        print(f"    - Scenes in scene_diagnostics.csv : {len(diag_scenes)}")
        print(f"    - Scenes in composite_scenes.csv  : {len(comp_scenes)}")
        print(f"    - Missing in composite_scenes.csv : {len(missing_in_comp)} -> {missing_in_comp if missing_in_comp else 'None (0 mismatch)'}")
        print(f"    - Extra in composite_scenes.csv   : {len(extra_in_comp)} -> {extra_in_comp if extra_in_comp else 'None (0 mismatch)'}")


# ==============================================================================
# TASK 2: Deduplicate (Tile, Date) Pairs & Recount
# ==============================================================================
def run_task2():
    print("\n" + "=" * 100, flush=True)
    print("TASK 2: DEDUPLICATE (TILE, DATE) PAIRS & RECOUNT SCENES/DATES PER YEAR", flush=True)
    print("=" * 100, flush=True)
    
    for city in ["ahmedabad", "pune"]:
        diag_csv = DATA_DIR / city / "scene_diagnostics.csv"
        diag_df = pd.read_csv(diag_csv)
        diag_records = diag_df.to_dict("records")
        
        deduped_records, dropped_dups = deduplicate_tile_date_records(diag_records)
        print(f"\n[+] {city.upper()}: Identified {len(dropped_dups)} Duplicate (Tile, Date) Scenes:")
        for d in dropped_dups:
            print(f"    - Year {d['year']} | Date {d['date']} | Tile {d['mgrs_tile']:<5} | Dropped ID: {d['scene_id']}")
            print(f"      Valid Px: {d['valid_stable_pixels']:,} | Cloud: {d['cloud_cover_pct']:.2f}% | Reason: {d['drop_reasons']}")
            
        kept_records, dropped_quality, tile_stats = apply_tile_quality_screening(deduped_records)
        
        print(f"\n[+] {city.upper()} Per-Year Recount (Nov 1 - Feb 28/29 Strict Window, Deduplicated):")
        summary_rows = []
        for yr in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
            if yr == 2022:
                summary_rows.append({
                    "Year": yr,
                    "Raw Scenes": 0,
                    "Deduplicated": 0,
                    "Quality Kept": 0,
                    "Quality Dropped": 0,
                    "Kept Dates": 0,
                    "Status": "ARCHIVE-GAP",
                })
                continue
            cand_yr = [r for r in diag_records if r["year"] == yr]
            dedup_yr = [r for r in deduped_records if r["year"] == yr]
            kept_yr = [r for r in kept_records if r["year"] == yr]
            drop_yr = [r for r in dropped_quality if r["year"] == yr]
            dates_yr = sorted(list({r["date"] for r in kept_yr}))
            summary_rows.append({
                "Year": yr,
                "Raw Scenes": len(cand_yr),
                "Deduplicated": len(dedup_yr),
                "Quality Kept": len(kept_yr),
                "Quality Dropped": len(drop_yr),
                "Kept Dates": len(dates_yr),
                "Status": "VALID" if len(dates_yr) >= 4 else "LOW_CONFIDENCE (<4 dates)",
            })
        sum_df = pd.DataFrame(summary_rows)
        print(sum_df.to_string(index=False), flush=True)


# ==============================================================================
# TASK 3: 2022 Legacy Stable-Pixel Medians & 2021 Legacy vs Collection 1
# ==============================================================================
def process_single_legacy_scene(it, city_bounds, city_shape, stable_mask, dilation_structure):
    dt_str = it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10]
    tile_id = extract_mgrs_tile(it)
    baseline = str(it.properties.get("s2:processing_baseline", it.properties.get("processing_baseline", "N/A")))
    cloud_cover = float(it.properties.get("eo:cloud_cover", 100.0))
    
    band_map = {"red": ["red", "B04"], "nir": ["nir", "B08"], "swir16": ["swir16", "B11"], "blue": ["blue", "B02"], "scl": ["scl", "SCL"]}
    assets_data = {}
    
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_TIMEOUT="20"):
        for b_k, candidate_keys in band_map.items():
            found_key = next((k for k in candidate_keys if k in it.assets), None)
            if found_key:
                href = it.assets[found_key].href
                for attempt in range(3):
                    try:
                        with rasterio.open(href) as b_src:
                            win = from_bounds(*city_bounds, b_src.transform)
                            resamp = Resampling.nearest if b_k == "scl" else Resampling.bilinear
                            arr = b_src.read(1, window=win, out_shape=city_shape, resampling=resamp)
                            assets_data[b_k] = arr.astype(np.float32)
                            break
                    except Exception:
                        pass
                        
    scl_arr = assets_data.get("scl")
    if scl_arr is not None:
        c_mask = ((scl_arr == 0) | (scl_arr == 1) | (scl_arr == 3) | (scl_arr == 8) | (scl_arr == 9) | (scl_arr == 10) | (scl_arr == 11) | np.isnan(scl_arr))
        dil_mask = scipy.ndimage.binary_dilation(c_mask, structure=dilation_structure, iterations=1)
    else:
        dil_mask = np.zeros(city_shape, dtype=bool)
        
    b_medians = {}
    v_counts = []
    for b_k in ["red", "nir", "swir16", "blue"]:
        raw = assets_data.get(b_k)
        if raw is not None:
            offset = -0.1 if (dt_str >= "2022-01-25" or baseline.startswith("04")) else 0.0
            scale = 0.0001
            scaled = np.clip(raw * scale + offset, 0.0, 1.0)
            scaled[dil_mask] = np.nan
            scaled[np.isnan(raw) | (raw <= 0)] = np.nan
            
            v_mask = stable_mask & np.isfinite(scaled) & (scaled > 0)
            v_px = scaled[v_mask]
            v_counts.append(int(v_mask.sum()))
            b_medians[b_k] = round(float(np.nanmedian(v_px)), 4) if len(v_px) > 0 else None
        else:
            b_medians[b_k] = None
            
    return {
        "date": dt_str,
        "tile": tile_id,
        "baseline": baseline,
        "cloud": round(cloud_cover, 2),
        "valid_px": int(np.median(v_counts)) if v_counts else 0,
        "red": b_medians.get("red"),
        "nir": b_medians.get("nir"),
        "swir16": b_medians.get("swir16"),
        "blue": b_medians.get("blue"),
        "scene_id": it.id,
    }


def run_task3():
    print("\n" + "=" * 100, flush=True)
    print("TASK 3: 2022 LEGACY STABLE-PIXEL MEDIANS & 2021 LEGACY VS COLLECTION 1 COMPARISON", flush=True)
    print("=" * 100, flush=True)
    
    client = Client.open("https://earth-search.aws.element84.com/v1")
    dilation_structure = scipy.ndimage.generate_binary_structure(2, 1)

    for city in ["ahmedabad", "pune"]:
        cfg = load_city_config(city)
        bbox = cfg["spatial"]["bbox"]
        
        # Load stable mask
        classified_tif = DATA_DIR / city / f"{city}_2024_classified.tif"
        with rasterio.open(classified_tif) as src:
            city_bounds = src.bounds
            city_shape = (src.height, src.width)
            
        cls_stack = []
        for y in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
            c_p = DATA_DIR / city / f"{city}_{y}_classified.tif"
            if c_p.exists():
                with rasterio.open(c_p) as s:
                    cls_stack.append(s.read(1))
        cls_arr = np.stack(cls_stack, axis=0)
        stable_mask = ((cls_arr == 1) | (cls_arr == 3)).sum(axis=0) >= 6
        
        print(f"\n[+] {city.upper()} 2022 Legacy Scenes (sentinel-2-l2a, Jan 1 - Feb 28, 2022):", flush=True)
        search_2022 = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2022-01-01/2022-02-28",
            query={"eo:cloud_cover": {"lt": 20.0}},
        )
        items_2022 = list(search_2022.items())
        print(f"    - Found {len(items_2022)} scenes. Processing reflectance levels...", flush=True)
        
        records_2022 = []
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [
                executor.submit(process_single_legacy_scene, it, city_bounds, city_shape, stable_mask, dilation_structure)
                for it in items_2022
            ]
            for f in as_completed(futures):
                records_2022.append(f.result())
                
        records_2022.sort(key=lambda r: (r["date"], r["tile"], r["scene_id"]))
        df_2022 = pd.DataFrame(records_2022)
        print(df_2022.to_string(index=False), flush=True)
        
        # 2021 Comparison on Matching Dates
        print(f"\n[+] {city.upper()} 2021 Legacy vs Collection 1 Reflectance Comparison on Matching Dates:", flush=True)
        search_2021_leg = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2021-01-01/2021-01-31",
            query={"eo:cloud_cover": {"lt": 10.0}},
        )
        search_2021_c1 = client.search(
            collections=["sentinel-2-c1-l2a"],
            bbox=bbox,
            datetime="2021-01-01/2021-01-31",
            query={"eo:cloud_cover": {"lt": 10.0}},
        )
        items_leg = list(search_2021_leg.items())
        items_c1 = list(search_2021_c1.items())
        
        leg_dict = {f"{it.datetime.strftime('%Y-%m-%d')}_{extract_mgrs_tile(it)}": it for it in items_leg if it.datetime}
        c1_dict = {f"{it.datetime.strftime('%Y-%m-%d')}_{extract_mgrs_tile(it)}": it for it in items_c1 if it.datetime}
        
        match_keys = sorted(list(set(leg_dict.keys()) & set(c1_dict.keys())))
        comp_records = []
        
        # Process a subset of matching scenes to compare reflectance values
        for k in match_keys[:4]:
            it_leg = leg_dict[k]
            it_c1 = c1_dict[k]
            res_leg = process_single_legacy_scene(it_leg, city_bounds, city_shape, stable_mask, dilation_structure)
            res_c1 = process_single_legacy_scene(it_c1, city_bounds, city_shape, stable_mask, dilation_structure)
            
            comp_records.append({
                "Date": res_leg["date"],
                "Tile": res_leg["tile"],
                "Coll1 Baseline": res_c1["baseline"],
                "Leg Baseline": res_leg["baseline"],
                "Coll1 Red": res_c1["red"],
                "Leg Red": res_leg["red"],
                "Coll1 NIR": res_c1["nir"],
                "Leg NIR": res_leg["nir"],
                "Coll1 SWIR": res_c1["swir16"],
                "Leg SWIR": res_leg["swir16"],
                "Coll1 Blue": res_c1["blue"],
                "Leg Blue": res_leg["blue"],
            })
            
        comp_df = pd.DataFrame(comp_records)
        print(comp_df.to_string(index=False), flush=True)


# ==============================================================================
# TASK 4: Re-run Growth for 2020, 2021, 2023, 2024: Rule 2 on 4 Bands vs 3 Bands
# ==============================================================================
def run_task4():
    print("\n" + "=" * 100, flush=True)
    print("TASK 4: RE-RUN GROWTH FOR 2020, 2021, 2023, 2024 (4 BANDS VS 3 BANDS RED/NIR/SWIR)", flush=True)
    print("=" * 100, flush=True)
    
    for city in ["ahmedabad", "pune"]:
        diag_csv = DATA_DIR / city / "scene_diagnostics.csv"
        diag_df = pd.read_csv(diag_csv)
        diag_records = diag_df.to_dict("records")
        deduped_records, _ = deduplicate_tile_date_records(diag_records)
        
        # Option A: Rule 2 on all 4 bands [Red, NIR, SWIR, Blue]
        kept_4b, drop_4b, _ = apply_tile_quality_screening(
            deduped_records, rule2_bands=["red", "nir", "swir16", "blue"]
        )
        
        # Option B: Rule 2 on 3 bands only [Red, NIR, SWIR] (Blue excluded)
        kept_3b, drop_3b, _ = apply_tile_quality_screening(
            deduped_records, rule2_bands=["red", "nir", "swir16"]
        )
        
        target_years = [2020, 2021, 2023, 2024]
        
        # Load area statistics from existing class area table for reference
        area_csv = DATA_DIR / f"{city}_class_areas.csv"
        base_areas = {}
        if area_csv.exists():
            adf = pd.read_csv(area_csv)
            for _, r in adf.iterrows():
                base_areas[int(r["Year"])] = r.get("Built-up", r.get("Builtup", 0.0))
                
        growth_rows = []
        for yr in target_years:
            k4 = [r for r in kept_4b if r["year"] == yr]
            d4 = sorted(list({r["date"] for r in k4}))
            
            k3 = [r for r in kept_3b if r["year"] == yr]
            d3 = sorted(list({r["date"] for r in k3}))
            
            # Additional scenes gained when blue is relaxed
            gained = [r for r in k3 if r["scene_id"] not in {x["scene_id"] for x in k4}]
            gained_str = ", ".join([f"{r['date']} ({r['mgrs_tile']})" for r in gained]) if gained else "None"
            
            ref_builtup = base_areas.get(yr, 0.0)
            
            growth_rows.append({
                "Year": yr,
                "4-Band Kept Scenes": len(k4),
                "4-Band Distinct Dates": len(d4),
                "3-Band Kept Scenes": len(k3),
                "3-Band Distinct Dates": len(d3),
                "Scenes Recovered without Blue": len(gained),
                "Recovered Dates/Tiles": gained_str,
                "Built-up Area (km²)": round(ref_builtup, 2) if ref_builtup else "N/A",
            })
            
        print(f"\n[+] {city.upper()} Comparison: Rule 2 on 4 Bands (with Blue) vs 3 Bands (Red/NIR/SWIR only):")
        gdf = pd.DataFrame(growth_rows)
        print(gdf.to_string(index=False), flush=True)


if __name__ == "__main__":
    run_task1()
    run_task2()
    run_task3()
    run_task4()
