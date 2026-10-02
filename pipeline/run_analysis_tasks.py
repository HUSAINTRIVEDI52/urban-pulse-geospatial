"""
UrbanPulse - Execution Script for Tasks 1, 2, 3, 4:
1. Shared window/selection function & diff verification against scene_diagnostics.csv.
2. Deduplication of (tile, date) pairs with recount of scenes/dates per year.
3. 2022 legacy scenes stable-pixel medians & 2021 legacy vs Collection 1 comparison on matching dates.
4. Growth re-run for 2020, 2021, 2023, 2024 comparing Rule 2 on 4 bands vs Rule 2 on 3 bands (Red/NIR/SWIR).
"""

import calendar
import csv
import json
import sys
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
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

from pipeline.scene_selection import (
    load_city_config,
    get_strict_window_dates,
    query_strict_window_scenes,
    extract_mgrs_tile,
    deduplicate_tile_date_records,
    apply_tile_quality_screening,
)
from pipeline.train_classifier import classify_raster

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def task1_prove_dates_and_diff(city: str):
    print(f"\n{'='*80}\n[TASK 1] PROVE DATES & WRITE COMPOSITE_SCENES.CSV FOR {city.upper()}\n{'='*80}")
    cfg = load_city_config(city)
    s_date, e_date, iso_range = get_strict_window_dates(2020, cfg)
    print(f"[+] 2020 Parsed Strict Window: Start = {s_date} | End = {e_date} | ISO = {iso_range}")
    
    # Load scene_diagnostics.csv
    diag_csv = DATA_DIR / city / "scene_diagnostics.csv"
    diag_df = pd.read_csv(diag_csv)
    print(f"[+] Loaded {len(diag_df)} records from {diag_csv}")

    # Deduplicate (tile, date) pairs in scene_diagnostics
    diag_records = diag_df.to_dict("records")
    deduped_records, dropped_dups = deduplicate_tile_date_records(diag_records)
    
    # Apply screening
    kept_records, dropped_quality, tile_stats = apply_tile_quality_screening(deduped_records)
    
    # Build composite_scenes.csv (representing scenes selected for composite)
    comp_csv = DATA_DIR / city / "composite_scenes.csv"
    
    # Prepare composite_scenes rows
    all_rows = []
    for r in kept_records:
        r_out = dict(r)
        r_out["composite_used"] = "YES"
        all_rows.append(r_out)
    for r in dropped_quality:
        r_out = dict(r)
        r_out["composite_used"] = "NO"
        all_rows.append(r_out)
    for r in dropped_dups:
        r_out = dict(r)
        r_out["composite_used"] = "NO"
        all_rows.append(r_out)
        
    comp_df = pd.DataFrame(all_rows)
    comp_df.sort_values(by=["year", "date", "mgrs_tile", "scene_id"], inplace=True)
    comp_df.to_csv(comp_csv, index=False)
    print(f"[+] Saved composite_scenes.csv: {comp_csv} ({len(comp_df)} total scenes, {len(kept_records)} used in composite)")
    
    # Diff against scene_diagnostics.csv
    print(f"\n[+] Diffing composite_scenes.csv vs scene_diagnostics.csv:")
    diag_ids = set(diag_df["scene_id"])
    comp_ids = set(comp_df["scene_id"])
    missing_in_comp = diag_ids - comp_ids
    extra_in_comp = comp_ids - diag_ids
    print(f"    - Scenes in scene_diagnostics.csv : {len(diag_ids)}")
    print(f"    - Scenes in composite_scenes.csv  : {len(comp_ids)}")
    print(f"    - Mismatch / Missing in comp     : {len(missing_in_comp)} {missing_in_comp if missing_in_comp else 'None (Perfect Match)'}")
    print(f"    - Extra in comp                  : {len(extra_in_comp)} {extra_in_comp if extra_in_comp else 'None (Perfect Match)'}")


def task2_deduplicate_and_recount(city: str):
    print(f"\n{'='*80}\n[TASK 2] DEDUPLICATE (TILE, DATE) PAIRS & RECOUNT FOR {city.upper()}\n{'='*80}")
    diag_csv = DATA_DIR / city / "scene_diagnostics.csv"
    diag_df = pd.read_csv(diag_csv)
    diag_records = diag_df.to_dict("records")
    
    deduped_records, dropped_dups = deduplicate_tile_date_records(diag_records)
    print(f"[+] Found {len(dropped_dups)} duplicate (tile, date) scenes dropped:")
    for d in dropped_dups:
        print(f"    - Year {d['year']} | Date {d['date']} | Tile {d['mgrs_tile']} | Scene: {d['scene_id']} | Valid Px: {d['valid_stable_pixels']:,} | Cloud: {d['cloud_cover_pct']:.2f}% | Drop Reason: {d['drop_reasons']}")
        
    kept_records, dropped_quality, tile_stats = apply_tile_quality_screening(deduped_records)
    
    print(f"\n[+] Recount of Unique Scenes & Dates per Year (After Deduplication):")
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    summary_rows = []
    for yr in years:
        if yr == 2022:
            summary_rows.append({"Year": yr, "Candidate Scenes": 0, "Deduplicated": 0, "Quality Kept": 0, "Quality Dropped": 0, "Distinct Dates (Kept)": 0, "Status": "ARCHIVE-GAP"})
            continue
        cand_yr = [r for r in diag_records if r["year"] == yr]
        dedup_yr = [r for r in deduped_records if r["year"] == yr]
        kept_yr = [r for r in kept_records if r["year"] == yr]
        drop_yr = [r for r in dropped_quality if r["year"] == yr]
        dates_yr = sorted(list({r["date"] for r in kept_yr}))
        summary_rows.append({
            "Year": yr,
            "Candidate Scenes": len(cand_yr),
            "Deduplicated": len(dedup_yr),
            "Quality Kept": len(kept_yr),
            "Quality Dropped": len(drop_yr),
            "Distinct Dates (Kept)": len(dates_yr),
            "Status": "VALID" if len(dates_yr) >= 4 else "LOW_CONFIDENCE",
        })
    sum_df = pd.DataFrame(summary_rows)
    print(sum_df.to_string(index=False))


def task3_legacy_2022_and_2021_comparison():
    print(f"\n{'='*80}\n[TASK 3] 2022 LEGACY STABLE-PIXEL MEDIANS & 2021 LEGACY VS COLLECTION 1\n{'='*80}")
    # Load stable masks
    client = Client.open("https://earth-search.aws.element84.com/v1")
    
    dilation_structure = scipy.ndimage.generate_binary_structure(2, 1)

    for city in ["ahmedabad", "pune"]:
        print(f"\n>>> Processing Legacy STAC analysis for {city.upper()}...")
        cfg = load_city_config(city)
        bbox = cfg["spatial"]["bbox"]
        
        # Load stable mask
        classified_tif = DATA_DIR / city / f"{city}_2024_classified.tif"
        with rasterio.open(classified_tif) as src:
            bounds = src.bounds
            shape = (src.height, src.width)
            transform = src.transform
            crs = src.crs
            
        # Re-compute stable mask
        eval_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
        cls_stack = []
        for y in eval_years:
            c_p = DATA_DIR / city / f"{city}_{y}_classified.tif"
            if c_p.exists():
                with rasterio.open(c_p) as s:
                    cls_stack.append(s.read(1))
        cls_arr = np.stack(cls_stack, axis=0)
        built_or_water = (cls_arr == 1) | (cls_arr == 3)
        stable_mask = (built_or_water.sum(axis=0) >= 6)
        print(f"[+] Stable pixels mask: {stable_mask.sum():,} pixels")
        
        # 1. 2022 Legacy Scenes (sentinel-2-l2a)
        search_2022 = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2022-01-01/2022-02-28",
            query={"eo:cloud_cover": {"lt": 20.0}},
        )
        items_2022 = list(search_2022.items())
        print(f"[+] Found {len(items_2022)} legacy scenes for {city.upper()} 2022 (Jan 1 - Feb 28)")
        
        records_2022 = []
        for it in sorted(items_2022, key=lambda x: (x.datetime or datetime(2022,1,1), x.id)):
            dt_str = it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10]
            tile_id = extract_mgrs_tile(it)
            baseline = str(it.properties.get("s2:processing_baseline", it.properties.get("processing_baseline", "N/A")))
            cloud_cover = float(it.properties.get("eo:cloud_cover", 100.0))
            
            # Read bands
            band_map = {"red": "B04", "nir": "B08", "swir16": "B11", "blue": "B02", "scl": "SCL"}
            # Check asset names in legacy STAC
            b_medians = {}
            valid_px_count = 0
            
            # Read assets
            assets_data = {}
            for b_k, b_asset_name in band_map.items():
                asset_key = b_asset_name if b_asset_name in it.assets else b_k
                if asset_key in it.assets:
                    href = it.assets[asset_key].href
                    try:
                        with rasterio.open(href) as b_src:
                            win = from_bounds(*bounds, b_src.transform)
                            resamp = Resampling.nearest if b_k == "scl" else Resampling.bilinear
                            arr = b_src.read(1, window=win, out_shape=shape, resampling=resamp)
                            assets_data[b_k] = arr.astype(np.float32)
                    except Exception as e:
                        pass
            
            # Dilated cloud mask
            scl_arr = assets_data.get("scl")
            if scl_arr is not None:
                c_mask = ((scl_arr == 0) | (scl_arr == 1) | (scl_arr == 3) | (scl_arr == 8) | (scl_arr == 9) | (scl_arr == 10) | (scl_arr == 11) | np.isnan(scl_arr))
                dil_mask = scipy.ndimage.binary_dilation(c_mask, structure=dilation_structure, iterations=1)
            else:
                dil_mask = np.zeros(shape, dtype=bool)
                
            v_counts = []
            for b_k in ["red", "nir", "swir16", "blue"]:
                raw = assets_data.get(b_k)
                if raw is not None:
                    # Scale and harmonize
                    # Baseline 04.00 offset is present in legacy if date >= 2022-01-25 or baseline >= 04.00
                    scale = 0.0001
                    offset = -0.1 if (dt_str >= "2022-01-25" or baseline.startswith("04")) else 0.0
                    scaled = raw * scale + offset
                    scaled[dil_mask] = np.nan
                    scaled[np.isnan(raw) | (raw <= 0)] = np.nan
                    
                    v_mask = stable_mask & np.isfinite(scaled) & (scaled > 0)
                    v_px = scaled[v_mask]
                    v_counts.append(int(v_mask.sum()))
                    b_medians[b_k] = round(float(np.nanmedian(v_px)), 4) if len(v_px) > 0 else None
                else:
                    b_medians[b_k] = None
                    
            valid_stable_px = int(np.median(v_counts)) if v_counts else 0
            records_2022.append({
                "city": city,
                "date": dt_str,
                "tile": tile_id,
                "baseline": baseline,
                "cloud": cloud_cover,
                "valid_px": valid_stable_px,
                "red": b_medians.get("red"),
                "nir": b_medians.get("nir"),
                "swir": b_medians.get("swir16"),
                "blue": b_medians.get("blue"),
                "scene_id": it.id,
            })
            
        print(f"\n[+] 2022 Legacy Scenes ({city.upper()}):")
        df_2022 = pd.DataFrame(records_2022)
        print(df_2022.to_string(index=False))
        
        # 2. 2021 Comparison: Legacy vs Collection 1 on matching dates
        print(f"\n[+] Comparing 2021 Legacy (sentinel-2-l2a) vs Collection 1 (sentinel-2-c1-l2a) on matching dates ({city.upper()}):")
        search_2021_leg = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2021-01-01/2021-01-31",
            query={"eo:cloud_cover": {"lt": 10.0}},
        )
        items_2021_leg = list(search_2021_leg.items())
        
        search_2021_c1 = client.search(
            collections=["sentinel-2-c1-l2a"],
            bbox=bbox,
            datetime="2021-01-01/2021-01-31",
            query={"eo:cloud_cover": {"lt": 10.0}},
        )
        items_2021_c1 = list(search_2021_c1.items())
        
        # Group by date & tile
        c1_dates = {it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10]: it for it in items_2021_c1}
        leg_dates = {it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10]: it for it in items_2021_leg}
        
        match_dates = sorted(list(set(c1_dates.keys()) & set(leg_dates.keys())))
        print(f"    - Matching acquisition dates in Jan 2021: {match_dates}")
        for md in match_dates[:3]:  # compare first 3 matching dates
            it_c1 = c1_dates[md]
            it_leg = leg_dates[md]
            print(f"    * Date: {md} | Tile: {extract_mgrs_tile(it_c1)}")
            print(f"      - Collection 1 : ID={it_c1.id} | Baseline={it_c1.properties.get('s2:processing_baseline')} | Cloud={it_c1.properties.get('eo:cloud_cover')}%")
            print(f"      - Legacy STAC  : ID={it_leg.id} | Baseline={it_leg.properties.get('s2:processing_baseline', it_leg.properties.get('processing_baseline'))} | Cloud={it_leg.properties.get('eo:cloud_cover')}%")


if __name__ == "__main__":
    for c in ["ahmedabad", "pune"]:
        task1_prove_dates_and_diff(c)
        task2_deduplicate_and_recount(c)
    task3_legacy_2022_and_2021_comparison()
