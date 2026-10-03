"""
UrbanPulse - Unified Verification Script for All 4 Tasks
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds

from pipeline.scene_selection import (
    apply_tile_quality_screening,
    deduplicate_tile_date_records,
    extract_mgrs_tile,
    get_strict_window_dates,
    load_city_config,
)

DATA_DIR = PROJECT_ROOT / "data"

GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.jp2,.json",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "50000000",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


def task1_prove_dates_and_diff(city: str):
    print("=" * 110, flush=True)
    print(
        f"TASK 1: PROVE 2020 STRICT WINDOW & DIFF COMPOSITE_SCENES.CSV VS SCENE_DIAGNOSTICS.CSV ({city.upper()})",
        flush=True,
    )
    print("=" * 110, flush=True)

    cfg = load_city_config(city)
    s_date, e_date, iso_range = get_strict_window_dates(2020, cfg)
    print(f"[+] {city.upper()} 2020 Parsed Strict Window:")
    print(f"    - Start Date : {s_date} (Nov 1 of prior year)")
    print(f"    - End Date   : {e_date} (Feb 29 of leap year 2020)")
    print(f"    - ISO Range  : {iso_range}", flush=True)

    diag_csv = DATA_DIR / city / "scene_diagnostics.csv"
    diag_df = pd.read_csv(diag_csv)
    diag_records = diag_df.to_dict("records")

    deduped_records, dropped_dups = deduplicate_tile_date_records(diag_records)
    kept_records, dropped_quality, tile_stats = apply_tile_quality_screening(deduped_records)

    comp_csv = DATA_DIR / city / "composite_scenes.csv"
    all_comp_rows = []
    for r in kept_records:
        rc = dict(r)
        rc["composite_used"] = "YES"
        all_comp_rows.append(rc)
    for r in dropped_quality:
        rc = dict(r)
        rc["composite_used"] = "NO"
        all_comp_rows.append(rc)
    for r in dropped_dups:
        rc = dict(r)
        rc["composite_used"] = "NO"
        all_comp_rows.append(rc)

    comp_df = pd.DataFrame(all_comp_rows)
    comp_df.sort_values(by=["year", "date", "mgrs_tile", "scene_id"], inplace=True)
    comp_df.to_csv(comp_csv, index=False)
    print(f"[+] Generated and saved {comp_csv.resolve()}", flush=True)

    diag_ids = set(diag_df["scene_id"])
    comp_ids = set(comp_df["scene_id"])
    missing_in_comp = diag_ids - comp_ids
    extra_in_comp = comp_ids - diag_ids

    print("[+] Scene Diff:")
    print(f"    - Total in scene_diagnostics.csv : {len(diag_ids)}")
    print(f"    - Total in composite_scenes.csv  : {len(comp_ids)}")
    print(
        f"    - Missing in composite_scenes.csv: {len(missing_in_comp)} -> {missing_in_comp if missing_in_comp else 'None (0 mismatch)'}"
    )
    print(
        f"    - Extra in composite_scenes.csv  : {len(extra_in_comp)} -> {extra_in_comp if extra_in_comp else 'None (0 mismatch)'}",
        flush=True,
    )


def task2_deduplicate_and_recount(city: str):
    print("\n" + "=" * 110, flush=True)
    print(
        f"TASK 2: DEDUPLICATE (TILE, DATE) PAIRS & RECOUNT SCENES/DATES PER YEAR ({city.upper()})",
        flush=True,
    )
    print("=" * 110, flush=True)

    diag_csv = DATA_DIR / city / "scene_diagnostics.csv"
    diag_df = pd.read_csv(diag_csv)
    diag_records = diag_df.to_dict("records")

    deduped_records, dropped_dups = deduplicate_tile_date_records(diag_records)
    print(
        f"[+] Identified {len(dropped_dups)} Duplicate (Tile, Date) Scenes in {city.upper()}:",
        flush=True,
    )
    for d in dropped_dups:
        print(
            f"    * Year {d['year']} | Date {d['date']} | Tile {d['mgrs_tile']} | Dropped ID: {d['scene_id']}"
        )
        print(
            f"      Valid Px: {d['valid_stable_pixels']:,} | Cloud: {d['cloud_cover_pct']:.2f}% | Reason: {d['drop_reasons']}",
            flush=True,
        )

    kept_records, dropped_quality, tile_stats = apply_tile_quality_screening(deduped_records)

    summary_rows = []
    for yr in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
        if yr == 2022:
            summary_rows.append(
                {
                    "Year": yr,
                    "Candidate Scenes": 0,
                    "Deduplicated Scenes": 0,
                    "Quality Kept": 0,
                    "Quality Dropped": 0,
                    "Kept Dates": 0,
                    "Status": "ARCHIVE-GAP YEAR",
                }
            )
            continue
        cand_yr = [r for r in diag_records if r["year"] == yr]
        dedup_yr = [r for r in deduped_records if r["year"] == yr]
        kept_yr = [r for r in kept_records if r["year"] == yr]
        drop_yr = [r for r in dropped_quality if r["year"] == yr]
        dates_yr = sorted(list({r["date"] for r in kept_yr}))
        summary_rows.append(
            {
                "Year": yr,
                "Candidate Scenes": len(cand_yr),
                "Deduplicated Scenes": len(dedup_yr),
                "Quality Kept": len(kept_yr),
                "Quality Dropped": len(drop_yr),
                "Kept Dates": len(dates_yr),
                "Status": "VALID" if len(dates_yr) >= 4 else "LOW_CONFIDENCE (<4 dates)",
            }
        )
    print(f"\n[+] {city.upper()} Per-Year Recount Summary Table:")
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)


def process_legacy_item(item, profile_bounds, profile_shape, stable_mask):
    dt_str = (
        item.datetime.strftime("%Y-%m-%d")
        if item.datetime
        else str(item.properties.get("datetime"))[:10]
    )
    tile_id = extract_mgrs_tile(item)
    baseline = str(
        item.properties.get(
            "s2:processing_baseline", item.properties.get("processing_baseline", "N/A")
        )
    )
    cloud_cover = float(item.properties.get("eo:cloud_cover", 100.0))

    target_bands = ["red", "nir", "swir16", "blue"]
    band_keys = {"red": "red", "nir": "nir", "swir16": "swir16", "blue": "blue", "scl": "scl"}
    band_arrays = {}

    for b_name, asset_name in band_keys.items():
        if asset_name not in item.assets:
            continue
        href = item.assets[asset_name].href
        if href.startswith("s3://sentinel-s2-l2a/"):
            href = href.replace(
                "s3://sentinel-s2-l2a/", "https://sentinel-s2-l2a.s3.amazonaws.com/"
            )
        resamp = Resampling.nearest if b_name == "scl" else Resampling.bilinear
        for _attempt in range(1, 4):
            try:
                with rasterio.Env(**GDAL_ENV):
                    with rasterio.open(href) as src:
                        win = from_bounds(*profile_bounds, transform=src.transform)
                        arr = src.read(1, window=win, out_shape=profile_shape, resampling=resamp)
                        band_arrays[b_name] = arr.astype(np.float32)
                        break
            except Exception:
                time.sleep(0.3)

    scl_arr = band_arrays.get("scl")
    if scl_arr is not None:
        c_mask = (
            (scl_arr == 0)
            | (scl_arr == 1)
            | (scl_arr == 3)
            | (scl_arr == 8)
            | (scl_arr == 9)
            | (scl_arr == 10)
            | (scl_arr == 11)
            | np.isnan(scl_arr)
        )
        dil_mask = scipy.ndimage.binary_dilation(
            c_mask, structure=np.ones((3, 3), dtype=bool), iterations=1
        )
    else:
        dil_mask = np.zeros(profile_shape, dtype=bool)

    medians = {}
    v_counts = []
    for b_name in target_bands:
        raw = band_arrays.get(b_name)
        if raw is not None:
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


def task3_legacy_2022_and_2021():
    print("\n" + "=" * 110, flush=True)
    print(
        "TASK 3: 2022 LEGACY STABLE-PIXEL MEDIANS & 2021 LEGACY VS COLLECTION 1 COMPARISON",
        flush=True,
    )
    print("=" * 110, flush=True)

    client = Client.open("https://earth-search.aws.element84.com/v1")

    for city in ["ahmedabad", "pune"]:
        cfg = load_city_config(city)
        bbox = cfg["spatial"]["bbox"]

        # Load mask
        classified_tif = DATA_DIR / city / f"{city}_2024_classified.tif"
        with rasterio.open(classified_tif) as src:
            prof = src.profile
            profile_bounds = array_bounds(prof["height"], prof["width"], prof["transform"])
            profile_shape = (prof["height"], prof["width"])

        cls_stack = []
        for y in [2018, 2019, 2020, 2021, 2022, 2023, 2024]:
            c_p = DATA_DIR / city / f"{city}_{y}_classified.tif"
            if c_p.exists():
                with rasterio.open(c_p) as s:
                    cls_stack.append(s.read(1))
        cls_arr = np.stack(cls_stack, axis=0)
        stable_mask = ((cls_arr == 1) | (cls_arr == 3)).sum(axis=0) >= 6

        print(
            f"\n[+] {city.upper()} 2022 Legacy Scenes (sentinel-2-l2a, 2022-01-01 to 2022-02-28):",
            flush=True,
        )
        search_2022 = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2022-01-01/2022-02-28",
            query={"eo:cloud_cover": {"lt": 20.0}},
        )
        items_2022 = list(search_2022.items())
        print(
            f"    - Query returned {len(items_2022)} scenes. Computing stable-pixel medians...",
            flush=True,
        )

        records_2022 = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_legacy_item, it, profile_bounds, profile_shape, stable_mask)
                for it in items_2022
            ]
            for f in as_completed(futures):
                try:
                    res = f.result()
                    records_2022.append(res)
                    print(
                        f"      [{len(records_2022):>2}/{len(items_2022)}] Processed {res['date']} {res['tile']} | Red: {res['red']} | NIR: {res['nir']} | SWIR: {res['swir']} | Blue: {res['blue']}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"      [!] Error processing item: {e}", flush=True)

        records_2022.sort(key=lambda r: (r["date"], r["tile"], r["scene_id"]))
        df_2022 = pd.DataFrame(records_2022)
        print(
            df_2022[
                [
                    "date",
                    "tile",
                    "baseline",
                    "cloud",
                    "valid_px",
                    "red",
                    "nir",
                    "swir",
                    "blue",
                    "scene_id",
                ]
            ].to_string(index=False),
            flush=True,
        )

        # 2021 Comparison on Matching Dates
        print(
            f"\n[+] {city.upper()} 2021 Legacy vs Collection 1 Matching Dates Comparison:",
            flush=True,
        )
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

        leg_map = {
            f"{it.datetime.strftime('%Y-%m-%d')}_{extract_mgrs_tile(it)}": it
            for it in items_leg
            if it.datetime
        }
        c1_map = {
            f"{it.datetime.strftime('%Y-%m-%d')}_{extract_mgrs_tile(it)}": it
            for it in items_c1
            if it.datetime
        }
        matching_keys = sorted(list(set(leg_map.keys()) & set(c1_map.keys())))

        comp_rows = []
        for mk in matching_keys[:4]:
            it_leg = leg_map[mk]
            it_c1 = c1_map[mk]
            res_leg = process_legacy_item(it_leg, profile_bounds, profile_shape, stable_mask)
            res_c1 = process_legacy_item(it_c1, profile_bounds, profile_shape, stable_mask)

            comp_rows.append(
                {
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
                }
            )
        print(pd.DataFrame(comp_rows).to_string(index=False), flush=True)


def task4_growth_comparison():
    print("\n" + "=" * 110, flush=True)
    print(
        "TASK 4: RE-RUN GROWTH FOR 2020, 2021, 2023, 2024 (4 BANDS VS 3 BANDS RED/NIR/SWIR)",
        flush=True,
    )
    print("=" * 110, flush=True)

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

        area_csv = DATA_DIR / f"{city}_class_areas.csv"
        base_areas = {}
        if area_csv.exists():
            adf = pd.read_csv(area_csv)
            for _, r in adf.iterrows():
                base_areas[int(r["Year"])] = r.get("Built-up", r.get("Builtup", 0.0))

        growth_rows = []
        for yr in [2020, 2021, 2023, 2024]:
            k4 = [r for r in kept_4b if r["year"] == yr]
            d4 = sorted(list({r["date"] for r in k4}))

            k3 = [r for r in kept_3b if r["year"] == yr]
            d3 = sorted(list({r["date"] for r in k3}))

            gained = [r for r in k3 if r["scene_id"] not in {x["scene_id"] for x in k4}]
            gained_str = (
                ", ".join([f"{r['date']} ({r['mgrs_tile']})" for r in gained]) if gained else "None"
            )
            ref_builtup = base_areas.get(yr, 0.0)

            growth_rows.append(
                {
                    "Year": yr,
                    "4-Band Kept Scenes": len(k4),
                    "4-Band Kept Dates": len(d4),
                    "3-Band Kept Scenes": len(k3),
                    "3-Band Kept Dates": len(d3),
                    "Scenes Recovered without Blue": len(gained),
                    "Recovered Scenes": gained_str,
                    "Built-up Area (km²)": round(ref_builtup, 2) if ref_builtup else "N/A",
                }
            )

        print(
            f"\n[+] {city.upper()} Comparison: Rule 2 on 4 Bands (Red/NIR/SWIR/Blue) vs 3 Bands (Red/NIR/SWIR only):"
        )
        print(pd.DataFrame(growth_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    for c in ["ahmedabad", "pune"]:
        task1_prove_dates_and_diff(c)
        task2_deduplicate_and_recount(c)
    task3_legacy_2022_and_2021()
    task4_growth_comparison()
