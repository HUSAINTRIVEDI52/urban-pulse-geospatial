"""
UrbanPulse - Sentinel-2 10m True Colour Chip Generator for Visual Change Validation
Builds 10 m true colour (RGB) composites for start and end years (e.g. 2020 and 2024)
from Sentinel-2 L2A via STAC, using Dec-Feb strict window scenes and SCL cloud mask,
applies a fixed 2%-98% percentile stretch, cuts 128x128 chips centered on validation points,
upscales 4x with bicubic resampling, and draws the 60m classifier pixel boundary.

Outputs:
  - data/{city}/validation/chips/{id}_start.jpg
  - data/{city}/validation/chips/{id}_end.jpg
  - data/{city}/validation/labeller.html (Interactive standalone visual labelling application)
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401
import scipy.ndimage
import stackstac
import xarray as xr
import yaml
from dask.diagnostics import ProgressBar
from PIL import Image, ImageDraw
from pyproj import Transformer
from pystac_client import Client
from rasterio.transform import Affine

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.build_composite import (
    load_city_config,
    scale_and_harmonize_dn,
)


def load_scene_list_for_year(city: str, year: int, data_dir: Path) -> list[str] | None:
    """
    Loads kept scene IDs for a given year from scene_diagnostics or composite_scenes.
    """
    city_key = city.lower()
    candidates = [
        data_dir / city_key / "composite_scenes.csv",
        data_dir / city_key / "scene_diagnostics.csv",
        data_dir / city_key / f"{city_key}_scene_diagnostics.csv",
        data_dir / "scene_diagnostics.csv",
    ]
    for cp in candidates:
        if cp.exists():
            try:
                df = pd.read_csv(cp)
                if "year" in df.columns and "scene_id" in df.columns:
                    df_yr = df[df["year"] == year]
                    if "composite_used" in df_yr.columns:
                        kept = df_yr[df_yr["composite_used"] == "YES"]["scene_id"].tolist()
                        if kept:
                            return kept
                    if "status" in df_yr.columns:
                        kept = df_yr[df_yr["status"] == "KEPT"]["scene_id"].tolist()
                        if kept:
                            return kept
            except Exception:
                pass
    return None


def fetch_and_composite_10m_rgb(
    city: str,
    year: int,
    config: dict[str, Any],
    data_dir: Path,
    cache_dir: Path | None = None,
    max_scenes_per_tile: int = 8,
) -> tuple[np.ndarray, Affine, Any]:
    """
    Builds a 10m true colour (B04, B03, B02) surface reflectance composite for the city AOI.
    Processes tile-by-tile in chunked Dask streaming to keep memory footprint low (< 200MB).

    Returns:
        (rgb_float_array (3, H, W), transform, crs)
    """
    city_key = city.lower()
    if cache_dir is None:
        cache_dir = data_dir / city_key / "validation" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_tif = cache_dir / f"{city_key}_rgb_10m_{year}.tif"

    # Check cache
    if cached_tif.exists():
        print(f"[+] Found cached 10m RGB composite for {city.capitalize()} ({year}): {cached_tif.name}")
        with rasterio.open(cached_tif) as src:
            rgb_arr = src.read()  # (3, H, W)
            transform = src.transform
            crs = src.crs
            return rgb_arr, transform, crs

    bbox = config["spatial"]["bbox"]
    stac_url = config.get("stac", {}).get(
        "earth_search_url", "https://earth-search.aws.element84.com/v1"
    )
    collection = (
        config.get("stac", {}).get("collections", {}).get("sentinel_2", "sentinel-2-c1-l2a")
    )

    client = Client.open(stac_url)
    kept_scene_ids = load_scene_list_for_year(city, year, data_dir)

    if kept_scene_ids:
        print(f"[*] Loading {len(kept_scene_ids)} validated scene IDs from diagnostics for {year}...")
        search = client.search(
            collections=[collection],
            ids=kept_scene_ids,
        )
        items = list(search.items())
    else:
        # Fallback to 8 least cloudy scenes in Dec 1 - Feb 15
        date_range = f"{year - 1}-12-01/{year}-02-15"
        print(f"[*] Querying STAC for {year} in strict window ({date_range})...")
        search = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": 20.0}},
        )
        all_items = list(search.items())
        if not all_items:
            search = client.search(
                collections=[collection],
                bbox=bbox,
                datetime=date_range,
                query={"eo:cloud_cover": {"lt": 35.0}},
            )
            all_items = list(search.items())

        # Group by MGRS tile and sort by cloud cover
        tile_dict: dict[str, list[Any]] = {}
        for it in all_items:
            z = str(it.properties.get("mgrs:utm_zone", ""))
            b = str(it.properties.get("mgrs:latitude_band", ""))
            g = str(it.properties.get("mgrs:grid_square", ""))
            tile_id = f"{z}{b}{g}" if (z and b and g) else it.id.split("_")[1].replace("T", "")
            tile_dict.setdefault(tile_id, []).append(it)

        items = []
        for t_id, t_items in tile_dict.items():
            t_items_sorted = sorted(t_items, key=lambda x: float(x.properties.get("eo:cloud_cover", 100.0)))
            items.extend(t_items_sorted[:max_scenes_per_tile])

    if not items:
        raise RuntimeError(f"No Sentinel-2 scenes found for {city} {year}.")

    # Group items by MGRS tile to process one tile at a time
    mgrs_groups: dict[str, list[Any]] = {}
    for it in items:
        z = str(it.properties.get("mgrs:utm_zone", ""))
        b = str(it.properties.get("mgrs:latitude_band", ""))
        g = str(it.properties.get("mgrs:grid_square", ""))
        tile_id = f"{z}{b}{g}" if (z and b and g) else it.id.split("_")[1].replace("T", "")
        mgrs_groups.setdefault(tile_id, []).append(it)

    print(f"[+] Processing {len(items)} scenes across {len(mgrs_groups)} MGRS tiles at 10m resolution (EPSG:32643)...")

    full_rgb = None
    ref_transform = None
    ref_crs = None
    requested_assets = ["red", "green", "blue", "scl"]

    for tile_idx, (t_id, t_scenes) in enumerate(mgrs_groups.items(), start=1):
        print(f"\n  --> [{tile_idx}/{len(mgrs_groups)}] Streaming MGRS Tile {t_id} ({len(t_scenes)} scenes)...")

        if len(t_scenes) > max_scenes_per_tile:
            t_scenes = t_scenes[:max_scenes_per_tile]

        stack = stackstac.stack(
            t_scenes,
            assets=requested_assets,
            epsg=32643,
            bounds_latlon=bbox,
            resolution=10.0,
            chunksize=1024,
            rescale=False,
            fill_value=np.nan,
        )

        if ref_transform is None:
            ref_transform = stack.rio.transform()
            ref_crs = stack.rio.crs

        # Dask-native SCL cloud masking & scaling
        scl = stack.sel(band="scl")
        cloud_mask = (
            (scl == 0)
            | (scl == 1)
            | (scl == 3)
            | (scl == 8)
            | (scl == 9)
            | (scl == 10)
            | (scl == 11)
            | np.isnan(scl)
        )

        r = stack.sel(band="red").where(~cloud_mask)
        g = stack.sel(band="green").where(~cloud_mask)
        b = stack.sel(band="blue").where(~cloud_mask)

        # Baseline 04.00 offset / scaling
        if year >= 2022:
            r = ((r - 1000.0) / 10000.0).clip(0.0, 1.0)
            g = ((g - 1000.0) / 10000.0).clip(0.0, 1.0)
            b = ((b - 1000.0) / 10000.0).clip(0.0, 1.0)
        else:
            r = (r / 10000.0).clip(0.0, 1.0)
            g = (g / 10000.0).clip(0.0, 1.0)
            b = (b / 10000.0).clip(0.0, 1.0)

        # Compute median across scenes in Dask chunk-by-chunk (low memory)
        r_med = r.median(dim="time")
        g_med = g.median(dim="time")
        b_med = b.median(dim="time")

        tile_rgb_dask = xr.concat([r_med, g_med, b_med], dim="band")

        with rasterio.Env(
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        ):
            with ProgressBar(minimum=0.2):
                tile_median = tile_rgb_dask.compute().values.astype(np.float32)

        if full_rgb is None:
            full_rgb = np.full_like(tile_median, np.nan)

        valid_tile_px = np.isfinite(tile_median)
        full_rgb = np.where(valid_tile_px, tile_median, full_rgb)

    # Save to cache GeoTIFF
    print(f"\n[+] Saving 10m RGB composite to cache: {cached_tif.resolve()}...")
    with rasterio.open(
        cached_tif,
        "w",
        driver="GTiff",
        height=full_rgb.shape[1],
        width=full_rgb.shape[2],
        count=3,
        dtype=np.float32,
        crs=ref_crs,
        transform=ref_transform,
        compress="lzw",
        nodata=np.nan,
    ) as dst:
        dst.write(full_rgb)

    return full_rgb, ref_transform, ref_crs


def extract_and_draw_chip(
    rgb_uint8: np.ndarray,
    center_row: int,
    center_col: int,
    chip_size_px: int = 128,
    upscale_factor: int = 4,
    pixel_box_size_orig_px: int = 6,
) -> Image.Image:
    """
    Extracts a chip_size_px x chip_size_px window from an RGB image, upscales it by upscale_factor
    using bicubic interpolation, and draws a crisp outline of the original pixel at the center.
    """
    h, w, c = rgb_uint8.shape
    half = chip_size_px // 2

    r_min = center_row - half
    r_max = center_row + half
    c_min = center_col - half
    c_max = center_col + half

    # Extract with border padding if near boundary
    chip = np.zeros((chip_size_px, chip_size_px, 3), dtype=np.uint8)

    src_r_min = max(0, r_min)
    src_r_max = min(h, r_max)
    src_c_min = max(0, c_min)
    src_c_max = min(w, c_max)

    dst_r_min = src_r_min - r_min
    dst_r_max = dst_r_min + (src_r_max - src_r_min)
    dst_c_min = src_c_min - c_min
    dst_c_max = dst_c_min + (src_c_max - src_c_min)

    if src_r_max > src_r_min and src_c_max > src_c_min:
        chip[dst_r_min:dst_r_max, dst_c_min:dst_c_max] = rgb_uint8[src_r_min:src_r_max, src_c_min:src_c_max]

    # Upscale 4x with bicubic resampling
    pil_chip = Image.fromarray(chip)
    out_dim = chip_size_px * upscale_factor
    upscaled = pil_chip.resize((out_dim, out_dim), resample=Image.Resampling.BICUBIC)

    # Draw thin outline of the 60m pixel at center (24x24 px at 4x upscale)
    draw = ImageDraw.Draw(upscaled)
    center_out = out_dim // 2
    half_box = (pixel_box_size_orig_px * upscale_factor) // 2

    box_r_min = center_out - half_box
    box_r_max = center_out + half_box
    box_c_min = center_out - half_box
    box_c_max = center_out + half_box

    # Draw bright yellow outline with thin 2px width
    draw.rectangle(
        [box_c_min, box_r_min, box_c_max, box_r_max],
        outline=(255, 235, 59),
        width=2,
    )

    return upscaled


def generate_standalone_labeller_html(
    city_name: str,
    start_year: int,
    end_year: int,
    points_data: list[dict[str, Any]],
    output_html_path: Path,
) -> None:
    """
    Generates a single self-contained interactive labelling tool (no external CDN / internet dependencies).
    """
    points_json = json.dumps(points_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>UrbanPulse Blind Change Validation Labeller - {city_name} ({start_year} -> {end_year})</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --built-btn: #10b981;
      --notbuilt-btn: #ef4444;
      --unclear-btn: #f59e0b;
      --highlight-box: #facc15;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; }}
    header {{ background: #0b1120; border-bottom: 1px solid var(--border); padding: 12px 24px; display: flex; justify-content: space-between; align-items: center; }}
    .header-title {{ font-size: 1.15rem; font-weight: 700; color: var(--accent); }}
    .progress-bar-container {{ width: 280px; background: var(--border); height: 10px; border-radius: 5px; overflow: hidden; margin-top: 4px; }}
    .progress-bar {{ background: var(--built-btn); height: 100%; width: 0%; transition: width 0.2s ease; }}
    .stats-badge {{ font-size: 0.85rem; color: var(--text-muted); }}

    main {{ flex: 1; display: flex; flex-direction: column; align-items: center; padding: 18px 24px; max-width: 1200px; margin: 0 auto; width: 100%; }}
    
    .nav-bar {{ width: 100%; display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }}
    .point-id-badge {{ font-size: 1.3rem; font-weight: 800; color: #fff; background: var(--card-bg); padding: 6px 16px; border-radius: 8px; border: 1px solid var(--border); }}
    .btn {{ background: var(--card-bg); color: var(--text); border: 1px solid var(--border); padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; transition: all 0.15s ease; text-decoration: none; }}
    .btn:hover {{ background: #334155; border-color: var(--accent); }}
    .btn-accent {{ background: #0284c7; color: white; border: none; }}
    .btn-accent:hover {{ background: #0369a1; }}
    .btn-success {{ background: #059669; color: white; border: none; }}
    .btn-success:hover {{ background: #047857; }}

    .chips-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; width: 100%; margin-bottom: 18px; }}
    .chip-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 16px; display: flex; flex-direction: column; align-items: center; }}
    .chip-header {{ width: 100%; display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-weight: 700; font-size: 1.05rem; }}
    .chip-img-wrapper {{ width: 512px; height: 512px; max-width: 100%; aspect-ratio: 1/1; background: #000; border-radius: 8px; overflow: hidden; border: 2px solid var(--border); position: relative; }}
    .chip-img-wrapper img {{ width: 100%; height: 100%; object-fit: contain; image-rendering: auto; }}
    
    .button-group {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; width: 100%; margin-top: 14px; }}
    .choice-btn {{ padding: 12px 8px; border-radius: 8px; font-size: 0.95rem; font-weight: 700; border: 2px solid transparent; cursor: pointer; background: #0f172a; color: var(--text-muted); transition: all 0.15s ease; text-align: center; }}
    .choice-btn:hover {{ border-color: var(--text-muted); color: #fff; }}
    
    .choice-btn.built.active {{ background: rgba(16, 185, 129, 0.2); border-color: var(--built-btn); color: #34d399; }}
    .choice-btn.notbuilt.active {{ background: rgba(239, 68, 68, 0.2); border-color: var(--notbuilt-btn); color: #f87171; }}
    .choice-btn.unclear.active {{ background: rgba(245, 158, 11, 0.2); border-color: var(--unclear-btn); color: #fbbf24; }}

    .notes-section {{ width: 100%; background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; display: flex; gap: 12px; align-items: center; margin-bottom: 16px; }}
    .notes-input {{ flex: 1; background: #0f172a; border: 1px solid var(--border); color: #fff; padding: 8px 12px; border-radius: 6px; font-size: 0.9rem; outline: none; }}
    .notes-input:focus {{ border-color: var(--accent); }}

    .shortcuts-help {{ font-size: 0.82rem; color: var(--text-muted); text-align: center; line-height: 1.5; }}
    .kbd {{ background: #334155; color: #fff; padding: 2px 6px; border-radius: 4px; font-weight: 700; font-size: 0.75rem; border: 1px solid #475569; }}
  </style>
</head>
<body>
  <header>
    <div>
      <div class="header-title">UrbanPulse Change Validation Labeller &mdash; {city_name}</div>
      <div class="stats-badge" id="answeredStats">Answered: 0 / 0 (0%)</div>
    </div>
    <div>
      <div class="progress-bar-container">
        <div class="progress-bar" id="progressBar"></div>
      </div>
    </div>
    <div style="display: flex; gap: 10px;">
      <button class="btn" onclick="jumpToFirstUnlabelled()">Jump to Unlabelled</button>
      <button class="btn btn-success" onclick="downloadCSV()">Download CSV</button>
    </div>
  </header>

  <main>
    <div class="nav-bar">
      <button class="btn" onclick="prevPoint()">&larr; Previous (Left Arrow)</button>
      <div style="display: flex; align-items: center; gap: 14px;">
        <span class="point-id-badge" id="pointBadge">Point #1</span>
        <a id="gmapsLink" class="btn" target="_blank" href="#">Open in Google Maps &nearr;</a>
      </div>
      <button class="btn" onclick="nextPoint()">Next (Right Arrow) &rarr;</button>
    </div>

    <div class="chips-container">
      <!-- Start Year Card -->
      <div class="chip-card">
        <div class="chip-header">
          <span>Start Year ({start_year})</span>
          <span style="font-size: 0.85rem; color: var(--text-muted);">Shortcuts: Q / W / E</span>
        </div>
        <div class="chip-img-wrapper">
          <img id="imgStart" src="" alt="Start Year Chip">
        </div>
        <div class="button-group">
          <button class="choice-btn built" id="btnStartBuilt" onclick="setLabel('start', 'Y')">Built (Q)</button>
          <button class="choice-btn notbuilt" id="btnStartNotBuilt" onclick="setLabel('start', 'N')">Not Built (W)</button>
          <button class="choice-btn unclear" id="btnStartUnclear" onclick="setLabel('start', 'unclear')">Unclear (E)</button>
        </div>
      </div>

      <!-- End Year Card -->
      <div class="chip-card">
        <div class="chip-header">
          <span>End Year ({end_year})</span>
          <span style="font-size: 0.85rem; color: var(--text-muted);">Shortcuts: I / O / P</span>
        </div>
        <div class="chip-img-wrapper">
          <img id="imgEnd" src="" alt="End Year Chip">
        </div>
        <div class="button-group">
          <button class="choice-btn built" id="btnEndBuilt" onclick="setLabel('end', 'Y')">Built (I)</button>
          <button class="choice-btn notbuilt" id="btnEndNotBuilt" onclick="setLabel('end', 'N')">Not Built (O)</button>
          <button class="choice-btn unclear" id="btnEndUnclear" onclick="setLabel('end', 'unclear')">Unclear (P)</button>
        </div>
      </div>
    </div>

    <div class="notes-section">
      <span style="font-size: 0.9rem; font-weight: 600; color: var(--text-muted);">Notes (S):</span>
      <input type="text" id="notesInput" class="notes-input" placeholder="Optional notes for this sample point..." onchange="saveNote()">
    </div>

    <div class="shortcuts-help">
      <b>Keyboard Shortcuts:</b> 
      <span class="kbd">&larr;</span> / <span class="kbd">&rarr;</span> = Prev / Next Point &nbsp;|&nbsp;
      <span class="kbd">Q</span> / <span class="kbd">W</span> / <span class="kbd">E</span> = Start Built / Not Built / Unclear &nbsp;|&nbsp;
      <span class="kbd">I</span> / <span class="kbd">O</span> / <span class="kbd">P</span> = End Built / Not Built / Unclear &nbsp;|&nbsp;
      <span class="kbd">S</span> = Focus Notes
    </div>
  </main>

  <script>
    const CITY_NAME = "{city_name}";
    const STORAGE_KEY = `urbanpulse_labels_${{CITY_NAME.toLowerCase()}}`;
    const points = {points_json};
    let currentIndex = 0;
    let annotations = {{}};

    function init() {{
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {{
        try {{ annotations = JSON.parse(stored); }} catch(e) {{}}
      }}
      renderPoint(0);
      updateStats();
    }}

    function saveState() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(annotations));
      updateStats();
    }}

    function renderPoint(index) {{
      if (index < 0 || index >= points.length) return;
      currentIndex = index;
      const pt = points[currentIndex];
      const ptId = pt.id;

      document.getElementById('pointBadge').textContent = `Point #${{ptId}} (${{currentIndex + 1}} of ${{points.length}})`;
      document.getElementById('imgStart').src = `chips/${{ptId}}_start.jpg`;
      document.getElementById('imgEnd').src = `chips/${{ptId}}_end.jpg`;

      const gmapsUrl = `https://www.google.com/maps/@${{pt.lat}},${{pt.lon}},17z/data=!3m1!1e3`;
      document.getElementById('gmapsLink').href = gmapsUrl;

      const cur = annotations[ptId] || {{ built_start: '', built_end: '', notes: '' }};
      
      ['btnStartBuilt', 'btnStartNotBuilt', 'btnStartUnclear'].forEach(id => document.getElementById(id).classList.remove('active'));
      ['btnEndBuilt', 'btnEndNotBuilt', 'btnEndUnclear'].forEach(id => document.getElementById(id).classList.remove('active'));

      if (cur.built_start === 'Y' || cur.built_start === '1') document.getElementById('btnStartBuilt').classList.add('active');
      if (cur.built_start === 'N' || cur.built_start === '0') document.getElementById('btnStartNotBuilt').classList.add('active');
      if (cur.built_start === 'unclear') document.getElementById('btnStartUnclear').classList.add('active');

      if (cur.built_end === 'Y' || cur.built_end === '1') document.getElementById('btnEndBuilt').classList.add('active');
      if (cur.built_end === 'N' || cur.built_end === '0') document.getElementById('btnEndNotBuilt').classList.add('active');
      if (cur.built_end === 'unclear') document.getElementById('btnEndUnclear').classList.add('active');

      document.getElementById('notesInput').value = cur.notes || '';
    }}

    function setLabel(period, val) {{
      const ptId = points[currentIndex].id;
      if (!annotations[ptId]) {{
        annotations[ptId] = {{ built_start: '', built_end: '', notes: '' }};
      }}
      if (period === 'start') {{
        annotations[ptId].built_start = val;
      }} else if (period === 'end') {{
        annotations[ptId].built_end = val;
      }}
      saveState();
      renderPoint(currentIndex);

      if (annotations[ptId].built_start && annotations[ptId].built_end && currentIndex < points.length - 1) {{
        setTimeout(() => nextPoint(), 120);
      }}
    }}

    function saveNote() {{
      const ptId = points[currentIndex].id;
      if (!annotations[ptId]) annotations[ptId] = {{ built_start: '', built_end: '', notes: '' }};
      annotations[ptId].notes = document.getElementById('notesInput').value.trim();
      saveState();
    }}

    function prevPoint() {{
      if (currentIndex > 0) renderPoint(currentIndex - 1);
    }}

    function nextPoint() {{
      if (currentIndex < points.length - 1) renderPoint(currentIndex + 1);
    }}

    function jumpToFirstUnlabelled() {{
      for (let i = 0; i < points.length; i++) {{
        const ptId = points[i].id;
        const ann = annotations[ptId];
        if (!ann || !ann.built_start || !ann.built_end) {{
          renderPoint(i);
          return;
        }}
      }}
      alert('All points are labelled!');
    }}

    function updateStats() {{
      let answered = 0;
      for (let i = 0; i < points.length; i++) {{
        const ann = annotations[points[i].id];
        if (ann && ann.built_start && ann.built_end) answered++;
      }}
      const pct = Math.round((answered / points.length) * 100);
      document.getElementById('answeredStats').textContent = `Answered: ${{answered}} / ${{points.length}} (${{pct}}%)`;
      document.getElementById('progressBar').style.width = `${{pct}}%`;
    }}

    function downloadCSV() {{
      let csvContent = 'id,lon,lat,built_start,built_end,notes\\n';
      for (const pt of points) {{
        const ann = annotations[pt.id] || {{ built_start: '', built_end: '', notes: '' }};
        const notesEsc = `"${{(ann.notes || '').replace(/"/g, '""')}}"`;
        csvContent += `${{pt.id}},${{pt.lon}},${{pt.lat}},${{ann.built_start || ''}},${{ann.built_end || ''}},${{notesEsc}}\\n`;
      }}
      const blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `change_sample_labelled_${{CITY_NAME.toLowerCase()}}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }}

    window.addEventListener('keydown', (e) => {{
      if (document.activeElement === document.getElementById('notesInput')) {{
        if (e.key === 'Escape' || e.key === 'Enter') {{
          document.getElementById('notesInput').blur();
        }}
        return;
      }}

      const key = e.key.toUpperCase();
      if (key === 'ARROWLEFT') prevPoint();
      else if (key === 'ARROWRIGHT') nextPoint();
      else if (key === 'Q') setLabel('start', 'Y');
      else if (key === 'W') setLabel('start', 'N');
      else if (key === 'E') setLabel('start', 'unclear');
      else if (key === 'I') setLabel('end', 'Y');
      else if (key === 'O') setLabel('end', 'N');
      else if (key === 'P') setLabel('end', 'unclear');
      else if (key === 'S') {{
        e.preventDefault();
        document.getElementById('notesInput').focus();
      }}
    }});

    window.onload = init;
  </script>
</body>
</html>
"""
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Exported Standalone Labeller HTML: {output_html_path.resolve()}")


def make_label_chips(
    city: str = "ahmedabad",
    start_year: int = 2020,
    end_year: int = 2024,
    data_dir: Path | str = PROJECT_ROOT / "data",
) -> dict[str, Any]:
    """
    Main entry point for generating validation image chips and labeller.html.
    """
    start_time = time.time()
    data_path = Path(data_dir)
    city_key = city.lower()
    city_name = city.capitalize()
    val_dir = data_path / city_key / "validation"
    chips_dir = val_dir / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)

    blind_csv_path = val_dir / "change_sample_blind.csv"
    if not blind_csv_path.exists():
        raise FileNotFoundError(
            f"Blind change sample CSV not found at '{blind_csv_path.resolve()}'. "
            f"Run pipeline/make_change_validation_sample.py first."
        )

    df_blind = pd.read_csv(blind_csv_path)
    print("=" * 86)
    print(f"[*] UrbanPulse 10m True Colour Chip Generator ({city_name})")
    print(f"    - Baseline Year (Start) : {start_year}")
    print(f"    - Target Year (End)     : {end_year}")
    print(f"    - Total Validation Points: {len(df_blind)}")
    print(f"    - Output Chips Directory: {chips_dir.resolve()}")
    print("=" * 86)

    config = load_city_config(city=city_key)

    # 1. Build 10m RGB Composites for Start and End years
    print(f"\n[Step 1/3] Building 10m true colour composite for start year ({start_year})...")
    rgb_start, transform_start, crs_start = fetch_and_composite_10m_rgb(
        city=city_key, year=start_year, config=config, data_dir=data_path
    )

    print(f"\n[Step 2/3] Building 10m true colour composite for end year ({end_year})...")
    rgb_end, transform_end, crs_end = fetch_and_composite_10m_rgb(
        city=city_key, year=end_year, config=config, data_dir=data_path
    )

    # 2. Compute fixed 2%-98% percentile stretch identical across both years
    print("\n[*] Computing 2% - 98% fixed percentile stretch across both years...")
    valid_r = rgb_start[0][np.isfinite(rgb_start[0]) & (rgb_start[0] > 0.0)]
    valid_g = rgb_start[1][np.isfinite(rgb_start[1]) & (rgb_start[1] > 0.0)]
    valid_b = rgb_start[2][np.isfinite(rgb_start[2]) & (rgb_start[2] > 0.0)]

    p2_r, p98_r = float(np.percentile(valid_r, 2)), float(np.percentile(valid_r, 98))
    p2_g, p98_g = float(np.percentile(valid_g, 2)), float(np.percentile(valid_g, 98))
    p2_b, p98_b = float(np.percentile(valid_b, 2)), float(np.percentile(valid_b, 98))

    print(f"    - Red   (2%-98%): [{p2_r:.4f}, {p98_r:.4f}]")
    print(f"    - Green (2%-98%): [{p2_g:.4f}, {p98_g:.4f}]")
    print(f"    - Blue  (2%-98%): [{p2_b:.4f}, {p98_b:.4f}]")

    def apply_stretch(rgb_arr: np.ndarray) -> np.ndarray:
        r = np.clip((rgb_arr[0] - p2_r) / max(1e-5, p98_r - p2_r), 0.0, 1.0) * 255.0
        g = np.clip((rgb_arr[1] - p2_g) / max(1e-5, p98_g - p2_g), 0.0, 1.0) * 255.0
        b = np.clip((rgb_arr[2] - p2_b) / max(1e-5, p98_b - p2_b), 0.0, 1.0) * 255.0
        return np.dstack([r, g, b]).astype(np.uint8)

    rgb_start_uint8 = apply_stretch(rgb_start)
    rgb_end_uint8 = apply_stretch(rgb_end)

    # 3. Extract chips for each sample point
    print(f"\n[Step 3/3] Cutting {len(df_blind)} 128x128 chips (4x upscaled with 60m pixel outline)...")
    transformer = Transformer.from_crs("EPSG:4326", crs_start, always_xy=True)

    chips_created = 0
    points_data = []

    for idx, row in df_blind.iterrows():
        pt_id = int(row["id"])
        lon = float(row["lon"])
        lat = float(row["lat"])
        x_proj, y_proj = transformer.transform(lon, lat)

        # Coordinate to row, col
        r_start, c_start = rasterio.transform.rowcol(transform_start, x_proj, y_proj)
        r_end, c_end = rasterio.transform.rowcol(transform_end, x_proj, y_proj)

        chip_start_img = extract_and_draw_chip(rgb_start_uint8, r_start, c_start)
        chip_end_img = extract_and_draw_chip(rgb_end_uint8, r_end, c_end)

        chip_start_path = chips_dir / f"{pt_id}_start.jpg"
        chip_end_path = chips_dir / f"{pt_id}_end.jpg"

        chip_start_img.save(chip_start_path, "JPEG", quality=95)
        chip_end_img.save(chip_end_path, "JPEG", quality=95)

        chips_created += 2
        points_data.append({
            "id": pt_id,
            "lon": lon,
            "lat": lat,
        })

    # 4. Generate Labeller HTML
    labeller_path = val_dir / "labeller.html"
    generate_standalone_labeller_html(
        city_name=city_name,
        start_year=start_year,
        end_year=end_year,
        points_data=points_data,
        output_html_path=labeller_path,
    )

    elapsed = time.time() - start_time
    print("=" * 86)
    print(f"[+] Finished chip generation for {city_name} in {elapsed:.2f} seconds.")
    print(f"    - Total chips created: {chips_created} ({len(df_blind)} start + {len(df_blind)} end)")
    print(f"    - Chips location: {chips_dir.resolve()}")
    print(f"    - Labeller tool: {labeller_path.resolve()}")
    print("=" * 86)

    return {
        "city": city_name,
        "chips_created": chips_created,
        "elapsed_seconds": elapsed,
        "labeller_html": str(labeller_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build 10m true colour Sentinel-2 chips & standalone labeller for change validation."
    )
    parser.add_argument("--city", type=str, default="ahmedabad", help="City key (default: ahmedabad)")
    parser.add_argument("--start", type=int, default=2020, help="Baseline start year (default: 2020)")
    parser.add_argument("--end", type=int, default=2024, help="Target end year (default: 2024)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Data directory")

    args = parser.parse_args()
    make_label_chips(
        city=args.city,
        start_year=args.start,
        end_year=args.end,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
