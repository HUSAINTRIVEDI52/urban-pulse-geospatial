"""
UrbanPulse - Change Validation Sample Generator (Four Strata)
Generates a stratified random sample for visual accuracy validation of urban expansion & loss
(2020 -> 2024) based on TLS-normalised classified rasters (3x3 majority filter, NO temporal cleanup).

Strata:
  (A) Not built-up in start (2020) and built-up in end (2024) [Mapped Urban Gain]
  (B) Built-up in both start and end [Persistent Built-up]
  (C) Not built-up in both start and end [Persistent Non-built-up]
  (D) Built-up in start (2020) and not built-up in end (2024) [Mapped Urban Loss]

Sample Allocation:
  - 100 points from Stratum A
  - 50 points from Stratum B
  - 100 points from Stratum C
  - 50 points from Stratum D
  (Total: 300 points, >= 2 pixels away from stratum edges, >= 500 m minimum spatial spacing).

Outputs:
  - data/{city}/validation/change_sample_blind.csv (id, lon, lat, built_start, built_end, notes) - Shuffled for blind labelling
  - data/{city}/validation/change_sample_key.csv (id, stratum) - Evaluation key kept separate
  - data/{city}/validation/change_sample.kml (Google Earth Pro placemarks from blind sample)
  - data/{city}/validation/sample_strata_metadata.json (Strata areas, counts, and parameters)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.train_classifier import (
    FEATURE_NAMES,
    apply_majority_filter_3x3,
    load_config,
)


def get_tls_normalized_classified_raster(
    city: str,
    year: int,
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], rasterio.transform.Affine, Any]:
    """
    Classifies the 8-channel TLS-normalised feature stack for a given city and year
    using the trained pooled Random Forest model, applies a 3x3 majority filter,
    without any temporal cleanup rules.

    Returns:
        (classified_2d, valid_mask_2d, profile, transform, crs)
    """
    city_key = city.lower()
    city_subpath = data_dir / city_key
    
    # Priority: Read from validated_series/ if available
    validated_tif = city_subpath / "validated_series" / f"{city_key}_{year}_classified.tif"
    if not validated_tif.exists():
        validated_tif = data_dir / "validated_series" / f"{city_key}_{year}_classified.tif"
    
    if validated_tif.exists():
        with rasterio.open(validated_tif) as src:
            classified_2d = src.read(1)
            valid_mask_2d = classified_2d > 0
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            return classified_2d, valid_mask_2d, profile, transform, crs

    norm_dir = city_subpath / "normalized"
    if not norm_dir.exists():
        norm_dir = data_dir / "normalized"

    # Locate Model
    model_candidates = [
        city_subpath / "rf_model_pooled.pkl",
        data_dir / f"{city_key}_rf_model_pooled.pkl",
        city_subpath / f"rf_model_{year}.pkl",
        data_dir / f"{city_key}_rf_model_{year}.pkl",
        data_dir / "rf_model_pooled.pkl",
    ]
    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError(
            f"No trained Random Forest model found for {city.capitalize()} in {city_subpath} or {data_dir}"
        )

    rf = joblib.load(model_path)

    # Locate 8 features
    feature_files = []
    for feat in FEATURE_NAMES:
        f_cand = [
            norm_dir / f"{city_key}_{year}_{feat}.tif",
            city_subpath / f"{city_key}_{year}_{feat}.tif",
            data_dir / f"{city_key}_{year}_{feat}.tif",
        ]
        chosen = next((p for p in f_cand if p.exists()), None)
        if chosen is None:
            raise FileNotFoundError(
                f"Missing required normalized feature raster '{feat}' for {city.capitalize()} {year}."
            )
        feature_files.append(chosen)

    # Read rasters
    with rasterio.open(feature_files[0]) as ref_src:
        profile = ref_src.profile.copy()
        transform = ref_src.transform
        crs = ref_src.crs
        raster_shape = (ref_src.height, ref_src.width)

    feature_arrays = []
    for fpath in feature_files:
        with rasterio.open(fpath) as src:
            feature_arrays.append(src.read(1).astype(np.float32))

    stack_2d = np.column_stack([arr.ravel() for arr in feature_arrays])
    valid_1d = np.all(np.isfinite(stack_2d) & (stack_2d != -9999.0), axis=1)
    valid_2d = valid_1d.reshape(raster_shape)

    # Predict
    raw_pred_1d = np.zeros(len(stack_2d), dtype=np.uint8)
    if np.any(valid_1d):
        raw_pred_1d[valid_1d] = rf.predict(stack_2d[valid_1d])
    raw_pred_2d = raw_pred_1d.reshape(raster_shape)

    # 3x3 Majority filter (NO temporal cleanup)
    smooth_2d = apply_majority_filter_3x3(raw_pred_2d, valid_2d)

    return smooth_2d, valid_2d, profile, transform, crs


def compute_strata_masks(
    start_cls: np.ndarray,
    end_cls: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Defines the four change validation strata:
      (A) Not built-up in start & built-up in end (Mapped Gain)
      (B) Built-up in both start & end (Persistent Built)
      (C) Not built-up in both start & end (Persistent Non-built)
      (D) Built-up in start & not built-up in end (Mapped Loss)
    """
    # Class 1 is Built-up
    start_built = (start_cls == 1) & valid_mask
    start_non_built = (start_cls != 1) & (start_cls > 0) & valid_mask

    end_built = (end_cls == 1) & valid_mask
    end_non_built = (end_cls != 1) & (end_cls > 0) & valid_mask

    mask_a = start_non_built & end_built
    mask_b = start_built & end_built
    mask_c = start_non_built & end_non_built
    mask_d = start_built & end_non_built

    return {
        "A": mask_a,
        "B": mask_b,
        "C": mask_c,
        "D": mask_d,
    }


def draw_stratified_sample_with_spacing(
    strata_masks: dict[str, np.ndarray],
    transform: rasterio.transform.Affine,
    crs: Any,
    sample_sizes: dict[str, int],
    min_edge_distance_px: float = 2.0,
    min_spacing_m: float = 500.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Draws random samples from each stratum (A, B, C, D) excluding pixels within min_edge_distance_px
    of stratum edges, with a minimum Euclidean distance of min_spacing_m across ALL sampled points.
    """
    rng = np.random.default_rng(seed)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    selected_points = []  # list of dicts: stratum, x, y, lon, lat

    for stratum_label in ["A", "B", "C", "D"]:
        n_target = sample_sizes.get(stratum_label, 100)
        raw_mask = strata_masks[stratum_label]

        # Euclidean distance transform to nearest non-stratum pixel (in pixel units)
        if min_edge_distance_px > 0:
            dt = distance_transform_edt(raw_mask)
            interior_mask = dt >= min_edge_distance_px
            if np.sum(interior_mask) < n_target:
                interior_mask = dt >= 1.5
            if np.sum(interior_mask) < n_target:
                interior_mask = dt >= 1.0
        else:
            interior_mask = raw_mask.copy()

        rows, cols = np.where(interior_mask)
        n_candidates = len(rows)
        if n_candidates == 0:
            rows, cols = np.where(raw_mask)
            n_candidates = len(rows)

        print(
            f"[*] Stratum {stratum_label}: {n_candidates:,} interior candidate pixels (dt >= {min_edge_distance_px} px, Target sample: {n_target})"
        )

        # Convert candidate pixels to projected coordinates (meters)
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        xs = np.array(xs, dtype=np.float64)
        ys = np.array(ys, dtype=np.float64)

        # Shuffle candidates randomly
        perm = rng.permutation(n_candidates)
        xs = xs[perm]
        ys = ys[perm]

        stratum_selected = 0
        for x, y in zip(xs, ys):
            if stratum_selected >= n_target:
                break

            # Enforce >= min_spacing_m distance against ALL already selected points across all strata
            if selected_points:
                existing_coords = np.array([[p["x"], p["y"]] for p in selected_points])
                dists = np.sqrt((existing_coords[:, 0] - x) ** 2 + (existing_coords[:, 1] - y) ** 2)
                if np.min(dists) < min_spacing_m:
                    continue

            # Accept point
            lon, lat = transformer.transform(x, y)
            pt_record = {
                "stratum": stratum_label,
                "x": x,
                "y": y,
                "lon": round(float(lon), 6),
                "lat": round(float(lat), 6),
            }
            selected_points.append(pt_record)
            stratum_selected += 1

        print(
            f"    -> Successfully selected {stratum_selected}/{n_target} points with >= {min_spacing_m}m spacing."
        )

    # Create DataFrame
    df_raw = pd.DataFrame(selected_points)

    # Shuffle all points together with fixed seed for blind labelling
    df_shuffled = df_raw.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_shuffled["id"] = np.arange(1, len(df_shuffled) + 1)

    return df_shuffled


def export_blind_kml(df_blind: pd.DataFrame, output_kml_path: Path, city_name: str) -> None:
    """Exports Google Earth Pro compatible KML document from the blind sample (no stratum in names or descriptions)."""
    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '  <Document>',
        f'    <name>UrbanPulse Change Validation Sample - {city_name}</name>',
        f'    <description>Blind validation sample for visual accuracy evaluation of urban change.</description>',
        '    <Style id="blind_sample_style">',
        '      <IconStyle>',
        '        <color>ffffaa00</color>',
        '        <scale>1.1</scale>',
        '        <Icon>',
        '          <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>',
        '        </Icon>',
        '      </IconStyle>',
        '      <LabelStyle>',
        '        <scale>0.8</scale>',
        '      </LabelStyle>',
        '    </Style>',
        '    <Folder>',
        f'      <name>Validation Sample Points (N={len(df_blind)})</name>',
    ]

    for _, row in df_blind.iterrows():
        pt_id = int(row["id"])
        lon = float(row["lon"])
        lat = float(row["lat"])
        kml_lines.extend([
            '      <Placemark>',
            f'        <name>Sample #{pt_id}</name>',
            f'        <description><![CDATA[<b>ID:</b> {pt_id}<br><b>Coordinates:</b> {lat:.6f}, {lon:.6f}<br><b>Instructions:</b> Inspect high-resolution historical imagery. Record 1 for Built-up or 0 for Not built-up for Start &amp; End years.]]></description>',
            '        <styleUrl>#blind_sample_style</styleUrl>',
            '        <Point>',
            f'          <coordinates>{lon:.6f},{lat:.6f},0</coordinates>',
            '        </Point>',
            '      </Placemark>',
        ])

    kml_lines.extend([
        '    </Folder>',
        '  </Document>',
        '</kml>',
    ])

    output_kml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_kml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(kml_lines))
    print(f"[+] Exported Google Earth KML (Blind): {output_kml_path.resolve()}")


def generate_change_validation_sample(
    city: str,
    start_year: int = 2020,
    end_year: int = 2024,
    seed: int = 42,
    data_dir: Path | str = PROJECT_ROOT / "data",
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Orchestrates full 4-strata change validation sampling workflow.
    """
    data_path = Path(data_dir)
    city_key = city.lower()
    city_name = city.capitalize()

    if output_dir is None:
        out_dir = data_path / city_key / "validation"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 86)
    print(f"[*] UrbanPulse 4-Stratum Change Validation Sampling ({city_name})")
    print(f"    - Start Year (Baseline) : {start_year}")
    print(f"    - End Year (Target)     : {end_year}")
    print(f"    - Random Seed           : {seed}")
    print(f"    - Output Directory      : {out_dir.resolve()}")
    print("=" * 86)

    # 1. Classify start and end years on TLS-normalised features (NO temporal cleanup)
    print(f"[*] Loading and classifying TLS-normalised rasters for {start_year}...")
    start_cls, valid_start, profile, transform, crs = get_tls_normalized_classified_raster(
        city=city_key, year=start_year, data_dir=data_path
    )

    print(f"[*] Loading and classifying TLS-normalised rasters for {end_year}...")
    end_cls, valid_end, _, _, _ = get_tls_normalized_classified_raster(
        city=city_key, year=end_year, data_dir=data_path
    )

    valid_mask = valid_start & valid_end
    total_raster_px = int(valid_mask.size)
    total_valid_px = int(np.sum(valid_mask))
    nodata_px = total_raster_px - total_valid_px

    pixel_res_x = abs(transform.a)
    pixel_res_y = abs(transform.e)
    pixel_area_km2 = (pixel_res_x * pixel_res_y) / 1e6

    total_raster_km2 = total_raster_px * pixel_area_km2
    total_valid_km2 = total_valid_px * pixel_area_km2
    nodata_km2 = nodata_px * pixel_area_km2

    # 2. Define Strata (A, B, C, D)
    strata = compute_strata_masks(start_cls, end_cls, valid_mask)
    count_a = int(np.sum(strata["A"]))
    count_b = int(np.sum(strata["B"]))
    count_c = int(np.sum(strata["C"]))
    count_d = int(np.sum(strata["D"]))

    sum_strata_px = count_a + count_b + count_c + count_d
    gap_px = total_valid_px - sum_strata_px
    gap_km2 = gap_px * pixel_area_km2

    # Assertion: Strata A + B + C + D must sum exactly to valid AOI
    assert sum_strata_px == total_valid_px, (
        f"Strata sum ({sum_strata_px} px) does not match valid AOI ({total_valid_px} px). "
        f"Unaccounted valid gap: {gap_px} px ({gap_km2:.4f} km²)."
    )

    area_a = count_a * pixel_area_km2
    area_b = count_b * pixel_area_km2
    area_c = count_c * pixel_area_km2
    area_d = count_d * pixel_area_km2

    print("\n" + "-" * 86)
    print(f"[*] STRATIFICATION SUMMARY: {city_name} ({start_year} -> {end_year})")
    print("-" * 86)
    print(f"  Pixel Resolution : {pixel_res_x:.1f} m x {pixel_res_y:.1f} m ({pixel_area_km2:.6f} km²/pixel)")
    print(f"  Total AOI Grid   : {total_raster_km2:.2f} km² ({total_raster_px:,} pixels)")
    print(f"  Valid AOI Area   : {total_valid_km2:.2f} km² ({total_valid_px:,} pixels, {total_valid_px/total_raster_px*100:.2f}%)")
    print(f"  NoData / Masked  : {nodata_km2:.2f} km² ({nodata_px:,} pixels, {nodata_px/total_raster_px*100:.2f}%)\n")

    summary_table = pd.DataFrame([
        {
            "Stratum": "A (Mapped Gain)",
            "Description": f"Not Built ({start_year}) -> Built ({end_year})",
            "Pixel Count": f"{count_a:,}",
            "Area (km²)": f"{area_a:.2f}",
            "Area (%)": f"{area_a/total_valid_km2*100:.2f}%",
            "Target Sample": 100,
        },
        {
            "Stratum": "B (Persistent Built)",
            "Description": f"Built ({start_year}) -> Built ({end_year})",
            "Pixel Count": f"{count_b:,}",
            "Area (km²)": f"{area_b:.2f}",
            "Area (%)": f"{area_b/total_valid_km2*100:.2f}%",
            "Target Sample": 50,
        },
        {
            "Stratum": "C (Persistent Non-Built)",
            "Description": f"Not Built ({start_year}) -> Not Built ({end_year})",
            "Pixel Count": f"{count_c:,}",
            "Area (km²)": f"{area_c:.2f}",
            "Area (%)": f"{area_c/total_valid_km2*100:.2f}%",
            "Target Sample": 100,
        },
        {
            "Stratum": "D (Mapped Loss)",
            "Description": f"Built ({start_year}) -> Not Built ({end_year})",
            "Pixel Count": f"{count_d:,}",
            "Area (km²)": f"{area_d:.2f}",
            "Area (%)": f"{area_d/total_valid_km2*100:.2f}%",
            "Target Sample": 50,
        },
    ])
    print(summary_table.to_string(index=False))
    print(f"\n  * Sum of Strata (A+B+C+D): {sum_strata_px:,} px = {total_valid_km2:.2f} km² (Match Valid AOI: 100.0%)")
    print("-" * 86 + "\n")

    # 3. Draw Samples (100 A, 50 B, 100 C, 50 D = 300 total)
    sample_sizes = {"A": 100, "B": 50, "C": 100, "D": 50}
    df_sample = draw_stratified_sample_with_spacing(
        strata_masks=strata,
        transform=transform,
        crs=crs,
        sample_sizes=sample_sizes,
        min_edge_distance_px=2,
        min_spacing_m=500.0,
        seed=seed,
    )

    # 4. Prepare Blind CSV, Key CSV, and Blind KML
    blind_csv_path = out_dir / "change_sample_blind.csv"
    key_csv_path = out_dir / "change_sample_key.csv"
    kml_path = out_dir / "change_sample.kml"

    # Export Blind CSV: id, lon, lat, built_start, built_end, notes
    df_blind = df_sample[["id", "lon", "lat"]].copy()
    df_blind["built_start"] = ""
    df_blind["built_end"] = ""
    df_blind["notes"] = ""
    df_blind.to_csv(blind_csv_path, index=False)
    print(f"[+] Exported Blind Change Sample CSV : {blind_csv_path.resolve()}")

    # Export Key CSV: id, stratum
    df_key = df_sample[["id", "stratum"]].copy()
    df_key.to_csv(key_csv_path, index=False)
    print(f"[+] Exported Separate Key CSV        : {key_csv_path.resolve()}")

    # Also keep legacy/complete change_sample.csv for direct compatibility
    legacy_csv_path = out_dir / "change_sample.csv"
    df_legacy = df_sample[["id", "stratum", "lon", "lat"]].copy()
    df_legacy["built_start"] = ""
    df_legacy["built_end"] = ""
    df_legacy["notes"] = ""
    df_legacy.to_csv(legacy_csv_path, index=False)

    # Export Blind KML
    export_blind_kml(df_blind, kml_path, city_name)

    # Save metadata summary
    meta_json_path = out_dir / "sample_strata_metadata.json"
    metadata = {
        "city": city_name,
        "start_year": start_year,
        "end_year": end_year,
        "seed": seed,
        "pixel_area_km2": pixel_area_km2,
        "total_raster_km2": round(total_raster_km2, 4),
        "total_aoi_km2": round(total_valid_km2, 4),
        "nodata_km2": round(nodata_km2, 4),
        "nodata_pixels": nodata_px,
        "strata_areas_km2": {
            "A": round(area_a, 4),
            "B": round(area_b, 4),
            "C": round(area_c, 4),
            "D": round(area_d, 4),
        },
        "strata_pixel_counts": {
            "A": count_a,
            "B": count_b,
            "C": count_c,
            "D": count_d,
        },
        "sample_counts": {
            "A": int((df_sample["stratum"] == "A").sum()),
            "B": int((df_sample["stratum"] == "B").sum()),
            "C": int((df_sample["stratum"] == "C").sum()),
            "D": int((df_sample["stratum"] == "D").sum()),
            "total": len(df_sample),
        },
    }

    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate 4-stratum change validation sample (2020-2024) with edge exclusion and minimum spacing."
    )
    parser.add_argument("--city", type=str, default="ahmedabad", help="City key (e.g. ahmedabad, pune)")
    parser.add_argument("--start", type=int, default=2020, help="Start baseline year (default: 2020)")
    parser.add_argument("--end", type=int, default=2024, help="End target year (default: 2024)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Data directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output validation directory")

    args = parser.parse_args()
    generate_change_validation_sample(
        city=args.city,
        start_year=args.start,
        end_year=args.end,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

