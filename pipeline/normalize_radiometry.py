"""
UrbanPulse - Radiometric Normalisation & Relative Calibration Module
Calibrates annual Sentinel-2 optical composites to a reference year (2021) using
Pseudo-Invariant Features (PIFs: pixels consistently classified as Water or Built-up
in >= 6 of 7 years, eroded away from edges to eliminate mixed boundary pixels).
Performs per-band Ordinary Least Squares (OLS) linear regression:
    Reflectance_ref = slope * Reflectance_year + intercept
Applies the fitted regression to each band, recomputes spectral indices (NDVI, NDBI, MNDWI),
saves normalized rasters and normalization coefficients, and prints before/after stable-pixel means.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import yaml
from scipy.ndimage import binary_erosion
from sklearn.linear_model import LinearRegression

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OPTICAL_BANDS = ["blue", "green", "red", "nir", "swir16"]
ALL_FEATURES = ["red", "green", "blue", "nir", "swir16", "ndvi", "ndbi", "mndwi"]
DEFAULT_YEARS = list(range(2018, 2025))
REF_YEAR_DEFAULT = 2021


def fit_tls_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Fits Orthogonal Distance Regression (Total Least Squares / Major Axis):
        y = m * x + c
    Minimizes perpendicular squared distances from points to the line.
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_c = x - x_mean
    y_c = y - y_mean
    M = np.column_stack([x_c, y_c])
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    v1 = Vt[0]
    slope = float(v1[1] / v1[0]) if v1[0] != 0 else 1.0
    intercept = float(y_mean - slope * x_mean)
    r = np.corrcoef(x, y)[0, 1]
    return slope, intercept, float(r**2)


def load_config(city: str = "ahmedabad", config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads city YAML configuration."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_normalized_difference(
    band_a: np.ndarray, band_b: np.ndarray, min_denom: float = 1e-5
) -> np.ndarray:
    """Computes normalized difference (A - B) / (A + B) safely."""
    denom = band_a + band_b
    valid = (np.abs(denom) > min_denom) & np.isfinite(band_a) & np.isfinite(band_b)
    res = np.full_like(band_a, fill_value=np.nan, dtype=np.float32)
    np.divide(band_a - band_b, denom, out=res, where=valid)
    return np.clip(res, -1.0, 1.0)


def extract_pif_mask(
    clean_stack_3d: np.ndarray,
    min_years_water: int = 6,
    min_years_built: int = 6,
    erosion_size: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts Pseudo-Invariant Feature (PIF) boolean mask from a 3D cleaned classified stack (T, H, W).
    PIFs are pixels identified as Water (class 3) in >= min_years_water OR Built-up (class 1)
    in >= min_years_built, eroded by erosion_size x erosion_size structural element away from edges.

    Returns:
        tuple of (combined_pif_mask, water_pif_eroded, built_pif_eroded)
    """
    water_counts = np.sum(clean_stack_3d == 3, axis=0)
    built_counts = np.sum(clean_stack_3d == 1, axis=0)

    water_pif_raw = water_counts >= min_years_water
    built_pif_raw = built_counts >= min_years_built

    struct = np.ones((erosion_size, erosion_size), dtype=bool)
    water_pif_eroded = binary_erosion(water_pif_raw, structure=struct)
    built_pif_eroded = binary_erosion(built_pif_raw, structure=struct)

    pif_mask = water_pif_eroded | built_pif_eroded
    return pif_mask, water_pif_eroded, built_pif_eroded


def normalize_city_radiometry(
    city: str = "ahmedabad",
    ref_year: int = REF_YEAR_DEFAULT,
    years: list[int] | None = None,
    exclude_years: list[int] | None = None,
    method: str = "tls",
    data_dir: str | Path = "data",
    output_dir: str | Path | None = None,
    save_rasters: bool = True,
) -> dict[str, Any]:
    """
    Calculates and applies per-band linear radiometric normalisation across all analysis years.

    Args:
        city: City name ('ahmedabad', 'pune').
        ref_year: Reference composite year (default: 2021).
        years: List of composite years to normalise (default: 2018..2024).
        exclude_years: List of years to exclude from calibration fit (retained as identity).
        method: Regression method ('tls' for Total Least Squares, 'ols' for Ordinary Least Squares).
        data_dir: Base data directory containing raw/cleaned rasters.
        output_dir: Destination directory for normalized rasters (default: data/{city}/normalized).
        save_rasters: If True, exports normalized GeoTIFFs to output_dir.

    Returns:
        dict containing coefficients, before/after statistics, and PIF metadata.
    """
    city_key = city.lower()
    city_name = city.capitalize()
    data_path = Path(data_dir)
    city_subpath = data_path / city_key
    clean_dir = city_subpath / "clean"

    if output_dir is None:
        out_path = city_subpath / "normalized"
    else:
        out_path = Path(output_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    if years is None:
        years = DEFAULT_YEARS

    if exclude_years is None:
        exclude_years = [2019] if city_key == "ahmedabad" else [2018, 2019]

    print("=" * 96)
    print(f"[*] UrbanPulse Radiometric Normalisation: {city_name} (Reference Year = {ref_year})")
    print(
        f"    - Calibration Method : {method.upper()} (Orthogonal / Total Least Squares)"
        if method.lower() == "tls"
        else "    - Calibration Method : OLS"
    )
    print(f"    - Target Years       : {years}")
    print(f"    - Excluded from Fit  : {exclude_years}")
    print(f"    - Optical Bands      : {OPTICAL_BANDS}")
    print(f"    - Normalized Output  : {out_path.resolve()}")
    print("=" * 96)

    # 1. Load cleaned classification stack to derive PIFs
    clean_rasters = []
    profiles = []
    for y in years:
        cand = [
            clean_dir / f"{city_key}_{y}_classified.tif",
            data_path / f"{city_key}_{y}_classified.tif",
        ]
        chosen = next((c for c in cand if c.exists()), None)
        if chosen is None:
            raise FileNotFoundError(f"Missing classified raster for {city_name} {y}")
        with rasterio.open(chosen) as src:
            clean_rasters.append(src.read(1))
            profiles.append(src.profile.copy())

    stack_clean = np.stack(clean_rasters, axis=0)
    pif_mask, water_pif_eroded, built_pif_eroded = extract_pif_mask(stack_clean, erosion_size=3)

    n_pifs = int(np.sum(pif_mask))
    n_water_pifs = int(np.sum(water_pif_eroded))
    n_built_pifs = int(np.sum(built_pif_eroded))

    print(f"[+] Extracted Pseudo-Invariant Pixels (PIFs): {n_pifs:,} total pixels away from edges")
    print(f"    * Water PIFs (Class 3 in >=6 of 7 yrs, eroded)    : {n_water_pifs:>7,} pixels")
    print(f"    * Built-up PIFs (Class 1 in >=6 of 7 yrs, eroded) : {n_built_pifs:>7,} pixels")

    # 2. Load Reference Year (2021) Bands
    ref_band_arrays = {}
    ref_profile = profiles[years.index(ref_year)] if ref_year in years else profiles[0]

    for b in OPTICAL_BANDS:
        p = data_path / f"{city_key}_{ref_year}_{b}.tif"
        if not p.exists():
            p = city_subpath / f"{city_key}_{ref_year}_{b}.tif"
        with rasterio.open(p) as src:
            ref_band_arrays[b] = src.read(1).astype(np.float32)

    normalization_results: dict[str, Any] = {
        "city": city_key,
        "ref_year": ref_year,
        "method": method.lower(),
        "exclude_years": exclude_years,
        "pif_counts": {
            "total_pifs": n_pifs,
            "water_pifs": n_water_pifs,
            "builtup_pifs": n_built_pifs,
        },
        "coefficients": {},
    }

    summary_rows = []

    # 3. Fit Regression per Band and Apply Normalisation
    for y in years:
        normalization_results["coefficients"][y] = {}
        normalized_bands = {}
        is_excluded = y in exclude_years

        print(f"\n--- Normalisation Statistics for {city_name} {y} (vs Ref {ref_year}) ---")
        print(
            f"{'Band':<8} | {'Slope (m)':<10} | {'Intercept (c)':<14} | {'R²':<8} | {'Before Mean':<12} | {'After Mean':<12} | {'Status'}"
        )
        print("-" * 96)

        for b in OPTICAL_BANDS:
            p = data_path / f"{city_key}_{y}_{b}.tif"
            if not p.exists():
                p = city_subpath / f"{city_key}_{y}_{b}.tif"
            with rasterio.open(p) as src:
                raw_arr = src.read(1).astype(np.float32)

            x_pif = raw_arr[pif_mask]
            y_pif = ref_band_arrays[b][pif_mask]

            valid_pif = (
                np.isfinite(x_pif) & np.isfinite(y_pif) & (x_pif != -9999.0) & (y_pif != -9999.0)
            )
            x_clean = x_pif[valid_pif]
            y_clean = y_pif[valid_pif]

            if y == ref_year:
                slope, intercept, r2 = 1.0, 0.0, 1.0
                status_str = "Reference Year"
            elif is_excluded:
                slope, intercept, r2 = 1.0, 0.0, 1.0
                status_str = "Excluded from Fit (Identity)"
            elif method.lower() == "tls":
                slope, intercept, r2 = fit_tls_regression(x_clean, y_clean)
                status_str = "TLS Calibrated"
            else:
                reg = LinearRegression().fit(x_clean.reshape(-1, 1), y_clean)
                slope = float(reg.coef_[0])
                intercept = float(reg.intercept_)
                r2 = float(reg.score(x_clean.reshape(-1, 1), y_clean))
                status_str = "OLS Calibrated"

            # Apply calibration to whole composite
            valid_raw = np.isfinite(raw_arr) & (raw_arr != -9999.0)
            norm_arr = np.full_like(raw_arr, -9999.0, dtype=np.float32)
            calibrated_vals = np.clip(slope * raw_arr[valid_raw] + intercept, 0.0, 1.5)
            norm_arr[valid_raw] = calibrated_vals

            normalized_bands[b] = norm_arr

            before_mean = float(np.mean(x_clean)) if len(x_clean) > 0 else 0.0
            after_mean = (
                float(np.mean(norm_arr[pif_mask & valid_raw]))
                if np.any(pif_mask & valid_raw)
                else 0.0
            )
            ref_mean = float(np.mean(y_clean)) if len(y_clean) > 0 else 0.0

            normalization_results["coefficients"][y][b] = {
                "slope": float(round(slope, 6)),
                "intercept": float(round(intercept, 6)),
                "r2": float(round(r2, 4)),
                "before_stable_mean": float(round(before_mean, 6)),
                "after_stable_mean": float(round(after_mean, 6)),
                "ref_stable_mean": float(round(ref_mean, 6)),
                "status": status_str,
            }

            summary_rows.append(
                {
                    "city": city_key,
                    "year": y,
                    "band": b,
                    "slope": round(slope, 6),
                    "intercept": round(intercept, 6),
                    "r2": round(r2, 4),
                    "before_mean": round(before_mean, 6),
                    "after_mean": round(after_mean, 6),
                    "ref_mean": round(ref_mean, 6),
                    "status": status_str,
                }
            )

            print(
                f"{b:<8} | {slope:<10.4f} | {intercept:<14.4f} | {r2:<8.4f} | {before_mean:<12.4f} | {after_mean:<12.4f} | {status_str}"
            )

        # Compute Spectral Indices from Normalized Bands
        red = normalized_bands["red"]
        green = normalized_bands["green"]
        nir = normalized_bands["nir"]
        swir16 = normalized_bands["swir16"]

        normalized_bands["ndvi"] = safe_normalized_difference(nir, red)
        normalized_bands["ndbi"] = safe_normalized_difference(swir16, nir)
        normalized_bands["mndwi"] = safe_normalized_difference(green, swir16)

        # 4. Save Normalized Rasters
        if save_rasters:
            prof = ref_profile.copy()
            prof.pop("blockxsize", None)
            prof.pop("blockysize", None)
            prof.pop("tiled", None)
            prof.update(
                {
                    "driver": "GTiff",
                    "count": 1,
                    "dtype": "float32",
                    "nodata": -9999.0,
                    "compress": "lzw",
                }
            )

            for feat_name, arr in normalized_bands.items():
                out_tif = out_path / f"{city_key}_{y}_{feat_name}.tif"
                with rasterio.open(out_tif, "w", **prof) as dst:
                    dst.write(np.where(np.isnan(arr), -9999.0, arr).astype(np.float32), 1)

    # Save coefficients to JSON and CSV
    coeff_json_path = city_subpath / "radiometric_normalization_coefficients.json"
    coeff_csv_path = city_subpath / "radiometric_normalization_coefficients.csv"

    with open(coeff_json_path, "w", encoding="utf-8") as f:
        json.dump(normalization_results, f, indent=2)
    pd.DataFrame(summary_rows).to_csv(coeff_csv_path, index=False)

    print(f"\n[+] Saved radiometric coefficients to: {coeff_json_path.resolve()}")
    print(f"[+] Saved {len(years) * len(ALL_FEATURES)} normalized rasters to: {out_path.resolve()}")
    print("=" * 96 + "\n")

    return normalization_results


def main():
    parser = argparse.ArgumentParser(
        description="Radiometric normalisation of optical composites using PIFs."
    )
    parser.add_argument("--city", type=str, default="ahmedabad", help="City name (ahmedabad, pune)")
    parser.add_argument("--ref-year", type=int, default=2021, help="Reference year (default: 2021)")
    parser.add_argument("--years", type=int, nargs="+", default=None, help="Years to normalise")
    parser.add_argument(
        "--exclude-years", type=int, nargs="+", default=None, help="Years to exclude from fit"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tls",
        choices=["tls", "ols"],
        help="Regression method (tls, ols)",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for normalized rasters"
    )
    parser.add_argument("--no-save-rasters", action="store_true", help="Skip saving rasters")

    args = parser.parse_args()
    normalize_city_radiometry(
        city=args.city,
        ref_year=args.ref_year,
        years=args.years,
        exclude_years=args.exclude_years,
        method=args.method,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        save_rasters=not args.no_save_rasters,
    )


if __name__ == "__main__":
    main()
