"""
UrbanPulse - Validated Series Raster & Manifest Generator
Generates reproducible classified GeoTIFF rasters for all analysis years (2018-2024)
using the trained pooled Random Forest model on TLS-normalised features with a 3x3 majority filter
(without temporal cleanup rules), and writes their cryptographic SHA-256 hashes to manifest.json.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.make_change_validation_sample import get_tls_normalized_classified_raster

ALL_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]


def generate_city_validated_series(
    city: str,
    data_dir: str | Path = "data",
    years: list[int] | None = None,
) -> dict[str, Any]:
    """
    Generates validated series GeoTIFFs for all years and records SHA-256 hashes in manifest.json.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    out_dir = data_path / city_key / "validated_series"
    out_dir.mkdir(parents=True, exist_ok=True)

    if years is None:
        years = ALL_YEARS

    manifest: dict[str, Any] = {
        "city": city_key,
        "pipeline": "TLS-normalised 8-band features + pooled RF (or single-year fallback) + 3x3 majority filter (no temporal cleanup)",
        "generated_from": "pipeline.make_change_validation_sample.get_tls_normalized_classified_raster",
        "files": {},
    }

    print("=" * 80)
    print(f"[*] GENERATING VALIDATED SERIES RASTERS: {city.upper()} ({years[0]}-{years[-1]})")
    print(f"    Target Directory : {out_dir.resolve()}")
    print("=" * 80)

    for yr in years:
        cls_arr, mask_arr, prof, transform, crs = get_tls_normalized_classified_raster(
            city=city_key,
            year=yr,
            data_dir=data_path,
        )
        out_tif = out_dir / f"{city_key}_{yr}_classified.tif"

        # Save classified raster
        prof.update(dtype=rasterio.uint8, count=1, nodata=0, compress="lzw")
        with rasterio.open(out_tif, "w", **prof) as dst:
            dst.write(cls_arr.astype(rasterio.uint8), 1)

        # Compute cryptographic SHA-256 hash
        with open(out_tif, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        px_km2 = (abs(transform.a) * abs(transform.e)) / 1e6
        built_px = int(np.sum((cls_arr == 1) & mask_arr))
        built_km2 = round(built_px * px_km2, 2)

        manifest["files"][str(yr)] = {
            "filename": out_tif.name,
            "sha256": file_hash,
            "builtup_pixels": built_px,
            "builtup_km2": built_km2,
        }
        print(
            f"  [+] Year {yr}: {built_km2:>6.2f} km² ({built_px:>7} px) -> {out_tif.name} [SHA-256: {file_hash[:16]}...]"
        )

    manifest_file = out_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[+] Saved manifest to: {manifest_file.resolve()}\n")

    return manifest


if __name__ == "__main__":
    for city_name in ["ahmedabad", "pune"]:
        generate_city_validated_series(city=city_name)
