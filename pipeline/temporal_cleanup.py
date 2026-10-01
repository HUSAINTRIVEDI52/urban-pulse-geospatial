"""
UrbanPulse - Multi-Temporal Land Cover Consistency & Urban Persistence Filter
Post-processes annual classified satellite rasters:
1. Majority rule: Built-up in year t only if Built-up in at least 2 of (t-1, t, t+1) (nearest 2 for endpoints).
2. Persistence rule: Once a pixel is Built-up for 2 consecutive years, it remains Built-up for all later years.
3. Non-built-up assignment: Replaced pixels receive their multi-year modal non-built-up class.
4. Outputs cleaned GeoTIFFs to data/{city}/clean/, leaving originals untouched.
5. Generates comparison table and plot: data/{city}/cleanup_comparison.png.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import rasterio
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.train_classifier import PROJECT_CLASS_NAMES


def clean_temporal_stack(
    stack_3d: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Applies majority and urban persistence rules to a 3D land cover stack (T, H, W).

    Args:
        stack_3d: Array of shape (T, H, W) with class IDs (1=Built-up, 2=Veg, 3=Water, 4=Agri, 5=Open land, 0=NoData).

    Returns:
        tuple of (cleaned_stack_3d, stats_dict)
    """
    n_years, height, width = stack_3d.shape
    built_raw = (stack_3d == 1).astype(np.int32)

    # -------------------------------------------------------------------------
    # Rule 1: Majority Rule (3-Year Moving Window, nearest 2 at endpoints)
    # -------------------------------------------------------------------------
    majority_built = np.zeros_like(built_raw)

    if n_years == 1:
        majority_built = built_raw.copy()
    elif n_years == 2:
        two_yr_agree = (built_raw[0] == 1) & (built_raw[1] == 1)
        majority_built[0] = two_yr_agree.astype(np.int32)
        majority_built[1] = two_yr_agree.astype(np.int32)
    else:
        # First year: check nearest two years (t=0, t=1)
        majority_built[0] = np.where(
            (built_raw[0] == 1) & (built_raw[1] == 1),
            1,
            0,
        )

        # Interior years: at least 2 of (t-1, t, t+1)
        for t in range(1, n_years - 1):
            window_sum = built_raw[t - 1] + built_raw[t] + built_raw[t + 1]
            majority_built[t] = np.where(window_sum >= 2, 1, 0)

        # Last year: check nearest two years (t=T-2, t=T-1)
        majority_built[-1] = np.where(
            (built_raw[-1] == 1) & (built_raw[-2] == 1),
            1,
            0,
        )

    rule1_changed_pixels = np.array(
        [np.sum((built_raw[t] == 1) & (majority_built[t] == 0)) for t in range(n_years)],
        dtype=np.int64,
    )

    # -------------------------------------------------------------------------
    # Rule 2: Urban Persistence Rule (2 Consecutive Years Lock-In)
    # -------------------------------------------------------------------------
    # Urban land persistence assumption: once land is urbanized and built-up for 2
    # consecutive years, it rarely reverts back to natural or agricultural land cover
    # in an expanding metropolitan area.
    persisted_built = majority_built.copy()
    locked_built = np.zeros((height, width), dtype=bool)

    for t in range(n_years):
        if t >= 1:
            two_consecutive = (majority_built[t - 1] == 1) & (majority_built[t] == 1)
            locked_built |= two_consecutive
        persisted_built[t] = np.where(locked_built, 1, majority_built[t])

    rule2_changed_pixels = np.array(
        [np.sum((majority_built[t] == 0) & (persisted_built[t] == 1)) for t in range(n_years)],
        dtype=np.int64,
    )

    # -------------------------------------------------------------------------
    # Rule 3: Non-Built-Up Classes Modal Assignment
    # -------------------------------------------------------------------------
    candidate_classes = [2, 3, 4, 5]
    counts = np.stack([(stack_3d == cid).sum(axis=0) for cid in candidate_classes], axis=0)
    max_idx = np.argmax(counts, axis=0)
    has_any_non_built = counts.sum(axis=0) > 0
    class_lut = np.array(candidate_classes, dtype=np.uint8)
    modal_non_built = np.where(has_any_non_built, class_lut[max_idx], 4)

    cleaned_stack = np.zeros_like(stack_3d)
    for t in range(n_years):
        raw_arr = stack_3d[t].copy()
        is_built = persisted_built[t] == 1
        is_valid = raw_arr > 0

        cleaned_stack[t][is_built] = 1
        non_built_mask = (~is_built) & is_valid
        cleaned_stack[t][non_built_mask] = np.where(
            raw_arr[non_built_mask] != 1,
            raw_arr[non_built_mask],
            modal_non_built[non_built_mask],
        )

    stats = {
        "rule1_removed_false_builtup": rule1_changed_pixels,
        "rule2_enforced_persistence": rule2_changed_pixels,
        "raw_builtup_pixels": np.array([(built_raw[t] == 1).sum() for t in range(n_years)], dtype=np.int64),
        "clean_builtup_pixels": np.array([(persisted_built[t] == 1).sum() for t in range(n_years)], dtype=np.int64),
    }

    return cleaned_stack, stats


def plot_cleanup_comparison(
    years: list[int],
    raw_km2: list[float],
    clean_km2: list[float],
    city_name: str,
    output_png: Path,
    dpi: int = 200,
) -> None:
    """Plots before and after built-up area comparison."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.grid(True, linestyle="--", linewidth=0.6, color="#334155", alpha=0.7, zorder=1)

    # Raw line (dashed amber/red)
    ax.plot(
        years,
        raw_km2,
        color="#f59e0b",
        linestyle="--",
        linewidth=2.2,
        marker="s",
        markersize=7,
        markerfacecolor="#fef3c7",
        markeredgecolor="#d97706",
        markeredgewidth=1.5,
        label="Raw Classified Built-up (km²)",
        zorder=3,
    )

    # Cleaned line (solid emerald green)
    ax.plot(
        years,
        clean_km2,
        color="#10b981",
        linestyle="-",
        linewidth=3.0,
        marker="o",
        markersize=8,
        markerfacecolor="#d1fae5",
        markeredgecolor="#059669",
        markeredgewidth=2.0,
        label="Temporally Cleaned & Persisted Built-up (km²)",
        zorder=4,
    )

    ax.fill_between(years, clean_km2, color="#10b981", alpha=0.15, zorder=2)

    # Annotate points
    for yr, r_val, c_val in zip(years, raw_km2, clean_km2):
        diff = c_val - r_val
        diff_str = f" ({diff:+.1f})" if abs(diff) > 0.1 else ""
        ax.annotate(
            f"{c_val:.1f}{diff_str}",
            (yr, c_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#f8fafc",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#059669", alpha=0.85),
        )

    ax.set_title(
        f"UrbanPulse: {city_name} Land Cover Temporal Cleanup Comparison (2018–2024)",
        fontsize=13,
        fontweight="bold",
        color="#f8fafc",
        pad=15,
    )
    ax.set_xlabel("Observation Year", fontsize=11, fontweight="medium", color="#cbd5e1", labelpad=10)
    ax.set_ylabel("Built-up Footprint (km²)", fontsize=11, fontweight="medium", color="#cbd5e1", labelpad=10)

    ax.set_xticks(years)
    ax.tick_params(axis="both", colors="#94a3b8", labelsize=10)

    legend = ax.legend(loc="upper left", frameon=True, facecolor="#0f172a", edgecolor="#475569", fontsize=9.5)
    for text in legend.get_texts():
        text.set_color("#e2e8f0")

    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_png, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison chart: {output_png.resolve()}")


def run_temporal_cleanup(
    city: str = "ahmedabad",
    data_dir: str | Path = "data",
    start_year: int = 2018,
    end_year: int = 2024,
) -> pd.DataFrame:
    """
    Executes temporal cleanup for all years of a city, saves cleaned rasters
    to data/{city}/clean/, prints before/after stats, and saves comparison chart.
    """
    city_key = city.lower()
    city_name = city.capitalize()
    data_path = Path(data_dir)
    clean_dir = data_path / city_key / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(start_year, end_year + 1))
    n_years = len(years)

    print("=" * 85)
    print(f"[*] UrbanPulse Temporal Consistency & Persistence Cleanup: {city_name} ({start_year}–{end_year})")
    print(f"    - Clean Outputs Directory: {clean_dir.resolve()}")
    print("=" * 85)

    # 1. Load all classified rasters
    raw_rasters = []
    profiles = []

    for y in years:
        candidates = [
            data_path / f"{city_key}_{y}_classified.tif",
            data_path / city_key / f"{city_key}_{y}_classified.tif",
            data_path / city_key / f"{y}_classified.tif",
        ]
        chosen = None
        for c in candidates:
            if c.exists():
                chosen = c
                break
        if chosen is None:
            raise FileNotFoundError(f"Missing classified GeoTIFF for {city_name} {y} in {data_path.resolve()}")

        with rasterio.open(chosen) as src:
            raw_rasters.append(src.read(1))
            profiles.append(src.profile.copy())

    stack = np.stack(raw_rasters, axis=0)
    transform = profiles[0]["transform"]
    pixel_res_x = abs(transform.a)
    pixel_res_y = abs(transform.e)
    pixel_area_km2 = (pixel_res_x * pixel_res_y) / 1e6

    # 2. Apply Temporal Cleanup
    cleaned_stack, stats = clean_temporal_stack(stack)

    # 3. Write Cleaned Rasters to data/{city}/clean/
    for idx, y in enumerate(years):
        out_tif = clean_dir / f"{city_key}_{y}_classified.tif"
        prof = profiles[idx].copy()
        prof.pop("blockxsize", None)
        prof.pop("blockysize", None)
        prof.pop("tiled", None)
        prof.update({
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "compress": "lzw",
        })
        with rasterio.open(out_tif, "w", **prof) as dst:
            dst.write(cleaned_stack[idx], 1)

    print(f"\n[+] Successfully saved {n_years} cleaned GeoTIFFs in: {clean_dir.resolve()}")

    # 4. Compile and Print Before/After Table
    raw_km2 = [stats["raw_builtup_pixels"][i] * pixel_area_km2 for i in range(n_years)]
    clean_km2 = [stats["clean_builtup_pixels"][i] * pixel_area_km2 for i in range(n_years)]
    r1_pix = stats["rule1_removed_false_builtup"]
    r2_pix = stats["rule2_enforced_persistence"]

    table_data = []
    for i, y in enumerate(years):
        table_data.append({
            "Year": y,
            "Raw_Builtup_km2": round(raw_km2[i], 2),
            "Clean_Builtup_km2": round(clean_km2[i], 2),
            "Net_Change_km2": round(clean_km2[i] - raw_km2[i], 2),
            "Rule1_Spikes_Removed_px": int(r1_pix[i]),
            "Rule2_Persisted_Added_px": int(r2_pix[i]),
        })

    df_summary = pd.DataFrame(table_data)

    print("\n" + "=" * 92)
    print(f"[*] TEMPORAL CLEANUP SUMMARY TABLE: {city_name.upper()} (2018–2024)")
    print("=" * 92)
    print(
        f"{'Year':<6} | {'Raw Built-up':<16} | {'Clean Built-up':<16} | {'Net Diff':<12} | {'Rule 1 Removed':<16} | {'Rule 2 Persisted'}"
    )
    print("-" * 92)
    for _, r in df_summary.iterrows():
        print(
            f"{int(r['Year']):<6} | {r['Raw_Builtup_km2']:>11.2f} km² | {r['Clean_Builtup_km2']:>11.2f} km² | "
            f"{r['Net_Change_km2']:>+8.2f} km² | {int(r['Rule1_Spikes_Removed_px']):>13,} px | {int(r['Rule2_Persisted_Added_px']):>14,} px"
        )
    print("=" * 92 + "\n")

    # 5. Save Comparison Chart
    chart_path1 = data_path / city_key / "cleanup_comparison.png"
    chart_path2 = data_path / f"{city_key}_cleanup_comparison.png"
    plot_cleanup_comparison(
        years=years,
        raw_km2=raw_km2,
        clean_km2=clean_km2,
        city_name=city_name,
        output_png=chart_path1,
    )
    if chart_path2 != chart_path1:
        plot_cleanup_comparison(
            years=years,
            raw_km2=raw_km2,
            clean_km2=clean_km2,
            city_name=city_name,
            output_png=chart_path2,
        )

    # Save summary CSV
    df_summary.to_csv(data_path / city_key / "cleanup_summary.csv", index=False)
    df_summary.to_csv(data_path / f"{city_key}_cleanup_summary.csv", index=False)

    return df_summary


def main():
    parser = argparse.ArgumentParser(description="Temporal consistency and persistence cleanup for land cover.")
    parser.add_argument("--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory (default: data)")
    parser.add_argument("--start-year", type=int, default=2018, help="Start year (default: 2018)")
    parser.add_argument("--end-year", type=int, default=2024, help="End year (default: 2024)")

    args = parser.parse_args()
    run_temporal_cleanup(
        city=args.city,
        data_dir=args.data_dir,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
