"""
UrbanPulse - Multi-Year Land Cover Dynamics & Urban Growth Tracker
Loops through years (2017-2024), executes the annual pipeline, logs errors for any failed years,
compiles class area statistics over time into a CSV file, and generates a publication-ready
trend plot of built-up expansion.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.run_year import run_year_pipeline
from pipeline.train_classifier import PROJECT_CLASS_NAMES


def plot_builtup_trend(
    df: pd.DataFrame,
    city_name: str = "Ahmedabad",
    output_png: str | Path = "data/builtup_trend.png",
    dpi: int = 200,
) -> Path:
    """
    Plots historical built-up area trend over time with key growth metrics and styling.
    """
    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if df.empty or "Built-up" not in df.columns:
        print("[!] Warning: DataFrame empty or 'Built-up' column missing. Skipping plot.")
        return out_file

    df_sorted = df.sort_values("Year").copy()
    years = df_sorted["Year"].values
    builtup_km2 = df_sorted["Built-up"].values

    # Calculate Growth Metrics
    start_val = builtup_km2[0]
    end_val = builtup_km2[-1]
    net_growth = end_val - start_val
    pct_growth = (net_growth / start_val) * 100.0
    n_years = years[-1] - years[0]
    cagr = ((end_val / start_val) ** (1.0 / n_years) - 1.0) * 100.0 if n_years > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    fig.patch.set_facecolor("#0f172a")  # Deep slate navy background
    ax.set_facecolor("#1e293b")

    # Grid lines
    ax.grid(True, linestyle="--", linewidth=0.6, color="#334155", alpha=0.7, zorder=1)

    # Plot line and glow effect
    ax.plot(
        years,
        builtup_km2,
        color="#ef4444",
        linewidth=3.0,
        marker="o",
        markersize=8,
        markerfacecolor="#fee2e2",
        markeredgecolor="#b91c1c",
        markeredgewidth=2.0,
        zorder=4,
        label="Built-up Area (km²)",
    )

    # Shaded gradient fill under curve
    ax.fill_between(
        years,
        builtup_km2,
        color="#ef4444",
        alpha=0.20,
        zorder=2,
    )

    # Value annotations on data points
    for yr, val in zip(years, builtup_km2):
        ax.annotate(
            f"{val:.1f}",
            (yr, val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9.5,
            fontweight="bold",
            color="#f8fafc",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#475569", alpha=0.85
            ),
            zorder=5,
        )

    # Axis Labels & Title
    ax.set_title(
        f"{city_name} Urban Expansion Trajectory (2017–2024)\n"
        f"Net Built-up Growth: +{net_growth:.1f} km² (+{pct_growth:.1f}%) | CAGR: {cagr:.2f}%",
        fontsize=13,
        fontweight="bold",
        color="#f8fafc",
        pad=16,
    )
    ax.set_xlabel("Year", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=10)
    ax.set_ylabel(
        "Built-up Area (km²)", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=10
    )

    # X-axis ticks
    ax.set_xticks(years)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

    # Styling ticks and spines
    ax.tick_params(colors="#94a3b8", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(1.2)

    # Legend
    ax.legend(
        loc="upper left",
        facecolor="#0f172a",
        edgecolor="#475569",
        fontsize=10,
        labelcolor="#f8fafc",
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close()

    print(f"[+] Saved built-up trend plot to: {out_file.resolve()}")
    return out_file


def run_all_years(
    city: str = "ahmedabad",
    start_year: int = 2017,
    end_year: int = 2024,
    resolution: float = 60.0,
    model_path: str | Path | None = None,
    force: bool = False,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
) -> tuple[pd.DataFrame, list[int]]:
    """
    Executes annual pipeline for each year in [start_year, end_year], captures results,
    saves summary CSV, and creates a trend visualization.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    csv_file = data_path / f"{city_key}_class_areas.csv"
    plot_file = data_path / f"{city_key}_builtup_trend.png"

    all_years = list(range(start_year, end_year + 1))
    print("\n" + "=" * 80)
    print(f" URBANPULSE MULTI-YEAR PROCESSING BATCH: {city.upper()} ({start_year} - {end_year})")
    print(f" Target Years       : {all_years}")
    print(f" Spatial Resolution : {resolution}m")
    print(f" Output CSV         : {csv_file.resolve()}")
    print("=" * 80 + "\n")

    results_records: list[dict[str, Any]] = []
    errors_log: dict[int, str] = {}

    batch_start = time.time()

    for yr in all_years:
        print(f"\n{'='*35} PROCESSING YEAR {yr} {'='*35}")
        try:
            res = run_year_pipeline(
                city=city,
                year=yr,
                resolution=resolution,
                model_path=model_path,
                force=force,
                config_path=config_path,
                data_dir=data_path,
            )

            record: dict[str, Any] = {
                "City": city.capitalize(),
                "Year": yr,
                "Resolution_m": resolution,
                "Composite_NoData_pct": res.get("nodata_pct", 0.0),
            }

            area_summary = res.get("area_summary", {})
            total_km2 = 0.0
            for cid in range(1, 6):
                cname = PROJECT_CLASS_NAMES[cid]
                km2 = area_summary.get(cname, {}).get("area_km2", 0.0)
                record[cname] = round(km2, 2)
                total_km2 += km2

            record["Total_Area_km2"] = round(total_km2, 2)
            results_records.append(record)
            print(f"[+] Successfully finished year {yr} in {res['timings']['total']:.2f}s")

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            errors_log[yr] = err_msg
            print(f"\n[!] ERROR processing year {yr}: {err_msg}")
            traceback.print_exc()
            print("[*] Continuing with next year in batch...\n")

    # ---------------------------------------------------------
    # COMPILE & DISPLAY RESULTS TABLE
    # ---------------------------------------------------------
    df = pd.DataFrame(results_records)
    if not df.empty:
        df = df.sort_values("Year").reset_index(drop=True)
        # Save CSV
        df.to_csv(csv_file, index=False)
        print("\n" + "=" * 80)
        print(f" URBANPULSE MULTI-YEAR LAND COVER AREA TABLE ({city.capitalize()})")
        print(f" Saved to: {csv_file.name}")
        print("=" * 80)

        # Print formatted summary table
        cols_to_print = [
            "Year",
            "Built-up",
            "Vegetation",
            "Water",
            "Agriculture",
            "Open land",
            "Total_Area_km2",
        ]
        fmt_df = df[cols_to_print].copy()
        for col in [
            "Built-up",
            "Vegetation",
            "Water",
            "Agriculture",
            "Open land",
            "Total_Area_km2",
        ]:
            fmt_df[col] = fmt_df[col].map(lambda x: f"{x:>8.2f} km²")

        print(fmt_df.to_string(index=False))
        print("=" * 80 + "\n")

        # Generate Trend Plot
        plot_builtup_trend(df, city_name=city.capitalize(), output_png=plot_file)
    else:
        print("\n[!] No years completed successfully. No CSV generated.")

    # ---------------------------------------------------------
    # BATCH SUMMARY & ERROR REPORT
    # ---------------------------------------------------------
    batch_elapsed = time.time() - batch_start
    print("\n" + "=" * 80)
    print(f" BATCH PROCESSING COMPLETED in {batch_elapsed:.2f}s")
    print(f" Successfully processed : {len(results_records)} / {len(all_years)} years")
    if errors_log:
        print(f" Failed years ({len(errors_log)}):")
        for f_yr, err in errors_log.items():
            print(f"   - Year {f_yr}: {err}")
    else:
        print(" All years processed with 100% success!")
    print("=" * 80 + "\n")

    successful_years = [int(r["Year"]) for r in results_records]
    return df, successful_years


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-year UrbanPulse land cover analysis (2017-2024)."
    )
    parser.add_argument(
        "--city",
        type=str,
        default="ahmedabad",
        help="City name matching configs/{city}.yaml (default: ahmedabad)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2017,
        help="Start year (default: 2017)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year (default: 2024)",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=60.0,
        help="Spatial resolution in meters (default: 60)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/rf_model_2024.pkl",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing outputs for each year",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Custom config file path",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory",
    )

    args = parser.parse_args()
    run_all_years(
        city=args.city,
        start_year=args.start_year,
        end_year=args.end_year,
        resolution=args.resolution,
        model_path=args.model,
        force=args.force,
        config_path=args.config,
        data_dir=args.data_dir,
    )
