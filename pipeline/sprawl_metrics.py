"""
UrbanPulse - Urban Sprawl Metrics & Spatial Entropy Analytics Module
Computes urban expansion velocity, Compound Annual Growth Rate (CAGR),
Shannon Entropy (measuring spatial dispersion vs. monocentric compactness),
and Core vs. Periphery built-up distribution ratios across years.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# Shannon Entropy Interpretation Guide:
# ------------------------------------------------------------------------------
# Shannon entropy (H_n) measures the degree of spatial dispersion of built-up
# land across concentric rings, normalized by ln(K) where K is the number of rings:
#   - H_n values near 0: Indicate COMPACT, monocentric urban development,
#     where built-up land is concentrated heavily in a few central rings.
#   - H_n values approaching 1: Indicate DISPERSED, leapfrog urban sprawl,
#     where urban development spreads out uniformly across the outer periphery.
# ==============================================================================


def compute_shannon_entropy(builtup_by_ring: np.ndarray) -> float:
    """
    Calculates normalized Shannon Entropy H_n = -sum(p_i * ln(p_i)) / ln(K).
    Returns a value between 0 (highly concentrated) and 1 (fully dispersed).
    """
    total_built = np.sum(builtup_by_ring)
    if total_built <= 0:
        return 0.0

    k = len(builtup_by_ring)
    if k <= 1:
        return 0.0

    # Probability distribution p_i of built-up area in ring i
    p = builtup_by_ring / total_built
    # Filter non-zero probabilities to avoid log(0)
    p_nz = p[p > 0]
    raw_entropy = -np.sum(p_nz * np.log(p_nz))
    norm_entropy = raw_entropy / np.log(k)
    return float(np.clip(norm_entropy, 0.0, 1.0))


def plot_sprawl_metrics(
    df_metrics: pd.DataFrame,
    city_name: str = "Ahmedabad",
    output_png: str | Path = "data/ahmedabad_metrics.png",
    dpi: int = 200,
) -> Path:
    """
    Generates a high-contrast publication-grade dual-panel figure:
      Panel 1: Total Built-up Area (km²) & Annual Growth Rate (%)
      Panel 2: Normalized Shannon Entropy & Periphery vs Core Share (%)
    """
    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    years = df_metrics["year"].values
    builtup_km2 = df_metrics["builtup_km2"].values
    entropy = df_metrics["shannon_entropy"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=dpi)
    fig.patch.set_facecolor("#0f172a")  # Deep slate navy

    # --- PANEL 1: Built-up Area Trend ---
    ax1.set_facecolor("#1e293b")
    ax1.grid(True, linestyle="--", linewidth=0.6, color="#334155", alpha=0.7, zorder=1)

    ax1.plot(
        years,
        builtup_km2,
        color="#ef4444",
        linewidth=2.8,
        marker="o",
        markersize=7.5,
        markerfacecolor="#fee2e2",
        markeredgecolor="#b91c1c",
        markeredgewidth=2.0,
        zorder=4,
        label="Total Built-up Area (km²)",
    )
    ax1.fill_between(years, builtup_km2, color="#ef4444", alpha=0.18, zorder=2)

    for yr, val in zip(years, builtup_km2):
        ax1.annotate(
            f"{val:.1f}",
            (yr, val),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9.0,
            fontweight="bold",
            color="#f8fafc",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#475569", alpha=0.85
            ),
            zorder=5,
        )

    ax1.set_title(
        f"{city_name} Total Built-up Expansion (km²)\nPhysical Footprint Growth",
        fontsize=12,
        fontweight="bold",
        color="#f8fafc",
        pad=14,
    )
    ax1.set_xlabel("Year", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=8)
    ax1.set_ylabel(
        "Built-up Land Area (km²)", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=8
    )
    ax1.set_xticks(years)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax1.tick_params(colors="#94a3b8", labelsize=9.5)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(1.2)
    ax1.legend(
        loc="upper left",
        facecolor="#0f172a",
        edgecolor="#475569",
        fontsize=9.5,
        labelcolor="#f8fafc",
    )

    # --- PANEL 2: Normalized Shannon Entropy Trend ---
    ax2.set_facecolor("#1e293b")
    ax2.grid(True, linestyle="--", linewidth=0.6, color="#334155", alpha=0.7, zorder=1)

    ax2.plot(
        years,
        entropy,
        color="#38bdf8",
        linewidth=2.8,
        marker="s",
        markersize=7.5,
        markerfacecolor="#e0f2fe",
        markeredgecolor="#0284c7",
        markeredgewidth=2.0,
        zorder=4,
        label="Normalized Shannon Entropy (0-1)",
    )
    ax2.fill_between(years, entropy, color="#38bdf8", alpha=0.18, zorder=2)

    for yr, ent in zip(years, entropy):
        ax2.annotate(
            f"{ent:.3f}",
            (yr, ent),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9.0,
            fontweight="bold",
            color="#f8fafc",
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#475569", alpha=0.85
            ),
            zorder=5,
        )

    ax2.set_title(
        f"{city_name} Spatial Sprawl Dynamics\nNormalized Shannon Entropy (Near 1 = Dispersed Sprawl)",
        fontsize=12,
        fontweight="bold",
        color="#f8fafc",
        pad=14,
    )
    ax2.set_xlabel("Year", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=8)
    ax2.set_ylabel(
        "Shannon Entropy (H_n)", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=8
    )
    ax2.set_xticks(years)
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax2.set_ylim(0.70, 1.0)
    ax2.tick_params(colors="#94a3b8", labelsize=9.5)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(1.2)
    ax2.legend(
        loc="lower right",
        facecolor="#0f172a",
        edgecolor="#475569",
        fontsize=9.5,
        labelcolor="#f8fafc",
    )

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close()

    print(f"[+] Saved sprawl metrics figure to: {out_file.resolve()}")
    return out_file


def run_sprawl_metrics(
    city: str = "ahmedabad",
    data_dir: str | Path = "data",
) -> pd.DataFrame:
    """
    Loads rings dataset and area summary dataset to compute multi-year sprawl metrics.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    rings_csv = data_path / f"{city_key}_rings.csv"
    areas_csv = data_path / f"{city_key}_class_areas.csv"
    output_csv = data_path / f"{city_key}_metrics.csv"
    output_png = data_path / f"{city_key}_metrics.png"

    if not rings_csv.exists():
        raise FileNotFoundError(
            f"Concentric rings dataset not found: {rings_csv.resolve()}.\n"
            f"Please run 'python pipeline/ring_analysis.py --city {city}' first."
        )

    print("=" * 80)
    print(f" URBANPULSE SPATIAL SPRAWL & SHANNON ENTROPY METRICS: {city.upper()}")
    print(f" Input Rings CSV  : {rings_csv.resolve()}")
    print(f" Output Metrics   : {output_csv.resolve()}")
    print("=" * 80)

    df_rings = pd.read_csv(rings_csv)
    years = sorted(df_rings["year"].unique())
    first_year = years[0]

    # Load class areas if available for verification
    class_areas_map: dict[int, float] = {}
    if areas_csv.exists():
        df_areas = pd.read_csv(areas_csv)
        for _, row in df_areas.iterrows():
            class_areas_map[int(row["Year"])] = float(row["Built-up"])

    metrics_records: list[dict[str, Any]] = []

    prev_builtup: float | None = None

    for yr in years:
        sub = df_rings[df_rings["year"] == yr].sort_values("ring_start_km").copy()

        # Ring-based built-up sum
        ring_builtup_sum = float(sub["builtup_km2"].sum())
        # Use class_areas built-up if present, otherwise ring built-up sum
        builtup_km2 = class_areas_map.get(yr, ring_builtup_sum)

        # 1. Annual Growth Rate (%) vs previous year
        if prev_builtup is not None and prev_builtup > 0:
            annual_growth_pct = ((builtup_km2 - prev_builtup) / prev_builtup) * 100.0
        else:
            annual_growth_pct = 0.0

        # 2. CAGR from first year to current year (%)
        n_years = yr - first_year
        if n_years > 0 and yr in class_areas_map and first_year in class_areas_map:
            first_val = class_areas_map[first_year]
            cagr_pct = (((builtup_km2 / first_val) ** (1.0 / n_years)) - 1.0) * 100.0
        elif n_years > 0:
            first_sub = df_rings[df_rings["year"] == first_year]
            first_val = float(first_sub["builtup_km2"].sum())
            cagr_pct = (
                (((builtup_km2 / first_val) ** (1.0 / n_years)) - 1.0) * 100.0
                if first_val > 0
                else 0.0
            )
        else:
            cagr_pct = 0.0

        # 3. Shannon Entropy of built-up distribution across concentric rings
        ring_builtup_arr = sub["builtup_km2"].values
        shannon_entropy = compute_shannon_entropy(ring_builtup_arr)

        # 4. Core (0-6 km) vs Periphery (> 12 km) Shares
        core_builtup = float(sub[sub["ring_end_km"] <= 6.0]["builtup_km2"].sum())
        periph_builtup = float(sub[sub["ring_start_km"] >= 12.0]["builtup_km2"].sum())

        core_share_pct = (core_builtup / ring_builtup_sum * 100.0) if ring_builtup_sum > 0 else 0.0
        periph_share_pct = (
            (periph_builtup / ring_builtup_sum * 100.0) if ring_builtup_sum > 0 else 0.0
        )

        metrics_records.append(
            {
                "city": city.capitalize(),
                "year": yr,
                "builtup_km2": round(builtup_km2, 2),
                "annual_growth_pct": round(annual_growth_pct, 2),
                "cagr_from_start_pct": round(cagr_pct, 2),
                "shannon_entropy": round(shannon_entropy, 4),
                "core_builtup_0_6km_km2": round(core_builtup, 2),
                "core_share_0_6km_pct": round(core_share_pct, 2),
                "periphery_builtup_gt_12km_km2": round(periph_builtup, 2),
                "periphery_share_gt_12km_pct": round(periph_share_pct, 2),
            }
        )

        prev_builtup = builtup_km2

    df_out = pd.DataFrame(metrics_records)
    df_out.to_csv(output_csv, index=False)
    print(f"[+] Saved sprawl metrics dataset to: {output_csv.resolve()}")

    # Display Formatted Summary Table
    print("\n" + "=" * 92)
    print(f"[*] URBANPULSE MULTI-YEAR SPATIAL SPRAWL & ENTROPY SUMMARY ({city.capitalize()})")
    print("=" * 92)
    display_cols = [
        "year",
        "builtup_km2",
        "annual_growth_pct",
        "cagr_from_start_pct",
        "shannon_entropy",
        "core_share_0_6km_pct",
        "periphery_share_gt_12km_pct",
    ]
    df_disp = df_out[display_cols].copy()
    df_disp.columns = [
        "Year",
        "Built-up (km²)",
        "Annual Growth",
        "CAGR from '18",
        "Entropy (H_n)",
        "Core (0-6km) %",
        "Periphery (>12km) %",
    ]
    df_disp["Built-up (km²)"] = df_disp["Built-up (km²)"].map(lambda x: f"{x:>8.2f} km²")
    df_disp["Annual Growth"] = df_disp["Annual Growth"].map(lambda x: f"{x:>+6.2f}%")
    df_disp["CAGR from '18"] = df_disp["CAGR from '18"].map(lambda x: f"{x:>+6.2f}%")
    df_disp["Entropy (H_n)"] = df_disp["Entropy (H_n)"].map(lambda x: f"{x:>6.4f}")
    df_disp["Core (0-6km) %"] = df_disp["Core (0-6km) %"].map(lambda x: f"{x:>6.2f}%")
    df_disp["Periphery (>12km) %"] = df_disp["Periphery (>12km) %"].map(lambda x: f"{x:>6.2f}%")

    print(df_disp.to_string(index=False))
    print("=" * 92 + "\n")

    # Plot Visualizations
    plot_sprawl_metrics(df_out, city_name=city.capitalize(), output_png=output_png)

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute urban expansion velocity, CAGR, Shannon entropy, and sprawl metrics."
    )
    parser.add_argument(
        "--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory for data rasters and CSVs"
    )

    args = parser.parse_args()
    run_sprawl_metrics(city=args.city, data_dir=args.data_dir)
