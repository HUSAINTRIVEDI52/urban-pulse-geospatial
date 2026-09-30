"""
UrbanPulse - Concentric Ring & Urban Density Gradient Analysis Module
Analyzes urban density gradients and sprawl dynamics across concentric distance rings
from the city center (0-2 km, 2-4 km, ..., 20-22 km) across multiple years.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_city_config(city: str = "ahmedabad", config_path: str | Path | None = None) -> dict[str, Any]:
    """Loads city YAML configuration."""
    cfg_file = Path(config_path) if config_path else Path(f"configs/{city.lower()}.yaml")
    if not cfg_file.exists():
        raise FileNotFoundError(f"City configuration file not found: {cfg_file.resolve()}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_ring_definitions(
    ring_width_km: float = 2.0, max_dist_km: float = 22.0
) -> list[tuple[float, float]]:
    """Generates (start_km, end_km) tuples for concentric rings."""
    ring_edges = np.arange(0, max_dist_km + ring_width_km, ring_width_km)
    return [(float(ring_edges[i]), float(ring_edges[i + 1])) for i in range(len(ring_edges) - 1)]


def assign_pixels_to_rings(
    dist_km: np.ndarray, ring_width_km: float = 2.0, max_dist_km: float = 22.0
) -> np.ndarray:
    """
    Assigns radial distances to integer ring indices:
    0 for [0, ring_width), 1 for [ring_width, 2*ring_width), etc.
    Returns -1 for pixels beyond max_dist_km or negative distances.
    """
    rings = generate_ring_definitions(ring_width_km=ring_width_km, max_dist_km=max_dist_km)
    ring_idx_arr = np.full_like(dist_km, fill_value=-1, dtype=np.int32)

    for idx, (r_start, r_end) in enumerate(rings):
        mask = (dist_km >= r_start) & (dist_km < r_end)
        ring_idx_arr[mask] = idx

    return ring_idx_arr


def get_city_center_projected(
    config: dict[str, Any], target_crs: Any
) -> tuple[float, float, float, float]:
    """
    Extracts lat/lon city center from config and projects to target raster CRS.
    Returns (center_x, center_y, center_lat, center_lon).
    """
    spatial_cfg = config.get("spatial", {})
    city_cfg = config.get("city", {})

    lat = spatial_cfg.get("center_lat") or city_cfg.get("latitude")
    lon = spatial_cfg.get("center_lon") or city_cfg.get("longitude")

    if lat is None or lon is None:
        bbox = spatial_cfg.get("bbox")
        if bbox and len(bbox) == 4:
            lon = (bbox[0] + bbox[2]) / 2.0
            lat = (bbox[1] + bbox[3]) / 2.0
        else:
            # Ahmedabad central fallback
            lat, lon = 23.0225, 72.5714

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    center_x, center_y = transformer.transform(lon, lat)
    return float(center_x), float(center_y), float(lat), float(lon)


def plot_ring_curves(
    df: pd.DataFrame,
    city_name: str = "Ahmedabad",
    output_png: str | Path = "data/ahmedabad_rings.png",
    dpi: int = 200,
) -> Path:
    """
    Plots built-up percentage vs. distance from city center for each year
    using a sequential colormap (older = light, newer = dark).
    """
    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    years = sorted(df["year"].unique())
    n_years = len(years)

    # Use a vibrant sequential colormap (e.g., plasma, viridis, or YlOrRd)
    color_map = plt.get_cmap("viridis", n_years + 2)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=dpi)
    fig.patch.set_facecolor("#0f172a")  # Deep slate navy background
    ax.set_facecolor("#1e293b")

    ax.grid(True, linestyle="--", linewidth=0.6, color="#334155", alpha=0.7, zorder=1)

    for idx, yr in enumerate(years):
        sub = df[df["year"] == yr].sort_values("ring_start_km")
        # Midpoint distance for smooth continuous curve
        midpoints = (sub["ring_start_km"] + sub["ring_end_km"]) / 2.0
        pcts = sub["builtup_pct"].values

        # Sequential color assignment (idx / (n_years - 1))
        norm_idx = 0.15 + 0.80 * (idx / max(1, n_years - 1))
        line_color = color_map(norm_idx)

        is_extreme = idx == 0 or idx == n_years - 1
        line_width = 3.2 if is_extreme else 1.8
        marker_size = 7.0 if is_extreme else 5.0
        alpha = 1.0 if is_extreme else 0.85

        ax.plot(
            midpoints,
            pcts,
            label=f"{yr}",
            color=line_color,
            linewidth=line_width,
            marker="o",
            markersize=marker_size,
            alpha=alpha,
            zorder=4 if is_extreme else 3,
        )

    ax.set_title(
        f"{city_name} Urban Density Gradient & Ring Sprawl Profiles\n"
        f"Built-up Land Density vs. Radial Distance from City Center",
        fontsize=13,
        fontweight="bold",
        color="#f8fafc",
        pad=16,
    )
    ax.set_xlabel("Distance from City Center (km)", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=10)
    ax.set_ylabel("Built-up Land Share (%)", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=10)

    # Ticks & Styling
    max_dist = df["ring_end_km"].max()
    ax.set_xlim(0, max_dist)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(0, max_dist + 2, 2))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))

    ax.tick_params(colors="#94a3b8", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(1.2)

    # Legend
    ax.legend(
        title="Analysis Year",
        title_fontsize=10,
        loc="upper right",
        facecolor="#0f172a",
        edgecolor="#475569",
        fontsize=9.5,
        labelcolor="#f8fafc",
        framealpha=0.92,
        fancybox=True,
        ncol=2 if n_years > 5 else 1,
    )

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close()

    print(f"[+] Saved concentric ring curves to: {out_file.resolve()}")
    return out_file


def plot_ring_growth_bar(
    df: pd.DataFrame,
    city_name: str = "Ahmedabad",
    output_png: str | Path = "data/ahmedabad_ring_growth.png",
    dpi: int = 200,
) -> Path:
    """
    Plots a bar chart of built-up km2 gained per distance ring between the first and last year.
    """
    out_file = Path(output_png)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    years = sorted(df["year"].unique())
    if len(years) < 2:
        print("[!] Warning: Need at least 2 years for ring growth comparison. Skipping growth bar chart.")
        return out_file

    start_year = years[0]
    end_year = years[-1]

    df_start = df[df["year"] == start_year].sort_values("ring_start_km").reset_index(drop=True)
    df_end = df[df["year"] == end_year].sort_values("ring_start_km").reset_index(drop=True)

    rings = [f"{int(r['ring_start_km'])}-{int(r['ring_end_km'])} km" for _, r in df_start.iterrows()]
    growth_km2 = df_end["builtup_km2"].values - df_start["builtup_km2"].values
    pct_point_growth = df_end["builtup_pct"].values - df_start["builtup_pct"].values

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=dpi)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.grid(True, linestyle="--", linewidth=0.6, color="#334155", alpha=0.7, zorder=1, axis="y")

    # Bar colors based on expansion magnitude
    colors = ["#ef4444" if g >= 0 else "#3b82f6" for g in growth_km2]
    bars = ax.bar(
        rings,
        growth_km2,
        color=colors,
        edgecolor="#ffffff",
        linewidth=0.8,
        alpha=0.88,
        zorder=3,
        width=0.65,
    )

    # Annotations on top of bars
    for bar, g_val, pp_val in zip(bars, growth_km2, pct_point_growth):
        height = bar.get_height()
        va_pos = "bottom" if height >= 0 else "top"
        offset = 0.5 if height >= 0 else -1.2
        ax.annotate(
            f"{g_val:+.1f} km²\n({pp_val:+.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height + offset),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va=va_pos,
            fontsize=8.5,
            fontweight="bold",
            color="#f8fafc",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", edgecolor="#475569", alpha=0.8),
            zorder=5,
        )

    total_net_gain = np.sum(growth_km2)
    ax.set_title(
        f"{city_name} Built-up Land Area Expansion by Radial Ring ({start_year} – {end_year})\n"
        f"Total Ring Net Built-up Gain: {total_net_gain:+.2f} km²",
        fontsize=13,
        fontweight="bold",
        color="#f8fafc",
        pad=16,
    )
    ax.set_xlabel("Concentric Distance Ring from City Center", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=10)
    ax.set_ylabel("Net Built-up Expansion (km²)", fontsize=11, fontweight="bold", color="#cbd5e1", labelpad=10)

    ax.tick_params(colors="#94a3b8", labelsize=9.5)
    plt.xticks(rotation=30, ha="right")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close()

    print(f"[+] Saved concentric ring growth bar chart to: {out_file.resolve()}")
    return out_file


def run_ring_analysis(
    city: str = "ahmedabad",
    ring_width_km: float = 2.0,
    max_dist_km: float = 22.0,
    config_path: str | Path | None = None,
    data_dir: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes concentric ring density gradient analysis across all available classified years.
    """
    city_key = city.lower()
    data_path = Path(data_dir)
    config = load_city_config(city=city, config_path=config_path)
    city_name = config.get("city", {}).get("name", city.capitalize())

    # 1. Discover all classified rasters
    pattern = re.compile(rf"^{re.escape(city_key)}_(\d{{4}})_classified\.tif$")
    year_raster_map: dict[int, Path] = {}
    for f in data_path.glob(f"{city_key}_*_classified.tif"):
        match = pattern.match(f.name)
        if match:
            yr = int(match.group(1))
            year_raster_map[yr] = f

    if not year_raster_map:
        raise FileNotFoundError(
            f"No classified rasters found matching pattern '{city_key}_<year>_classified.tif' in {data_path.resolve()}"
        )

    sorted_years = sorted(year_raster_map.keys())
    print("=" * 80)
    print(f" URBANPULSE CONCENTRIC RING ANALYSIS: {city_name.upper()}")
    print(f" Available Years  : {sorted_years}")
    print(f" Ring Width       : {ring_width_km} km")
    print(f" Max Distance     : {max_dist_km} km")
    print("=" * 80)

    # 2. Open reference raster to extract geometry & resolution
    ref_raster_path = year_raster_map[sorted_years[0]]
    with rasterio.open(ref_raster_path) as src_ref:
        h, w = src_ref.shape
        raster_transform = src_ref.transform
        raster_crs = src_ref.crs
        pixel_res_x = abs(raster_transform.a)
        pixel_res_y = abs(raster_transform.e)
        pixel_area_km2 = (pixel_res_x * pixel_res_y) / 1e6

    # 3. Project city center to raster CRS
    cx, cy, lat, lon = get_city_center_projected(config, raster_crs)
    print(f"[+] City Center: Lat {lat:.5f}, Lon {lon:.5f} -> {raster_crs}: ({cx:.1f}, {cy:.1f})")

    # 4. Generate distance raster aligned to grid
    print(f"[*] Constructing distance-from-centre raster for {w}x{h} grid...")
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs, ys = rasterio.transform.xy(raster_transform, rows.ravel(), cols.ravel())
    xs_arr = np.array(xs, dtype=np.float64).reshape((h, w))
    ys_arr = np.array(ys, dtype=np.float64).reshape((h, w))
    dist_km_arr = np.sqrt((xs_arr - cx) ** 2 + (ys_arr - cy) ** 2) / 1000.0

    # 5. Define Rings: [0, 2), [2, 4), ..., [20, 22)
    ring_edges = np.arange(0, max_dist_km + ring_width_km, ring_width_km)
    rings = [(float(ring_edges[i]), float(ring_edges[i + 1])) for i in range(len(ring_edges) - 1)]

    # 6. Analyze each year per ring
    records: list[dict[str, Any]] = []

    for yr in sorted_years:
        raster_file = year_raster_map[yr]
        with rasterio.open(raster_file) as src:
            classified = src.read(1)

        for r_start, r_end in rings:
            ring_mask = (dist_km_arr >= r_start) & (dist_km_arr < r_end)
            valid_mask = ring_mask & (classified > 0)

            valid_px = int(np.sum(valid_mask))
            builtup_px = int(np.sum(valid_mask & (classified == 1)))

            valid_km2 = valid_px * pixel_area_km2
            builtup_km2 = builtup_px * pixel_area_km2
            builtup_pct = (builtup_km2 / valid_km2 * 100.0) if valid_km2 > 0 else 0.0

            records.append(
                {
                    "year": yr,
                    "ring_start_km": r_start,
                    "ring_end_km": r_end,
                    "ring_label": f"{int(r_start):02d}-{int(r_end):02d} km",
                    "builtup_km2": round(builtup_km2, 2),
                    "valid_km2": round(valid_km2, 2),
                    "builtup_pct": round(builtup_pct, 2),
                }
            )

    df_long = pd.DataFrame(records)

    # 7. Save Long-Format CSV
    csv_out = data_path / f"{city_key}_rings.csv"
    csv_cols = ["year", "ring_start_km", "ring_end_km", "builtup_km2", "valid_km2", "builtup_pct"]
    df_long[csv_cols].to_csv(csv_out, index=False)
    print(f"\n[+] Saved long-format ring statistics to: {csv_out.resolve()}")

    # 8. Compute and Print Pivot Table
    pivot_pct = df_long.pivot(index="ring_label", columns="year", values="builtup_pct")

    print("\n" + "=" * 85)
    print(f"[*] CONCENTRIC RING BUILT-UP DENSITY PIVOT TABLE: {city_name} (% Built-up)")
    print("=" * 85)
    # Format pivot table nicely
    fmt_pivot = pivot_pct.copy()
    for col in fmt_pivot.columns:
        fmt_pivot[col] = fmt_pivot[col].map(lambda x: f"{x:>6.2f}%")
    print(fmt_pivot.to_string())
    print("=" * 85 + "\n")

    # 9. Generate Visualizations
    png_rings = data_path / f"{city_key}_rings.png"
    png_growth = data_path / f"{city_key}_ring_growth.png"

    plot_ring_curves(df_long, city_name=city_name, output_png=png_rings)
    plot_ring_growth_bar(df_long, city_name=city_name, output_png=png_growth)

    return df_long, pivot_pct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Concentric distance ring & urban density gradient analysis."
    )
    parser.add_argument("--city", type=str, default="ahmedabad", help="City name (default: ahmedabad)")
    parser.add_argument("--ring-width", type=float, default=2.0, help="Ring width in km (default: 2.0)")
    parser.add_argument("--max-dist", type=float, default=22.0, help="Maximum radial distance in km (default: 22.0)")
    parser.add_argument("--config", type=str, default=None, help="Custom city config file path")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory for data files (default: data)")

    args = parser.parse_args()
    run_ring_analysis(
        city=args.city,
        ring_width_km=args.ring_width,
        max_dist_km=args.max_dist,
        config_path=args.config,
        data_dir=args.data_dir,
    )
