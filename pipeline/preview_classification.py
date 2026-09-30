"""
UrbanPulse - Classified Land Cover Visualization & Map Exporter
Renders the 2024 Random Forest classified raster with a curated, high-contrast
categorical color scheme (Built-up=Red, Vegetation=Green, Water=Blue,
Agriculture=Yellow, Open land=Tan) and exports data/preview_2024_classified.png.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Curated, vibrant categorical color palette for publication-grade cartography
CLASS_COLORS = {
    0: ("#0d0d0d", "NoData"),
    1: ("#e41a1c", "Built-up"),  # Bright Red
    2: ("#238b45", "Vegetation"),  # Lush Forest Green
    3: ("#1f78b4", "Water"),  # Deep Cerulean Blue
    4: ("#ffd92f", "Agriculture"),  # Golden Yellow
    5: ("#d2b48c", "Open land"),  # Warm Sand Tan
}


def plot_classified_map(
    classified_raster_path: str | Path = "data/ahmedabad_2024_classified.tif",
    output_png_path: str | Path = "data/preview_2024_classified.png",
    dpi: int = 200,
) -> Path:
    """
    Loads classified GeoTIFF, creates categorical land cover plot with legend, and saves PNG.
    """
    raster_path = Path(classified_raster_path)
    out_path = Path(output_png_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raster_path.exists():
        raise FileNotFoundError(
            f"Classified raster not found: {raster_path.resolve()}. "
            "Please run pipeline/train_classifier.py first."
        )

    print("=" * 75)
    print("[*] Rendering Land Cover Classification Map")
    print(f"    - Input Raster : {raster_path.resolve()}")
    print(f"    - Output Image : {out_path.resolve()}")
    print("=" * 75)

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        crs = src.crs

    print(f"[+] Loaded raster shape: {data.shape[0]} rows x {data.shape[1]} cols in {crs}")

    # Build Discrete Colormap & Boundaries
    # Values: 0 (NoData), 1 (Built-up), 2 (Vegetation), 3 (Water), 4 (Agriculture), 5 (Open land)
    color_list = [CLASS_COLORS[i][0] for i in range(6)]
    cmap = ListedColormap(color_list)
    bounds_norm = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds_norm, cmap.N)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 14), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#0d0d0d")

    ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")

    # Extract year/city from filename if possible
    stem = raster_path.stem
    parts = stem.split("_")
    city_str = parts[0].capitalize() if len(parts) > 0 else "Ahmedabad"
    year_str = parts[1] if len(parts) > 1 and parts[1].isdigit() else ""

    title_text = f"{city_str} Metropolitan Area - {year_str} Land Cover Classification\n(Random Forest 200 Trees, 3x3 Majority Filter, Sentinel-2 Composite)"

    ax.set_title(
        title_text,
        fontsize=14,
        fontweight="bold",
        pad=14,
        color="#111111",
    )
    ax.axis("off")

    # Legend Patches (Classes 1 to 5)
    legend_patches = [
        mpatches.Patch(
            facecolor=CLASS_COLORS[cid][0],
            edgecolor="#333333",
            linewidth=0.8,
            label=f"{CLASS_COLORS[cid][1]} (Class {cid})",
        )
        for cid in range(1, 6)
    ]

    legend = ax.legend(
        handles=legend_patches,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02),
        title="Land Cover Legend",
        title_fontsize=11,
        fontsize=10,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#cccccc",
        framealpha=0.92,
        fancybox=True,
        shadow=True,
    )
    legend.get_title().set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close()

    print(f"[+] Successfully exported classification preview to: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render classified land cover map preview.")
    parser.add_argument(
        "--raster",
        type=str,
        default="data/ahmedabad_2024_classified.tif",
        help="Path to classified GeoTIFF",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/preview_2024_classified.png",
        help="Path to output PNG preview",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI (default: 200)")

    args = parser.parse_args()
    plot_classified_map(
        classified_raster_path=args.raster,
        output_png_path=args.out,
        dpi=args.dpi,
    )
