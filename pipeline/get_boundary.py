"""
UrbanPulse - Boundary Extraction & 45x45 km AOI Generation Module
Generates a 45x45 km analysis bounding box centered on Ahmedabad city center,
saves configs/ahmedabad_boundary.geojson, and syncs configs/ahmedabad.yaml.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import yaml
from shapely.geometry import Point, box


def extract_city_boundary(
    place_name: str = "Ahmedabad, Gujarat, India",
    aoi_size_km: float = 45.0,
    config_path: str | Path = "configs/ahmedabad.yaml",
    output_geojson_path: str | Path = "configs/ahmedabad_boundary.geojson",
) -> tuple[gpd.GeoDataFrame, list[float]]:
    """
    Generates a 45x45 km urban AOI centered on Ahmedabad city centre.
    """
    config_path = Path(config_path)
    output_geojson_path = Path(output_geojson_path)

    print(f"[*] Geocoding city center for: {place_name}...")
    try:
        lat, lon = ox.geocode(place_name)
        print(f"[+] Found city center coordinates: Lat {lat:.5f}, Lon {lon:.5f}")
    except Exception as e:
        print(f"[!] Fallback to central Ahmedabad coordinates: Lat 23.0225, Lon 72.5714 ({e})")
        lat, lon = 23.0225, 72.5714

    # Create center point GeoDataFrame in WGS84 (EPSG:4326)
    gdf_pt = gpd.GeoDataFrame(
        [{"name": "Ahmedabad", "place": place_name}],
        geometry=[Point(lon, lat)],
        crs="EPSG:4326",
    )

    # Project to metric UTM Zone 43N (EPSG:32643) for precise square extent
    utm_crs = "EPSG:32643"
    print(f"[*] Projecting to {utm_crs} for {aoi_size_km}x{aoi_size_km} km bounding box...")
    gdf_utm = gdf_pt.to_crs(utm_crs)
    center_geom = gdf_utm.geometry.iloc[0]
    cx, cy = center_geom.x, center_geom.y

    half_side_m = (aoi_size_km * 1000.0) / 2.0
    aoi_box_utm = box(cx - half_side_m, cy - half_side_m, cx + half_side_m, cy + half_side_m)

    gdf_aoi_utm = gpd.GeoDataFrame(
        [{"name": "Ahmedabad Metropolitan AOI", "size_km": aoi_size_km}],
        geometry=[aoi_box_utm],
        crs=utm_crs,
    )

    # Re-project to WGS84 for STAC & GeoJSON compatibility
    gdf_aoi_wgs84 = gdf_aoi_utm.to_crs("EPSG:4326")

    output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_aoi_wgs84.to_file(output_geojson_path, driver="GeoJSON")
    print(f"[+] Saved AOI boundary polygon to: {output_geojson_path.resolve()}")

    # Extract bounds [min_lon, min_lat, max_lon, max_lat]
    total_bounds = gdf_aoi_wgs84.total_bounds
    bbox = [
        round(float(total_bounds[0]), 6),
        round(float(total_bounds[1]), 6),
        round(float(total_bounds[2]), 6),
        round(float(total_bounds[3]), 6),
    ]

    print("\n" + "=" * 55)
    print("[*] UrbanPulse 45x45 km Bounding Box (EPSG:4326):")
    print(f"   min_lon (West) : {bbox[0]}")
    print(f"   min_lat (South): {bbox[1]}")
    print(f"   max_lon (East) : {bbox[2]}")
    print(f"   max_lat (North): {bbox[3]}")
    print(f"   bbox list      : {bbox}")
    print("=" * 55 + "\n")

    # Update configs/ahmedabad.yaml
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        if "spatial" not in config_data:
            config_data["spatial"] = {}

        config_data["spatial"]["bbox"] = bbox
        config_data["spatial"]["crs"] = "EPSG:4326"
        config_data["spatial"]["aoi_size_km"] = aoi_size_km
        config_data["spatial"]["boundary_geojson"] = str(output_geojson_path.as_posix())

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, sort_keys=False, default_flow_style=False)

        print(f"[+] Updated bounding box in config file: {config_path.resolve()}")

    return gdf_aoi_wgs84, bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 45x45 km Ahmedabad AOI boundary.")
    parser.add_argument(
        "--size-km", type=float, default=45.0, help="AOI side length in km (default: 45.0)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/ahmedabad.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--out", type=str, default="configs/ahmedabad_boundary.geojson", help="Output GeoJSON path"
    )

    args = parser.parse_args()
    extract_city_boundary(
        aoi_size_km=args.size_km,
        config_path=args.config,
        output_geojson_path=args.out,
    )
