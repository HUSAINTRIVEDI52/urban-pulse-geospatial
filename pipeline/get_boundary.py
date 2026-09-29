"""
UrbanPulse - Boundary Extraction & Peri-Urban Buffering Module
Fetches Ahmedabad administrative boundary from OpenStreetMap via OSMnx,
applies an 8km peri-urban buffer, saves GeoJSON, and updates city configuration.
"""

from pathlib import Path
import osmnx as ox
import geopandas as gpd
import yaml


def extract_and_buffer_boundary(
    place_name: str = "Ahmedabad, Gujarat, India",
    buffer_km: float = 8.0,
    config_path: str | Path = "configs/ahmedabad.yaml",
    output_geojson_path: str | Path = "configs/ahmedabad_boundary.geojson",
) -> tuple[gpd.GeoDataFrame, list[float]]:
    """
    Geocodes city boundary, applies metric buffer, exports GeoJSON, and updates config YAML.
    """
    config_path = Path(config_path)
    output_geojson_path = Path(output_geojson_path)

    print(f"[*] Geocoding boundary for: {place_name}...")
    gdf = ox.geocode_to_gdf(place_name)

    if gdf.empty:
        raise ValueError(f"No boundary found for query: {place_name}")

    print(f"[+] Retrieved administrative boundary with CRS: {gdf.crs}")

    # Estimate optimal UTM projection for accurate metric buffering
    utm_crs = gdf.estimate_utm_crs()
    print(f"[*] Projecting to UTM CRS ({utm_crs}) for {buffer_km}km buffer calculation...")
    gdf_utm = gdf.to_crs(utm_crs)

    # Buffer geometry by buffer_km (converted to meters)
    buffer_meters = buffer_km * 1000.0
    gdf_utm["geometry"] = gdf_utm.geometry.buffer(buffer_meters)

    # Transform back to WGS84 (EPSG:4326) for standardized GeoJSON & STAC queries
    gdf_buffered_wgs84 = gdf_utm.to_crs("EPSG:4326")

    # Save to GeoJSON
    output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_buffered_wgs84.to_file(output_geojson_path, driver="GeoJSON")
    print(f"[+] Saved buffered boundary to: {output_geojson_path.resolve()}")

    # Calculate bounding box [min_lon, min_lat, max_lon, max_lat]
    total_bounds = gdf_buffered_wgs84.total_bounds  # array([minx, miny, maxx, maxy])
    bbox = [
        round(float(total_bounds[0]), 6),
        round(float(total_bounds[1]), 6),
        round(float(total_bounds[2]), 6),
        round(float(total_bounds[3]), 6),
    ]

    print("\n" + "=" * 50)
    print("[*] UrbanPulse Bounding Box (EPSG:4326):")
    print(f"   min_lon (West) : {bbox[0]}")
    print(f"   min_lat (South): {bbox[1]}")
    print(f"   max_lon (East) : {bbox[2]}")
    print(f"   max_lat (North): {bbox[3]}")
    print(f"   bbox list      : {bbox}")
    print("=" * 50 + "\n")

    # Update configs/ahmedabad.yaml
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        if "spatial" not in config_data:
            config_data["spatial"] = {}

        config_data["spatial"]["bbox"] = bbox
        config_data["spatial"]["crs"] = "EPSG:4326"
        config_data["spatial"]["buffered_distance_km"] = buffer_km
        config_data["spatial"]["boundary_geojson"] = str(output_geojson_path.as_posix())

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, sort_keys=False, default_flow_style=False)

        print(f"[+] Updated bounding box in config file: {config_path.resolve()}")

    return gdf_buffered_wgs84, bbox


if __name__ == "__main__":
    extract_and_buffer_boundary()
