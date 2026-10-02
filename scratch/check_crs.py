import geopandas as gpd
import rasterio

for city in ["ahmedabad", "pune"]:
    tr = gpd.read_file(f"data/{city}/train_points_pooled.geojson")
    with rasterio.open(f"data/{city}_2021_blue.tif") as s:
        raster_crs = s.crs
    print(f"{city}: GeoJSON CRS = {tr.crs}, Raster CRS = {raster_crs}")
    print(f"Sample geom: {tr.geometry.iloc[0]}")
    tr_proj = tr.to_crs(raster_crs)
    print(f"Sample projected geom: {tr_proj.geometry.iloc[0]}")
