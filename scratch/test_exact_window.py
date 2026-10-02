import time
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from pystac_client import Client
import numpy as np

# Load classified profile
with rasterio.open("data/ahmedabad/clean/ahmedabad_2024_classified.tif") as src:
    ref_bounds = src.bounds
    ref_crs = src.crs
    ref_shape = src.shape
    ref_transform = src.transform

print(f"Ref CRS: {ref_crs}, Shape: {ref_shape}, Bounds: {ref_bounds}")

client = Client.open("https://earth-search.aws.element84.com/v1")
bbox = [72.356712, 22.814991, 72.802746, 23.22784]

search = client.search(
    collections=["sentinel-2-c1-l2a"],
    bbox=bbox,
    datetime="2021-10-01/2022-03-31",
    query={"eo:cloud_cover": {"lt": 1.0}},
)
items = list(search.items())[:2]

gdal_env = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "50000000",
    "GDAL_HTTP_MAX_RETRY": "3",
}

with rasterio.Env(**gdal_env):
    for item in items:
        t0 = time.time()
        # Read red, scl
        red_href = item.assets["red"].href
        scl_href = item.assets["scl"].href
        with rasterio.open(red_href) as src_red, rasterio.open(scl_href) as src_scl:
            win_red = from_bounds(*ref_bounds, transform=src_red.transform)
            win_scl = from_bounds(*ref_bounds, transform=src_scl.transform)
            red_arr = src_red.read(1, window=win_red, out_shape=ref_shape, resampling=Resampling.bilinear)
            scl_arr = src_scl.read(1, window=win_scl, out_shape=ref_shape, resampling=Resampling.nearest)
        print(f"Item {item.id} read in {time.time() - t0:.2f}s, shapes: {red_arr.shape}, {scl_arr.shape}, red median={np.median(red_arr):.1f}")
