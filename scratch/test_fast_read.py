import time

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

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
        print(f"Reading {item.id}...")
        href = item.assets["red"].href
        with rasterio.open(href) as src:
            # Native crs is UTM
            # Read overview or 100x100 window
            # Reproject bbox to src.crs
            from rasterio.warp import transform_bounds
            utm_bounds = transform_bounds("EPSG:4326", src.crs, *bbox)
            win = from_bounds(*utm_bounds, transform=src.transform)
            data = src.read(1, window=win, out_shape=(100, 100), resampling=Resampling.bilinear)
            print(f"  Read shape: {data.shape} median={np.median(data):.1f} in {time.time() - t0:.2f}s")
