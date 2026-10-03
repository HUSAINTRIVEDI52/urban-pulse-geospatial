import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

with rasterio.open("data/ahmedabad/clean/ahmedabad_2024_classified.tif") as src:
    ref_bounds = src.bounds
    ref_shape = src.shape

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

def read_band(asset_info):
    name, href, resampling_mode = asset_info
    with rasterio.Env(**gdal_env):
        with rasterio.open(href) as src:
            win = from_bounds(*ref_bounds, transform=src.transform)
            arr = src.read(1, window=win, out_shape=ref_shape, resampling=resampling_mode)
            return name, arr

for item in items:
    t0 = time.time()
    tasks = [
        ("red", item.assets["red"].href, Resampling.bilinear),
        ("nir", item.assets["nir"].href, Resampling.bilinear),
        ("swir16", item.assets["swir16"].href, Resampling.bilinear),
        ("blue", item.assets["blue"].href, Resampling.bilinear),
        ("scl", item.assets["scl"].href, Resampling.nearest),
    ]
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = dict(pool.map(read_band, tasks))
    print(f"Item {item.id} (5 bands parallel) in {time.time() - t0:.2f}s, red median: {np.median(results['red']):.1f}")
