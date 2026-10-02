import time
import stackstac
import rasterio
from pystac_client import Client
import numpy as np

stac_url = "https://earth-search.aws.element84.com/v1"
client = Client.open(stac_url)
bbox = [72.356712, 22.814991, 72.802746, 23.22784]

search = client.search(
    collections=["sentinel-2-c1-l2a"],
    bbox=bbox,
    datetime="2021-10-01/2022-03-31",
    query={"eo:cloud_cover": {"lt": 10.0}},
)
items = list(search.items())[:4]
print(f"Items found: {len(items)}")

t0 = time.time()
stack = stackstac.stack(
    items,
    assets=["red", "nir", "swir16", "blue", "scl"],
    epsg=32643,
    bounds_latlon=bbox,
    resolution=120.0,
    rescale=False,
    fill_value=np.nan,
)

with rasterio.Env(
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="50000000",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="1",
):
    arr = stack.compute()

print(f"Computed shape: {arr.shape} in {time.time() - t0:.2f}s")
