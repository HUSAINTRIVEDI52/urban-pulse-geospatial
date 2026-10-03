
import yaml
from pystac_client import Client

client = Client.open("https://earth-search.aws.element84.com/v1")

for city in ["ahmedabad", "pune"]:
    with open(f"configs/{city}.yaml") as f:
        cfg = yaml.safe_load(f)
    bbox = cfg["spatial"]["bbox"]
    print("=" * 60)
    print(f"City: {city.upper()} (Strict Window: Dec 1 - Feb 15)")
    for year in range(2018, 2025):
        dt_range = f"{year-1}-12-01/{year}-02-15"
        search = client.search(
            collections=["sentinel-2-c1-l2a"],
            bbox=bbox,
            datetime=dt_range,
            query={"eo:cloud_cover": {"lt": 20.0}},
        )
        items = list(search.items())
        dates = sorted(list({it.datetime.strftime("%Y-%m-%d") if it.datetime else str(it.properties.get("datetime"))[:10] for it in items}))
        is_low_conf = len(dates) < 4
        status = "LOW_CONFIDENCE (DROPPED)" if is_low_conf else "VALID"
        print(f"  Year {year}: {len(items)} scenes, {len(dates)} distinct dates -> {status} {dates}")
