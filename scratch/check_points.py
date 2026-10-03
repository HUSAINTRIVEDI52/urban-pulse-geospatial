from pathlib import Path

import geopandas as gpd

for city in ["ahmedabad", "pune"]:
    tr_p = Path(f"data/{city}/train_points_pooled.geojson")
    te_p = Path(f"data/{city}/test_points_pooled.geojson")
    tr = gpd.read_file(tr_p)
    te = gpd.read_file(te_p)
    print(f"\n{city}:")
    print(f"Train points pooled: {len(tr)} rows, years: {tr['year'].value_counts().to_dict()}")
    print(f"Test points pooled: {len(te)} rows, years: {te['year'].value_counts().to_dict()}")
    print(f"Columns: {list(tr.columns)}")
    print(f"Classes in train: {tr['class_id'].value_counts().to_dict()}")
