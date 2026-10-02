import pandas as pd
import json

for city in ["ahmedabad", "pune"]:
    df = pd.read_csv(f"data/{city}/scene_diagnostics.csv")
    print("=" * 80)
    print(f"=== {city.upper()} DIAGNOSTICS ===")
    print(f"Total scenes: {len(df)}")
    flagged = df[df["flagged"] == "FLAGGED"]
    print(f"Flagged scenes: {len(flagged)} ({len(flagged)/len(df)*100:.1f}%)")
    print("\nProcessing Baselines:")
    print(df["processing_baseline"].value_counts())
    
    pre = df[df["date"] < "2022-01-25"]
    post = df[df["date"] >= "2022-01-25"]
    print(f"\nPre 2022-01-25 (n={len(pre)}) means:")
    print(pre[["stable_median_red", "stable_median_nir", "stable_median_swir16", "stable_median_blue"]].mean())
    print(f"\nPost 2022-01-25 (n={len(post)}) means:")
    print(post[["stable_median_red", "stable_median_nir", "stable_median_swir16", "stable_median_blue"]].mean())

    # Check 2019 scenes outside Nov-Feb
    # Check Pune 2018 scenes outside Nov-Feb
    print("\n--- Question (c) Inspection ---")
    df_2019 = df[df["year"] == 2019]
    outside_2019 = df_2019[(df_2019["date"] < "2018-11-01") | (df_2019["date"] > "2019-02-28")]
    print(f"2019 scenes outside Nov-Feb (n={len(outside_2019)}):")
    for _, r in outside_2019.iterrows():
        print(f"  {r['date']} | Tile: {r['mgrs_tile']} | Cloud: {r['cloud_cover_pct']}% | Baseline: {r['processing_baseline']}")
        
    if city == "pune":
        df_2018 = df[df["year"] == 2018]
        outside_2018 = df_2018[(df_2018["date"] < "2017-11-01") | (df_2018["date"] > "2018-02-28")]
        print(f"\nPune 2018 scenes outside Nov-Feb (n={len(outside_2018)}):")
        for _, r in outside_2018.iterrows():
            print(f"  {r['date']} | Tile: {r['mgrs_tile']} | Cloud: {r['cloud_cover_pct']}% | Baseline: {r['processing_baseline']}")

    # Check 2022 dates
    df_2022 = df[df["year"] == 2022]
    print(f"\n2022 dates (min: {df_2022['date'].min()}, max: {df_2022['date'].max()}):")
    print(df_2022[["date", "mgrs_tile", "cloud_cover_pct"]].to_string())
    print("\n")
