import json
import pandas as pd
from pathlib import Path

for city in ['ahmedabad', 'pune']:
    stats_path = Path(f'web/data/{city}/stats.json')
    meta_path = Path(f'web/data/{city}/meta.json')
    
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    meta['years'] = [2020, 2021, 2022, 2023, 2024]
    meta['analysis_window'] = stats['analysis_window']
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
        
    # Read class areas
    ca_file = Path(f'data/{city}_class_areas.csv')
    if ca_file.exists():
        df_ca = pd.read_csv(ca_file)
        class_areas = []
        for _, r in df_ca.iterrows():
            class_areas.append({
                'year': int(r['Year']),
                'built_up_km2': float(r.get('Built-up', 0.0)),
                'vegetation_km2': float(r.get('Vegetation', 0.0)),
                'water_km2': float(r.get('Water', 0.0)),
                'agriculture_km2': float(r.get('Agriculture', 0.0)),
                'open_land_km2': float(r.get('Open land', 0.0)),
                'total_area_km2': float(r.get('Total_Area_km2', r.get('total_area_km2', 0.0)))
            })
        stats['class_areas'] = class_areas
        
    # Read metrics
    m_file = Path(f'data/{city}_metrics.csv')
    if m_file.exists():
        df_m = pd.read_csv(m_file)
        metrics = []
        for _, r in df_m.iterrows():
            metrics.append({
                'year': int(r['year']),
                'builtup_km2': float(r['builtup_km2']),
                'shannon_entropy': float(r.get('shannon_entropy', 0.0)),
                'core_share_pct': float(r.get('core_share_0_6km_pct', 0.0)),
                'periphery_share_pct': float(r.get('periphery_share_gt_12km_pct', 0.0))
            })
        stats['metrics'] = metrics

    # Read rings
    r_file = Path(f'data/{city}_rings.csv')
    if r_file.exists():
        df_r = pd.read_csv(r_file)
        rings_dict = {}
        for yr in [2020, 2021, 2022, 2023, 2024]:
            sub = df_r[df_r['year'] == yr].sort_values('ring_start_km')
            rings_dict[str(yr)] = [
                {
                    'ring_start_km': float(row['ring_start_km']),
                    'ring_end_km': float(row['ring_end_km']),
                    'ring_label': f"{int(row['ring_start_km'])}-{int(row['ring_end_km'])} km",
                    'builtup_km2': float(row['builtup_km2']),
                    'valid_km2': float(row['valid_km2']),
                    'builtup_pct': float(row['builtup_pct'])
                }
                for _, row in sub.iterrows()
            ]
        stats['rings'] = rings_dict
        
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f'Enriched stats and meta for {city}')
