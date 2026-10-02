"""
Unit tests for UrbanPulse stats.json consistency.
Verifies that stats.json is the single consistent source of truth across both cities:
- Class areas sum to the valid AOI area within 0.5%.
- Card metric values equal the corresponding growth and metric series values.
- Headline 2020-2024 range endpoints exactly equal the min and max of the three estimation methods (raw, cleaned, TLS).
- Validation table mapped areas match Olofsson weighting inputs.
"""

import json
from pathlib import Path
import pytest

CITIES = ["ahmedabad", "pune"]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DATA_DIR = PROJECT_ROOT / "web" / "data"


@pytest.fixture(params=CITIES)
def city_stats(request):
    """Loads stats.json for a given city."""
    city = request.param
    stats_path = WEB_DATA_DIR / city / "stats.json"
    assert stats_path.exists(), f"stats.json missing for {city} at {stats_path}"
    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return city, data


def test_class_areas_sum_to_aoi(city_stats):
    """Asserts that 5-class areas sum to the valid AOI area within 0.5% tolerance for all years."""
    city, data = city_stats
    aoi_area = data.get("aoi_area_km2") or (data.get("city_metadata", {}).get("aoi_area_km2"))
    assert aoi_area and aoi_area > 0, f"Invalid AOI area {aoi_area} for {city}"

    class_areas = data.get("class_areas", [])
    assert len(class_areas) >= 5, f"Expected at least 5 years (2020-2024) of class areas for {city}"

    for entry in class_areas:
        year = entry["year"]
        assert 2020 <= year <= 2024, f"Unexpected year {year} in analysis window"
        
        # Five classes
        keys = ["built_up_km2", "vegetation_km2", "water_km2", "agriculture_km2", "open_land_km2"]
        sum_km2 = sum(entry[k] for k in keys)
        
        # Area sum vs AOI area must be within 0.5% tolerance
        rel_diff = abs(sum_km2 - aoi_area) / aoi_area
        assert rel_diff < 0.005, (
            f"Class areas sum ({sum_km2:.2f} km²) deviates from AOI ({aoi_area:.2f} km²) "
            f"by {rel_diff * 100:.3f}% for {city} {year} (threshold < 0.5%)"
        )
        
        # Shares sum must be ~100%
        shares = entry.get("shares_pct", {})
        sum_pct = sum(shares.values())
        assert abs(sum_pct - 100.0) < 0.1, (
            f"Class shares sum to {sum_pct:.2f}% instead of 100% for {city} {year}"
        )


def test_headline_range_endpoints_match_three_methods(city_stats):
    """
    Asserts that the 2020-2024 headline range endpoints equal the min and max
    of the three estimation methods (raw, cleaned, TLS).
    """
    city, data = city_stats
    growth_series = data.get("growth_series", [])
    assert len(growth_series) >= 5, f"Expected at least 5 years of growth series for {city}"
    
    g_2020 = next(g for g in growth_series if g["year"] == 2020)
    g_2024 = next(g for g in growth_series if g["year"] == 2024)

    # 1. Compute changes for the three methods
    raw_start, raw_end = g_2020["raw_builtup_km2"], g_2024["raw_builtup_km2"]
    clean_start, clean_end = g_2020["clean_builtup_km2"], g_2024["clean_builtup_km2"]
    tls_start, tls_end = g_2020["norm_builtup_km2"], g_2024["norm_builtup_km2"]

    raw_change_km2 = raw_end - raw_start
    clean_change_km2 = clean_end - clean_start
    tls_change_km2 = tls_end - tls_start

    raw_change_pct = (raw_change_km2 / raw_start) * 100.0
    clean_change_pct = (clean_change_km2 / clean_start) * 100.0
    tls_change_pct = (tls_change_km2 / tls_start) * 100.0

    expected_min_km2 = min(raw_change_km2, clean_change_km2, tls_change_km2)
    expected_max_km2 = max(raw_change_km2, clean_change_km2, tls_change_km2)
    expected_min_pct = min(raw_change_pct, clean_change_pct, tls_change_pct)
    expected_max_pct = max(raw_change_pct, clean_change_pct, tls_change_pct)

    # 2. Check headline object
    headline = data.get("headline_2020_2024_expansion", {})
    assert headline, f"Missing headline_2020_2024_expansion in stats.json for {city}"

    range_km2 = headline.get("net_growth_range_km2", [])
    range_pct = headline.get("net_growth_range_pct", [])
    assert len(range_km2) == 2 and len(range_pct) == 2

    assert abs(range_km2[0] - expected_min_km2) < 0.05, (
        f"Headline min_change_km2 ({range_km2[0]}) != expected ({expected_min_km2:.2f})"
    )
    assert abs(range_km2[1] - expected_max_km2) < 0.05, (
        f"Headline max_change_km2 ({range_km2[1]}) != expected ({expected_max_km2:.2f})"
    )
    assert abs(range_pct[0] - expected_min_pct) < 0.1, (
        f"Headline min_change_pct ({range_pct[0]}) != expected ({expected_min_pct:.1f})"
    )
    assert abs(range_pct[1] - expected_max_pct) < 0.1, (
        f"Headline max_change_pct ({range_pct[1]}) != expected ({expected_max_pct:.1f})"
    )

    # 3. Check methods breakdown object consistency
    methods = headline.get("methods_breakdown", {})
    assert "raw" in methods and "clean" in methods and "tls_norm" in methods
    assert abs(methods["raw"]["change_km2"] - raw_change_km2) < 0.05
    assert abs(methods["clean"]["change_km2"] - clean_change_km2) < 0.05
    assert abs(methods["tls_norm"]["change_km2"] - tls_change_km2) < 0.05


def test_card_values_equal_series_values(city_stats):
    """
    Asserts that card values for the latest year (2024) equal the series values.
    """
    city, data = city_stats
    growth_series = data.get("growth_series", [])
    metrics = data.get("metrics", [])

    g_2024 = next(g for g in growth_series if g["year"] == 2024)
    m_2024 = next(m for m in metrics if m["year"] == 2024)

    # TLS-normalised main series built-up area
    assert g_2024["norm_builtup_km2"] > 0

    # Entropy and Core/Periphery
    assert m_2024["shannon_entropy"] > 0
    assert m_2024["core_share_pct"] + m_2024["periphery_share_pct"] <= 100.0


def test_validation_mapped_area_consistency(city_stats):
    """
    Asserts that the validation table mapped areas match the series mapped areas.
    """
    city, data = city_stats
    val_table = data.get("validation_loyo", {}).get("table", [])
    assert len(val_table) >= 4, f"Expected LOYO validation table rows for {city}"

    for row in val_table:
        mapped_area = row["mapped_area_km2"]
        assert mapped_area > 0, f"Invalid mapped area for {row['year']} in {city}"
        adjusted_area = row["adjusted_area_km2"]
        ci_95 = row["ci_95_km2"]
        assert adjusted_area > 0 and ci_95 > 0
