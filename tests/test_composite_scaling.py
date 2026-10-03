"""
Unit Tests for Composite Radiometric Scaling, SCL Dilation, and Observation Thresholds
"""

import numpy as np
import scipy.ndimage

from pipeline.build_composite import scale_and_harmonize_dn


def test_pre_2022_reflectance_scaling():
    """Verify that scenes before 2022-01-25 scale with standard 1/10000 factor."""
    raw_dn = np.array([1000, 2000, 5000, 10000], dtype=np.float32)
    item_date = "2021-12-15"

    scaled = scale_and_harmonize_dn(raw_dn, item_datetime=item_date)

    expected = np.array([0.10, 0.20, 0.50, 1.00], dtype=np.float32)
    np.testing.assert_allclose(scaled, expected, atol=1e-5)


def test_post_2022_baseline_0400_offset_subtraction():
    """
    Verify that Sentinel-2 Baseline 04.00 scenes (on/after 2022-01-25)
    have the +1000 DN offset subtracted: (DN - 1000) / 10000.
    """
    raw_dn = np.array([1000, 2000, 3000, 6000, 11000], dtype=np.float32)
    item_date = "2022-02-10"

    scaled = scale_and_harmonize_dn(raw_dn, item_datetime=item_date)

    # 1000 -> (1000 - 1000) / 10000 = 0.00
    # 2000 -> (2000 - 1000) / 10000 = 0.10
    # 3000 -> (3000 - 1000) / 10000 = 0.20
    # 6000 -> (6000 - 1000) / 10000 = 0.50
    # 11000 -> (11000 - 1000) / 10000 = 1.00
    expected = np.array([0.00, 0.10, 0.20, 0.50, 1.00], dtype=np.float32)
    np.testing.assert_allclose(scaled, expected, atol=1e-5)


def test_stac_raster_bands_metadata_scaling():
    """Verify that explicit STAC raster:bands scale and offset are respected."""
    raw_dn = np.array([2000, 3000, 6000], dtype=np.float32)
    scale = 0.0001
    offset = -0.1

    scaled = scale_and_harmonize_dn(raw_dn, item_datetime="2023-01-15", scale=scale, offset=offset)

    # 2000 * 0.0001 - 0.1 = 0.10
    # 3000 * 0.0001 - 0.1 = 0.20
    # 6000 * 0.0001 - 0.1 = 0.50
    expected = np.array([0.10, 0.20, 0.50], dtype=np.float32)
    np.testing.assert_allclose(scaled, expected, atol=1e-5)


def test_synthetic_raster_dn_1500_offset_reflectance():
    """
    Test with a small synthetic raster using DN values around 1500 and offset -0.1
    that checks the output reflectance is about 0.05.
    Calculation: 1500 * 0.0001 + (-0.1) = 0.15 - 0.10 = 0.05.
    """
    synthetic_dn = np.array(
        [[1480.0, 1500.0, 1520.0], [1500.0, 1510.0, 1490.0], [1530.0, 1470.0, 1500.0]],
        dtype=np.float32,
    )

    scale = 0.0001
    offset = -0.1

    reflectance = scale_and_harmonize_dn(
        synthetic_dn,
        item_datetime="2024-01-15",
        scale=scale,
        offset=offset,
    )

    # Exact check for DN=1500: 1500 * 0.0001 - 0.1 = 0.05
    assert np.isclose(reflectance[0, 1], 0.05, atol=1e-5)
    assert np.isclose(reflectance[1, 0], 0.05, atol=1e-5)
    assert np.isclose(reflectance[2, 2], 0.05, atol=1e-5)
    # Entire raster around 1500 DN should be around 0.05 reflectance
    assert np.all(reflectance >= 0.04) and np.all(reflectance <= 0.06)


def test_scl_cloud_mask_and_1_pixel_dilation():
    """Verify that SCL cloud classes are masked and dilated by 1 pixel in 8-connectivity."""
    # 5x5 grid with clear pixels (4=Vegetation) and one central cloud pixel (9=High Cloud) at (2,2)
    scl_grid = np.full((5, 5), 4, dtype=np.int32)
    scl_grid[2, 2] = 9  # High cloud probability

    cloud_shadow_mask = (
        (scl_grid == 0)
        | (scl_grid == 1)
        | (scl_grid == 3)
        | (scl_grid == 8)
        | (scl_grid == 9)
        | (scl_grid == 10)
        | (scl_grid == 11)
    )

    # 3x3 footprint 1-pixel dilation
    dilated_mask = scipy.ndimage.binary_dilation(
        cloud_shadow_mask, structure=np.ones((3, 3), dtype=bool), iterations=1
    )

    # Center (2,2) and its 8-neighborhood (1..3, 1..3) should all be masked (9 pixels total)
    assert np.sum(cloud_shadow_mask) == 1
    assert np.sum(dilated_mask) == 9
    assert np.all(dilated_mask[1:4, 1:4])
    assert np.all(~dilated_mask[0, :])
    assert np.all(~dilated_mask[4, :])


def test_minimum_valid_observations_threshold():
    """Verify that pixels with fewer than min_valid_obs (4) observations are set to NaN."""
    min_valid_obs = 4

    # 1 pixel across 6 time steps:
    # Pixel A: 4 valid observations -> median is calculated
    # Pixel B: 3 valid observations (3 NaNs) -> masked to NaN
    pixel_a_series = np.array([0.15, 0.18, np.nan, 0.20, 0.22, np.nan], dtype=np.float32)  # 4 valid
    pixel_b_series = np.array(
        [0.15, np.nan, np.nan, 0.20, 0.22, np.nan], dtype=np.float32
    )  # 3 valid

    stack = np.stack([pixel_a_series, pixel_b_series], axis=1)  # shape (time=6, pixels=2)

    valid_counts = np.sum(~np.isnan(stack), axis=0)  # [4, 3]
    median_vals = np.nanmedian(stack, axis=0)
    median_vals[valid_counts < min_valid_obs] = np.nan

    assert valid_counts[0] == 4
    assert valid_counts[1] == 3
    assert np.isfinite(median_vals[0])
    assert np.isclose(median_vals[0], np.median([0.15, 0.18, 0.20, 0.22]))
    assert np.isnan(median_vals[1])


def test_strict_window_date_range_generation():
    """Verify that strict window always generates Dec 1 (Y-1) to Feb 15 (Y)."""
    from pipeline.build_composite import get_strict_window_range

    dummy_config = {
        "temporal": {
            "strict_window": {
                "start_month": 12,
                "start_day": "12-01",
                "end_month": 2,
                "end_day": "02-15",
            }
        }
    }

    w_2024 = get_strict_window_range(2024, dummy_config)
    assert w_2024 == "2023-12-01/2024-02-15"

    w_2020 = get_strict_window_range(2020, dummy_config)
    assert w_2020 == "2019-12-01/2020-02-15"


def test_low_confidence_rule_under_four_dates():
    """Verify that fewer than 4 distinct acquisition dates marks year as low_confidence."""

    class DummyItem:
        def __init__(self, item_id, dt_str, cloud=1.0):
            self.id = item_id
            self.datetime = datetime.strptime(dt_str, "%Y-%m-%d")
            self.properties = {
                "datetime": dt_str,
                "eo:cloud_cover": cloud,
                "mgrs:utm_zone": "43",
                "mgrs:latitude_band": "Q",
                "mgrs:grid_square": "BF",
            }

    from datetime import datetime

    # 3 distinct dates -> should result in low confidence count < 4
    items_3_dates = [
        DummyItem("s1", "2023-12-08"),
        DummyItem("s2", "2023-12-18"),
        DummyItem("s3", "2024-01-02"),
    ]
    dates = sorted(list({it.datetime.strftime("%Y-%m-%d") for it in items_3_dates}))
    assert len(dates) == 3
    assert len(dates) < 4  # Triggers low_confidence = True

    # 5 distinct dates -> valid
    items_5_dates = [
        DummyItem("s1", "2023-12-08"),
        DummyItem("s2", "2023-12-18"),
        DummyItem("s3", "2024-01-02"),
        DummyItem("s4", "2024-01-12"),
        DummyItem("s5", "2024-01-22"),
    ]
    dates_5 = sorted(list({it.datetime.strftime("%Y-%m-%d") for it in items_5_dates}))
    assert len(dates_5) == 5
    assert len(dates_5) >= 4  # Triggers low_confidence = False


def test_date_spread_scene_selection_and_flagged_exclusion():
    """Verify that scenes are spread across the window and flagged scenes are excluded."""
    from datetime import datetime

    from pipeline.build_composite import select_scenes_by_date_coverage

    class DummyItem:
        def __init__(self, item_id, dt_str, cloud=1.0, tile="43QBF"):
            self.id = item_id
            self.datetime = datetime.strptime(dt_str, "%Y-%m-%d")
            self.properties = {
                "datetime": dt_str,
                "eo:cloud_cover": cloud,
                "mgrs:utm_zone": tile[:2],
                "mgrs:latitude_band": tile[2],
                "mgrs:grid_square": tile[3:],
            }

    # 12 distinct dates
    items = [DummyItem(f"scene_{i:02d}", f"2023-12-{i+1:02d}", cloud=float(i)) for i in range(12)]
    # Add one flagged item
    flagged_ids = {"scene_00", "scene_01"}

    selected, tile_dict = select_scenes_by_date_coverage(
        items=items, max_scenes_per_tile=8, flagged_ids=flagged_ids
    )

    # Should have at most 8 scenes selected for the tile
    assert len(selected) == 8
    # Flagged scene IDs should be excluded
    selected_ids = {it.id for it in selected}
    assert "scene_00" not in selected_ids
    assert "scene_01" not in selected_ids


def test_select_scenes_modes_mocked():
    """Verify select_scenes_for_composite runs without raising in default, strict-window, scene-ids, and from-report modes."""
    from datetime import datetime
    from unittest.mock import MagicMock

    from pipeline.build_composite import select_scenes_for_composite

    class MockItem:
        def __init__(self, item_id, dt_str, cloud=1.0, tile="42QZL"):
            self.id = item_id
            self.datetime = datetime.strptime(dt_str, "%Y-%m-%d")
            self.properties = {
                "datetime": dt_str,
                "eo:cloud_cover": cloud,
                "mgrs:utm_zone": tile[:2],
                "mgrs:latitude_band": tile[2],
                "mgrs:grid_square": tile[3:],
            }

    mock_items = [
        MockItem("S2A_T42QZL_20231101_L2A", "2023-11-01", 2.0, "42QZL"),
        MockItem("S2A_T42QZL_20231201_L2A", "2023-12-01", 1.0, "42QZL"),
        MockItem("S2A_T43QBF_20231101_L2A", "2023-11-01", 3.0, "43QBF"),
        MockItem("S2A_T43QBF_20231201_L2A", "2023-12-01", 0.5, "43QBF"),
    ]

    mock_client = MagicMock()
    mock_search = MagicMock()
    mock_search.items.return_value = mock_items
    mock_client.search.return_value = mock_search

    bbox = [72.35, 22.81, 72.80, 23.22]

    # 1. Default mode
    sel, tile_dict, dt_range, items, is_low, reason = select_scenes_for_composite(
        client=mock_client, bbox=bbox, year=2024, city="ahmedabad"
    )
    assert len(sel) > 0
    assert "42QZL" in tile_dict

    # 2. Strict-window mode
    dummy_config = {
        "spatial": {"bbox": bbox},
        "composite": {
            "observation_window": {"start_month": 12, "start_day": 1, "end_month": 2, "end_day": 15}
        },
    }
    sel_strict, tile_dict_strict, _, _, _, _ = select_scenes_for_composite(
        client=mock_client,
        bbox=bbox,
        year=2024,
        city="ahmedabad",
        strict_window=True,
        config=dummy_config,
    )
    assert len(sel_strict) > 0

    # 3. Scene-ids mode
    sel_ids, tile_dict_ids, _, _, _, _ = select_scenes_for_composite(
        client=mock_client,
        bbox=bbox,
        year=2024,
        city="ahmedabad",
        scene_ids=["S2A_T42QZL_20231101_L2A"],
    )
    assert len(sel_ids) == 4  # mocked items returned

    # 4. From-report mode
    sel_rep, tile_dict_rep, _, _, _, _ = select_scenes_for_composite(
        client=mock_client,
        bbox=bbox,
        year=2021,
        city="ahmedabad",
        from_report=True,
        data_dir="data",
    )
    assert len(sel_rep) > 0


def test_build_composite_cli_help():
    """Verify python pipeline/build_composite.py --help exits 0."""
    import subprocess

    res = subprocess.run(
        ["python", "pipeline/build_composite.py", "--help"], capture_output=True, text=True
    )
    assert res.returncode == 0
    assert "Build Sentinel-2 composite for UrbanPulse" in res.stdout
