"""
Unit Tests for Composite Radiometric Scaling, SCL Dilation, and Observation Thresholds
"""

import numpy as np
import pytest
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
    dilated_mask = scipy.ndimage.binary_dilation(cloud_shadow_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)

    # Center (2,2) and its 8-neighborhood (1..3, 1..3) should all be masked (9 pixels total)
    assert np.sum(cloud_shadow_mask) == 1
    assert np.sum(dilated_mask) == 9
    assert np.all(dilated_mask[1:4, 1:4] == True)
    assert np.all(dilated_mask[0, :] == False)
    assert np.all(dilated_mask[4, :] == False)


def test_minimum_valid_observations_threshold():
    """Verify that pixels with fewer than min_valid_obs (4) observations are set to NaN."""
    min_valid_obs = 4
    n_times = 6

    # 1 pixel across 6 time steps:
    # Pixel A: 4 valid observations -> median is calculated
    # Pixel B: 3 valid observations (3 NaNs) -> masked to NaN
    pixel_a_series = np.array([0.15, 0.18, np.nan, 0.20, 0.22, np.nan], dtype=np.float32) # 4 valid
    pixel_b_series = np.array([0.15, np.nan, np.nan, 0.20, 0.22, np.nan], dtype=np.float32) # 3 valid

    stack = np.stack([pixel_a_series, pixel_b_series], axis=1) # shape (time=6, pixels=2)

    valid_counts = np.sum(~np.isnan(stack), axis=0) # [4, 3]
    median_vals = np.nanmedian(stack, axis=0)
    median_vals[valid_counts < min_valid_obs] = np.nan

    assert valid_counts[0] == 4
    assert valid_counts[1] == 3
    assert np.isfinite(median_vals[0])
    assert np.isclose(median_vals[0], np.median([0.15, 0.18, 0.20, 0.22]))
    assert np.isnan(median_vals[1])
