"""
Unit tests for spectral indices computation (NDVI, NDBI, MNDWI).
Tests pure mathematical correctness, zero-division handling, and boundary clipping.
"""

import numpy as np
import pytest
from pipeline.compute_indices import safe_normalized_difference


def test_normalized_difference_hand_calculated():
    """
    Tests (A - B) / (A + B) against manually verified values:
    Example 1: A = 0.6, B = 0.2 -> (0.6 - 0.2) / (0.6 + 0.2) = 0.4 / 0.8 = 0.50
    Example 2: A = 0.1, B = 0.3 -> (0.1 - 0.3) / (0.1 + 0.3) = -0.2 / 0.4 = -0.50
    Example 3: A = 0.5, B = 0.5 -> (0.5 - 0.5) / (0.5 + 0.5) = 0.0
    """
    band_a = np.array([[0.6, 0.1], [0.5, 0.8]], dtype=np.float32)
    band_b = np.array([[0.2, 0.3], [0.5, 0.0]], dtype=np.float32)

    result = safe_normalized_difference(band_a, band_b)

    assert result[0, 0] == pytest.approx(0.5, abs=1e-5)
    assert result[0, 1] == pytest.approx(-0.5, abs=1e-5)
    assert result[1, 0] == pytest.approx(0.0, abs=1e-5)
    assert result[1, 1] == pytest.approx(1.0, abs=1e-5)


def test_zero_denominator_safe_handling():
    """
    Ensures safe_normalized_difference does not crash or return +/-inf
    when denominator is zero (e.g. A = 0.0, B = 0.0). Must return NaN.
    """
    band_a = np.array([[0.0, 0.4], [0.0, -0.2]], dtype=np.float32)
    band_b = np.array([[0.0, 0.1], [0.0, 0.2]], dtype=np.float32)

    result = safe_normalized_difference(band_a, band_b)

    # [0, 0] -> (0 - 0)/(0 + 0) -> NaN, not inf, no crash
    assert np.isnan(result[0, 0])
    assert not np.isinf(result[0, 0])

    # [1, 0] -> (0 - 0)/(0 + 0) -> NaN
    assert np.isnan(result[1, 0])

    # [1, 1] -> (-0.2 - 0.2)/(-0.2 + 0.2) -> denominator = 0 -> NaN
    assert np.isnan(result[1, 1])

    # [0, 1] -> valid (0.4 - 0.1)/(0.4 + 0.1) = 0.3 / 0.5 = 0.6
    assert result[0, 1] == pytest.approx(0.6, abs=1e-5)


def test_indices_clipping_bounds():
    """
    Tests that output index values are safely clipped to [-1.0, 1.0].
    """
    band_a = np.array([[10.0, -10.0]], dtype=np.float32)
    band_b = np.array([[1.0, 1.0]], dtype=np.float32)

    result = safe_normalized_difference(band_a, band_b)
    valid_vals = result[np.isfinite(result)]

    assert np.all(valid_vals >= -1.0)
    assert np.all(valid_vals <= 1.0)
