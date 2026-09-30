"""
Unit tests for land cover transition matrix computation.
Tests matrix cell counts against hand-calculated synthetic 4x4 rasters and validates NoData exclusion.
"""

import numpy as np
import pytest
from pipeline.change_detection import compute_transition_matrix


def test_transition_matrix_hand_calculated():
    """
    Tests 4x4 raster pair with known transitions and nodata (0) pixels:
    Start:
      [1, 1, 2, 2]
      [1, 2, 4, 4]
      [3, 4, 5, 5]
      [0, 1, 2, 0]  <-- [3,0] and [3,3] are nodata in start
    End:
      [1, 2, 2, 1]
      [1, 1, 4, 1]
      [3, 4, 1, 5]
      [0, 1, 0, 0]  <-- [3,0], [3,2], [3,3] are nodata in end

    Valid pixels (13 total):
      1->1: (0,0), (1,0), (3,1) = 3
      1->2: (0,1)               = 1
      2->1: (0,3), (1,1)        = 2
      2->2: (0,2)               = 1
      3->3: (2,0)               = 1
      4->1: (1,3)               = 1
      4->4: (1,2), (2,1)        = 2
      5->1: (2,2)               = 1
      5->5: (2,3)               = 1
    """
    start_arr = np.array(
        [
            [1, 1, 2, 2],
            [1, 2, 4, 4],
            [3, 4, 5, 5],
            [0, 1, 2, 0],
        ],
        dtype=np.uint8,
    )

    end_arr = np.array(
        [
            [1, 2, 2, 1],
            [1, 1, 4, 1],
            [3, 4, 1, 5],
            [0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )

    expected_counts = np.array(
        [
            [3.0, 1.0, 0.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Test with pixel_area_km2 = 1.0
    result_matrix = compute_transition_matrix(
        start_arr=start_arr,
        end_arr=end_arr,
        num_classes=5,
        pixel_area_km2=1.0,
        nodata_val=0,
    )

    np.testing.assert_array_equal(result_matrix, expected_counts)
    assert result_matrix.sum() == 13.0

    # Test with pixel_area_km2 = 0.5
    result_area_matrix = compute_transition_matrix(
        start_arr=start_arr,
        end_arr=end_arr,
        num_classes=5,
        pixel_area_km2=0.5,
        nodata_val=0,
    )

    np.testing.assert_array_equal(result_area_matrix, expected_counts * 0.5)
    assert result_area_matrix.sum() == 6.5
