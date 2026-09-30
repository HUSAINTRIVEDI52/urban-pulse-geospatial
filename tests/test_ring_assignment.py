"""
Unit tests for concentric distance ring assignment.
Validates radial distance discretization into discrete concentric intervals.
"""

import numpy as np
import pytest
from pipeline.ring_analysis import generate_ring_definitions, assign_pixels_to_rings


def test_generate_ring_definitions():
    """
    Validates generation of ring boundaries for 2km width up to 22km.
    """
    rings = generate_ring_definitions(ring_width_km=2.0, max_dist_km=22.0)

    assert len(rings) == 11
    assert rings[0] == (0.0, 2.0)
    assert rings[1] == (2.0, 4.0)
    assert rings[-1] == (20.0, 22.0)


def test_ring_assignment_known_distances():
    """
    Tests assigning pixels at known radial distances to ring indices:
      - 0.0 km  -> Ring 0 [0-2 km)
      - 1.5 km  -> Ring 0 [0-2 km)
      - 2.0 km  -> Ring 1 [2-4 km)
      - 3.99 km -> Ring 1 [2-4 km)
      - 5.0 km  -> Ring 2 [4-6 km)
      - 21.5 km -> Ring 10 [20-22 km)
      - 22.0 km -> -1 (beyond max 22km)
      - 25.0 km -> -1 (out of bounds)
      - -2.0 km -> -1 (invalid negative)
    """
    test_distances = np.array(
        [
            [0.0, 1.5, 2.0],
            [3.99, 5.0, 21.5],
            [22.0, 25.0, -2.0],
        ],
        dtype=np.float64,
    )

    expected_rings = np.array(
        [
            [0, 0, 1],
            [1, 2, 10],
            [-1, -1, -1],
        ],
        dtype=np.int32,
    )

    result_rings = assign_pixels_to_rings(test_distances, ring_width_km=2.0, max_dist_km=22.0)

    np.testing.assert_array_equal(result_rings, expected_rings)
