"""
Unit Tests for Temporal Consistency and Urban Persistence Cleanup Filter
Uses hand-verifiable 5-year synthetic land cover stacks.
"""

import numpy as np

from pipeline.temporal_cleanup import clean_temporal_stack


def test_temporal_cleanup_synthetic_5_years():
    """
    Tests hand-calculated synthetic 5-year pixels:
    - Pixel 0 (One-year spike in interior): [4, 4, 1, 4, 4] -> must remove spike -> [4, 4, 4, 4, 4]
    - Pixel 1 (Real urban conversion):      [4, 4, 1, 1, 1] -> must keep conversion -> [4, 4, 1, 1, 1]
    - Pixel 2 (2-yr built then false loss): [4, 1, 1, 4, 4] -> persistence keeps built -> [4, 1, 1, 1, 1]
    - Pixel 3 (First year boundary spike):  [1, 4, 4, 4, 4] -> must remove spike -> [4, 4, 4, 4, 4]
    - Pixel 4 (Last year boundary spike):   [4, 4, 4, 4, 1] -> must remove spike -> [4, 4, 4, 4, 4]
    - Pixel 5 (Stable Water):               [3, 3, 3, 3, 3] -> must remain water -> [3, 3, 3, 3, 3]
    - Pixel 6 (Veg to Agri transition):     [2, 2, 4, 4, 4] -> must remain as classified -> [2, 2, 4, 4, 4]
    """
    # Construct synthetic stack: Shape (5 years, 1 row, 7 pixels)
    raw_trajectories = np.array(
        [
            [4, 4, 1, 4, 4],  # Pixel 0: Spike at t=2
            [4, 4, 1, 1, 1],  # Pixel 1: Permanent conversion at t=2
            [4, 1, 1, 4, 4],  # Pixel 2: 2-year built-up at t=1,2, dropout at t=3,4
            [1, 4, 4, 4, 4],  # Pixel 3: Boundary spike at t=0
            [4, 4, 4, 4, 1],  # Pixel 4: Boundary spike at t=4
            [3, 3, 3, 3, 3],  # Pixel 5: Pure water
            [2, 2, 4, 4, 4],  # Pixel 6: Veg (2) to Agri (4)
        ],
        dtype=np.uint8,
    )  # Shape (7, 5)

    # Transpose to (5 years, 1, 7)
    stack_3d = raw_trajectories.T[:, np.newaxis, :]  # Shape (5, 1, 7)

    cleaned_stack, stats = clean_temporal_stack(stack_3d)

    # Extract cleaned trajectories: Shape (7, 5)
    cleaned_trajectories = cleaned_stack[:, 0, :].T

    # Expected hand-calculated trajectories:
    expected_trajectories = np.array(
        [
            [4, 4, 4, 4, 4],  # Pixel 0: Spike removed, replaced by modal non-built (4)
            [4, 4, 1, 1, 1],  # Pixel 1: Real conversion maintained
            [4, 1, 1, 1, 1],  # Pixel 2: Built-up persisted after 2 consecutive years
            [4, 4, 4, 4, 4],  # Pixel 3: Start boundary spike removed
            [4, 4, 4, 4, 4],  # Pixel 4: End boundary spike removed
            [3, 3, 3, 3, 3],  # Pixel 5: Water preserved
            [2, 2, 4, 4, 4],  # Pixel 6: Veg/Agri preserved
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(cleaned_trajectories, expected_trajectories)

    # Verify rule counts:
    # Rule 1 removals:
    # - Pixel 0: removed at t=2
    # - Pixel 2: removed at t=3 (then reinstated by Rule 2)
    # - Pixel 3: removed at t=0
    # - Pixel 4: removed at t=4
    assert stats["rule1_removed_false_builtup"][0] == 1  # Pixel 3
    assert stats["rule1_removed_false_builtup"][2] == 1  # Pixel 0
    assert stats["rule1_removed_false_builtup"][4] == 1  # Pixel 4

    # Rule 2 persistence additions:
    # - Pixel 2: at t=3 and t=4 persisted to Built-up (was 0 in majority)
    assert stats["rule2_enforced_persistence"][3] == 1
    assert stats["rule2_enforced_persistence"][4] == 1
