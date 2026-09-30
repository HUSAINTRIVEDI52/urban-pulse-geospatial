"""
Unit tests for normalized Shannon Entropy calculation.
Tests theoretical boundary limits: maximum dispersion (1.0), absolute concentration (0.0),
and zero-safe exception handling.
"""

import numpy as np
import pytest

from pipeline.sprawl_metrics import compute_shannon_entropy


def test_entropy_uniform_distribution():
    """
    Uniform distribution across K rings must produce normalized entropy H_n = 1.0.
    For K = 5 rings, p_i = 1/5 = 0.2:
    raw_H = -5 * (0.2 * ln(0.2)) = ln(5)
    normalized_H = ln(5) / ln(5) = 1.0
    """
    uniform_arr = np.array([25.0, 25.0, 25.0, 25.0, 25.0])
    entropy_val = compute_shannon_entropy(uniform_arr)

    assert entropy_val == pytest.approx(1.0, abs=1e-5)


def test_entropy_single_ring_monocentric():
    """
    Absolute monocentric concentration in a single ring must produce H_n = 0.0.
    For [100, 0, 0, 0, 0], p_1 = 1.0, p_others = 0:
    raw_H = -(1.0 * ln(1.0)) = 0.0
    normalized_H = 0.0 / ln(5) = 0.0
    """
    single_ring_arr = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
    entropy_val = compute_shannon_entropy(single_ring_arr)

    assert entropy_val == pytest.approx(0.0, abs=1e-5)


def test_entropy_all_zero_input():
    """
    Ensures an all-zero input does not crash (e.g. division by zero) and returns 0.0.
    """
    all_zero_arr = np.array([0.0, 0.0, 0.0, 0.0])
    entropy_val = compute_shannon_entropy(all_zero_arr)

    assert entropy_val == 0.0
