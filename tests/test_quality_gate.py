"""
Unit tests for UrbanPulse Data Quality Gate.
Validates failure triggers on excessive NoData (>5%), erratic built-up growth (>25%),
and low classification accuracy (<0.70).
"""

import pandas as pd
import pytest

from pipeline.quality_gate import DataQualityGateError, validate_quality_gate


def test_quality_gate_passes_valid_data():
    """Tests that standard stable city metrics pass all quality gates."""
    df_areas = pd.DataFrame(
        {
            "Year": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "Built-up": [400.0, 420.0, 415.0, 440.0, 460.0, 455.0, 470.0],
            "Composite_NoData_pct": [0.0, 0.1, 0.0, 0.2, 0.0, 0.4, 0.0],
        }
    )
    result = validate_quality_gate(df_areas, overall_accuracy=0.74)
    assert result["status"] == "PASSED"


def test_quality_gate_fails_excessive_nodata():
    """Fails when any year has composite NoData > 5.0%."""
    df_areas = pd.DataFrame(
        {
            "Year": [2021, 2022],
            "Built-up": [400.0, 420.0],
            "Composite_NoData_pct": [0.0, 7.8],  # 7.8% > 5.0%
        }
    )
    with pytest.raises(DataQualityGateError) as exc_info:
        validate_quality_gate(df_areas, overall_accuracy=0.75)
    assert "NoData percentage (7.80%) exceeds threshold" in str(exc_info.value)


def test_quality_gate_fails_consecutive_builtup_volatility():
    """Fails when built-up area jumps or drops by > 25% in consecutive years."""
    df_areas = pd.DataFrame(
        {
            "Year": [2021, 2022],
            "Built-up": [100.0, 135.0],  # 35% growth > 25%
            "Composite_NoData_pct": [0.0, 0.0],
        }
    )
    with pytest.raises(DataQualityGateError) as exc_info:
        validate_quality_gate(df_areas, overall_accuracy=0.80)
    assert "Built-up area changed by 35.00%" in str(exc_info.value)


def test_quality_gate_fails_low_accuracy():
    """Fails when spatial block validation accuracy is below 0.70."""
    df_areas = pd.DataFrame(
        {
            "Year": [2021, 2022],
            "Built-up": [400.0, 410.0],
            "Composite_NoData_pct": [0.0, 0.0],
        }
    )
    with pytest.raises(DataQualityGateError) as exc_info:
        validate_quality_gate(df_areas, overall_accuracy=0.64)  # 64% < 70%
    assert "Model validation accuracy (64.00%) is below minimum threshold" in str(exc_info.value)
