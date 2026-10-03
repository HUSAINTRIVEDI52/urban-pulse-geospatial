"""
Unit tests for UrbanPulse Data Quality Gate.
Validates failure triggers on:
1. Excessive NoData (>5%)
2. Year-to-year built-up volatility (>15% on cleaned series)
3. Low per-year classification accuracy (<0.70)
4. Excessive built-up loss (>30% of gross gain)
"""

import pandas as pd
import pytest

from pipeline.quality_gate import DataQualityGateError, validate_quality_gate


def test_quality_gate_passes_valid_data():
    """Tests that standard stable city metrics pass all quality gates."""
    df_areas = pd.DataFrame(
        {
            "Year": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
            "Built-up": [400.0, 420.0, 435.0, 450.0, 465.0, 480.0, 495.0],
            "Composite_NoData_pct": [0.0, 0.1, 0.0, 0.2, 0.0, 0.4, 0.0],
        }
    )
    result = validate_quality_gate(
        df_areas,
        per_year_accuracies={2018: 0.75, 2021: 0.78, 2024: 0.82},
        gross_gain_km2=100.0,
        gross_loss_km2=5.0,  # 5% < 30%
    )
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
    """Fails when built-up area jumps or drops by > 15% in consecutive years."""
    df_areas = pd.DataFrame(
        {
            "Year": [2021, 2022],
            "Built-up": [100.0, 118.0],  # 18% growth > 15% threshold
            "Composite_NoData_pct": [0.0, 0.0],
        }
    )
    with pytest.raises(DataQualityGateError) as exc_info:
        validate_quality_gate(df_areas, overall_accuracy=0.80)
    assert "Built-up area changed by 18.00%" in str(exc_info.value)


def test_quality_gate_fails_low_per_year_accuracy():
    """Fails when any year's held-out validation accuracy is below 0.70."""
    df_areas = pd.DataFrame(
        {
            "Year": [2021, 2022],
            "Built-up": [400.0, 410.0],
            "Composite_NoData_pct": [0.0, 0.0],
        }
    )
    with pytest.raises(DataQualityGateError) as exc_info:
        validate_quality_gate(
            df_areas,
            per_year_accuracies={2018: 0.75, 2021: 0.68, 2024: 0.80},  # 2021 has 68% < 70%
        )
    assert "Year 2021 validation accuracy (68.00%) is below minimum threshold" in str(
        exc_info.value
    )


def test_quality_gate_fails_excessive_builtup_loss():
    """Fails when built-up loss exceeds 30% of gross gain."""
    df_areas = pd.DataFrame(
        {
            "Year": [2021, 2022],
            "Built-up": [400.0, 410.0],
            "Composite_NoData_pct": [0.0, 0.0],
        }
    )
    with pytest.raises(DataQualityGateError) as exc_info:
        validate_quality_gate(
            df_areas,
            overall_accuracy=0.75,
            gross_gain_km2=50.0,
            gross_loss_km2=20.0,  # 20 / 50 = 40% > 30%
        )
    assert "Built-up loss (20.00 km²) is 40.00% of gross gain (50.00 km²)" in str(exc_info.value)
