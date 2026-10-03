"""
UrbanPulse - Data Quality Gate Module
Validates satellite composites, land cover classification accuracy, and temporal
time-series consistency after cleanup before allowing a city pipeline run to be approved.
"""

from typing import Any

import pandas as pd


class DataQualityGateError(Exception):
    """Raised when pipeline outputs fail quality gate constraints."""

    pass


def validate_quality_gate(
    df_areas: pd.DataFrame,
    per_year_accuracies: dict[int, float] | None = None,
    overall_accuracy: float | None = None,
    gross_gain_km2: float | None = None,
    gross_loss_km2: float | None = None,
    max_nodata_pct: float = 5.0,
    max_builtup_change_pct: float = 15.0,
    min_accuracy: float = 0.70,
    max_loss_to_gain_ratio: float = 0.30,
) -> dict[str, Any]:
    """
    Evaluates pipeline quality gate checks on cleaned series:
    1. Composite NoData percentage <= 5.0% for all years.
    2. Year-to-year built-up change <= 15.0% on the cleaned series.
    3. Per-year validation accuracy >= 0.70 (70%) on held-out blocks.
    4. Built-up loss <= 30.0% of gross gain (loss / gain ratio <= 0.30).

    Raises DataQualityGateError if any check fails.
    Returns validation summary dictionary if all pass.
    """
    failures = []

    # 1. Check NoData percentage (<= 5.0%)
    nodata_col = None
    for col in ["Composite_NoData_pct", "nodata_pct", "NoData_pct", "nodata_percentage"]:
        if col in df_areas.columns:
            nodata_col = col
            break

    if nodata_col is not None:
        bad_nodata = df_areas[df_areas[nodata_col] > max_nodata_pct]
        if not bad_nodata.empty:
            for _, r in bad_nodata.iterrows():
                yr = int(r["Year"])
                pct = float(r[nodata_col])
                failures.append(
                    f"Year {yr} NoData percentage ({pct:.2f}%) exceeds threshold of {max_nodata_pct:.1f}%"
                )

    # 2. Check year-to-year built-up change (<= 15.0% on cleaned series)
    builtup_col = None
    for col in ["Built-up", "builtup_km2", "Clean_Builtup_km2", "Built_up"]:
        if col in df_areas.columns:
            builtup_col = col
            break

    if builtup_col is not None and len(df_areas) > 1:
        df_sorted = df_areas.sort_values("Year").reset_index(drop=True)
        for i in range(1, len(df_sorted)):
            prev_row = df_sorted.iloc[i - 1]
            curr_row = df_sorted.iloc[i]
            prev_year = int(prev_row["Year"])
            curr_year = int(curr_row["Year"])
            prev_builtup = float(prev_row[builtup_col])
            curr_builtup = float(curr_row[builtup_col])

            if prev_builtup > 0:
                pct_change = abs(curr_builtup - prev_builtup) / prev_builtup * 100.0
                if pct_change > max_builtup_change_pct:
                    failures.append(
                        f"Built-up area changed by {pct_change:.2f}% between {prev_year} ({prev_builtup:.1f} km²) "
                        f"and {curr_year} ({curr_builtup:.1f} km²), exceeding maximum allowed {max_builtup_change_pct:.1f}%"
                    )

    # 3. Check per-year validation accuracy threshold (>= 0.70)
    if per_year_accuracies:
        for yr, acc in per_year_accuracies.items():
            if acc < min_accuracy:
                failures.append(
                    f"Year {yr} validation accuracy ({acc * 100:.2f}%) is below minimum threshold of {min_accuracy * 100:.1f}%"
                )
    elif overall_accuracy is not None:
        if overall_accuracy < min_accuracy:
            failures.append(
                f"Model validation accuracy ({overall_accuracy * 100:.2f}%) is below minimum threshold of {min_accuracy * 100:.1f}%"
            )

    # 4. Check built-up loss over 30% of gross gain
    if gross_gain_km2 is not None and gross_loss_km2 is not None:
        if gross_gain_km2 > 0:
            loss_ratio = gross_loss_km2 / gross_gain_km2
            if loss_ratio > max_loss_to_gain_ratio:
                failures.append(
                    f"Built-up loss ({gross_loss_km2:.2f} km²) is {loss_ratio * 100:.2f}% of gross gain ({gross_gain_km2:.2f} km²), "
                    f"exceeding maximum allowed {max_loss_to_gain_ratio * 100:.1f}%"
                )
        elif gross_loss_km2 > 0:
            failures.append(
                f"Built-up loss ({gross_loss_km2:.2f} km²) occurred with 0 gross gain, exceeding maximum allowed {max_loss_to_gain_ratio * 100:.1f}%"
            )

    if failures:
        reason_msg = " | ".join(failures)
        raise DataQualityGateError(f"Data Quality Gate Failed: {reason_msg}")

    return {
        "status": "PASSED",
        "nodata_checked": nodata_col is not None,
        "builtup_volatility_checked": builtup_col is not None,
        "accuracy_checked": per_year_accuracies is not None or overall_accuracy is not None,
        "loss_gain_ratio_checked": gross_gain_km2 is not None and gross_loss_km2 is not None,
    }
