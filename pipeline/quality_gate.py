"""
UrbanPulse - Data Quality Gate
Validates satellite composites, land cover classification accuracy, and temporal
time-series consistency before allowing a city pipeline run to be approved.
"""

from typing import Any
import pandas as pd


class DataQualityGateError(Exception):
    """Raised when pipeline outputs fail quality gate constraints."""
    pass


def validate_quality_gate(
    df_areas: pd.DataFrame,
    overall_accuracy: float | None = None,
    max_nodata_pct: float = 5.0,
    max_builtup_change_pct: float = 25.0,
    min_accuracy: float = 0.70,
) -> dict[str, Any]:
    """
    Evaluates pipeline quality gate checks:
    1. Composite NoData percentage <= 5.0% for all years.
    2. Consecutive year-over-year built-up area change <= 25.0%.
    3. Spatial block validation accuracy >= 0.70 (70%).

    Raises DataQualityGateError if any check fails.
    Returns validation summary dictionary if all pass.
    """
    failures = []

    # 1. Check NoData percentage
    if "Composite_NoData_pct" in df_areas.columns:
        bad_nodata = df_areas[df_areas["Composite_NoData_pct"] > max_nodata_pct]
        if not bad_nodata.empty:
            for _, r in bad_nodata.iterrows():
                yr = int(r["Year"])
                pct = float(r["Composite_NoData_pct"])
                failures.append(
                    f"Year {yr} NoData percentage ({pct:.2f}%) exceeds threshold of {max_nodata_pct:.1f}%"
                )

    # 2. Check consecutive built-up area change percentage
    if "Built-up" in df_areas.columns and len(df_areas) > 1:
        df_sorted = df_areas.sort_values("Year").reset_index(drop=True)
        for i in range(1, len(df_sorted)):
            prev_row = df_sorted.iloc[i - 1]
            curr_row = df_sorted.iloc[i]
            prev_year = int(prev_row["Year"])
            curr_year = int(curr_row["Year"])
            prev_builtup = float(prev_row["Built-up"])
            curr_builtup = float(curr_row["Built-up"])

            if prev_builtup > 0:
                pct_change = abs(curr_builtup - prev_builtup) / prev_builtup * 100.0
                if pct_change > max_builtup_change_pct:
                    failures.append(
                        f"Built-up area changed by {pct_change:.2f}% between {prev_year} ({prev_builtup:.1f} km²) "
                        f"and {curr_year} ({curr_builtup:.1f} km²), exceeding maximum allowed {max_builtup_change_pct:.1f}%"
                    )

    # 3. Check classification accuracy threshold
    if overall_accuracy is not None:
        if overall_accuracy < min_accuracy:
            failures.append(
                f"Model validation accuracy ({overall_accuracy * 100:.2f}%) is below minimum threshold of {min_accuracy * 100:.1f}%"
            )

    if failures:
        reason_msg = " | ".join(failures)
        raise DataQualityGateError(f"Data Quality Gate Failed: {reason_msg}")

    return {
        "status": "PASSED",
        "nodata_checked": True,
        "builtup_volatility_checked": True,
        "accuracy_checked": overall_accuracy is not None,
    }
