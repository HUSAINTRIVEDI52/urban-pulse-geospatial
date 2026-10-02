"""
UrbanPulse - Four-Stratum Change Validation Scoring Module (Olofsson et al. 2014)
Reads filled visual validation CSV samples (or blind sample joined with key file),
filters out 'unclear' annotations, and computes:
1. Per-stratum counts & rates for True Gain, True Loss, Persistent Built, Persistent Non-Built,
   and Built-up at start and end dates.
2. Area-adjusted Gross Gain, Gross Loss, and NET Change (km²) with 95% Confidence Intervals
   using stratified area weights (Strata A, B, C, D).
3. Mapped vs Area-Adjusted comparisons for Gain, Loss, and Net Change.
4. Continuity / Wilson proportion correction when p=0 or p=1 so variance never collapses to zero.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_boolean_label(val: Any) -> int | None:
    """
    Parses human annotation into 1 (Built-up), 0 (Not built-up), or None (Unclear / Missing).
    """
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ["unclear", "?", "na", "n/a", "none", "", "null", "nan"]:
        return None
    if s in ["1", "1.0", "true", "t", "built", "built-up", "yes", "y", "urban"]:
        return 1
    if s in ["0", "0.0", "false", "f", "not built", "non-built", "no", "n", "non-urban", "veg", "agri", "water", "open"]:
        return 0
    try:
        f = float(s)
        return 1 if f >= 0.5 else 0
    except ValueError:
        return None


def calculate_proportion_variance(x: int, n: int) -> float:
    """
    Calculates sample variance for a binomial proportion p = x / n.
    If x == 0 or x == n, applies a continuity / Laplace-Wilson correction
    p_adj = (x + 1) / (n + 2) so variance never collapses to 0.
    """
    if n <= 1:
        return 0.0
    p = x / n
    if 0 < x < n:
        # Standard unbiased sample variance of sample proportion
        return (p * (1.0 - p)) / (n - 1)
    else:
        # Continuity correction (Laplace / Wilson adjusted variance)
        p_adj = (x + 1.0) / (n + 2.0)
        return (p_adj * (1.0 - p_adj)) / n


def calculate_stratified_change_metrics(
    df_eval: pd.DataFrame,
    strata_areas_km2: dict[str, float],
) -> dict[str, Any]:
    """
    Computes per-stratum counts, accuracies, and area-adjusted gross gain,
    gross loss, and net change with 95% CIs across all four strata (A, B, C, D).

    Args:
        df_eval: DataFrame with columns ['stratum', 'ref_start', 'ref_end'].
        strata_areas_km2: Dict of mapped areas in km² for {'A': ..., 'B': ..., 'C': ..., 'D': ...}.
    """
    strata_keys = ["A", "B", "C", "D"]
    total_aoi_km2 = sum(float(strata_areas_km2.get(k, 0.0)) for k in strata_keys)

    mapped_gain_km2 = float(strata_areas_km2.get("A", 0.0))
    mapped_loss_km2 = float(strata_areas_km2.get("D", 0.0))
    mapped_net_km2 = mapped_gain_km2 - mapped_loss_km2

    strata_metrics: dict[str, Any] = {}

    adj_gain_sum = 0.0
    adj_loss_sum = 0.0
    adj_net_sum = 0.0

    var_gain_sum = 0.0
    var_loss_sum = 0.0
    var_net_sum = 0.0

    for st in strata_keys:
        st_df = df_eval[df_eval["stratum"] == st]
        n_h = len(st_df)
        area_h = float(strata_areas_km2.get(st, 0.0))

        if n_h == 0:
            strata_metrics[st] = {
                "sample_size": 0,
                "mapped_area_km2": area_h,
                "true_gain_count": 0,
                "true_loss_count": 0,
                "true_built_start_count": 0,
                "true_built_end_count": 0,
                "accuracy": 0.0,
                "gain_proportion": 0.0,
                "loss_proportion": 0.0,
                "var_gain": 0.0,
                "var_loss": 0.0,
            }
            continue

        ref_start = st_df["ref_start"].values
        ref_end = st_df["ref_end"].values

        # Category indicators
        is_gain = (ref_start == 0) & (ref_end == 1)
        is_loss = (ref_start == 1) & (ref_end == 0)
        is_persist_built = (ref_start == 1) & (ref_end == 1)
        is_persist_nonbuilt = (ref_start == 0) & (ref_end == 0)
        is_built_start = (ref_start == 1)
        is_built_end = (ref_end == 1)

        x_gain = int(np.sum(is_gain))
        x_loss = int(np.sum(is_loss))
        x_start_built = int(np.sum(is_built_start))
        x_end_built = int(np.sum(is_built_end))

        p_gain = x_gain / n_h
        p_loss = x_loss / n_h
        p_start = x_start_built / n_h
        p_end = x_end_built / n_h

        # Accuracy definition per stratum
        if st == "A":
            correct_count = int(np.sum(is_gain))
        elif st == "B":
            correct_count = int(np.sum(is_persist_built))
        elif st == "C":
            correct_count = int(np.sum(is_persist_nonbuilt))
        elif st == "D":
            correct_count = int(np.sum(is_loss))
        else:
            correct_count = 0
        acc_h = correct_count / n_h

        # Proportion Variances (with continuity correction for p=0 or 1)
        var_prop_gain = calculate_proportion_variance(x_gain, n_h)
        var_prop_loss = calculate_proportion_variance(x_loss, n_h)

        # Net change variable: d_i = ref_end - ref_start = is_gain - is_loss
        d_vals = (is_gain.astype(float) - is_loss.astype(float))
        mean_d = p_gain - p_loss
        if n_h > 1:
            sample_var_d = np.var(d_vals, ddof=1)
            if sample_var_d == 0.0:
                # All points identical, use continuity correction
                var_mean_d = var_prop_gain + var_prop_loss
            else:
                var_mean_d = sample_var_d / n_h
        else:
            var_mean_d = var_prop_gain + var_prop_loss

        # Area contributions
        st_gain_km2 = area_h * p_gain
        st_loss_km2 = area_h * p_loss
        st_net_km2 = area_h * mean_d

        st_var_gain = (area_h ** 2) * var_prop_gain
        st_var_loss = (area_h ** 2) * var_prop_loss
        st_var_net = (area_h ** 2) * var_mean_d

        adj_gain_sum += st_gain_km2
        adj_loss_sum += st_loss_km2
        adj_net_sum += st_net_km2

        var_gain_sum += st_var_gain
        var_loss_sum += st_var_loss
        var_net_sum += st_var_net

        strata_metrics[st] = {
            "sample_size": n_h,
            "mapped_area_km2": round(area_h, 4),
            "true_gain_count": x_gain,
            "true_loss_count": x_loss,
            "true_built_start_count": x_start_built,
            "true_built_end_count": x_end_built,
            "true_persistent_built_count": int(np.sum(is_persist_built)),
            "true_persistent_nonbuilt_count": int(np.sum(is_persist_nonbuilt)),
            "accuracy": round(acc_h, 4),
            "gain_proportion": round(p_gain, 4),
            "loss_proportion": round(p_loss, 4),
            "built_start_proportion": round(p_start, 4),
            "built_end_proportion": round(p_end, 4),
            "estimated_gain_km2": round(st_gain_km2, 4),
            "estimated_loss_km2": round(st_loss_km2, 4),
            "estimated_net_km2": round(st_net_km2, 4),
            "var_gain_km2": round(st_var_gain, 4),
            "var_loss_km2": round(st_var_loss, 4),
            "var_net_km2": round(st_var_net, 4),
        }

    # Standard errors and 95% CIs
    se_gain = float(np.sqrt(var_gain_sum))
    ci95_gain = float(1.96 * se_gain)

    se_loss = float(np.sqrt(var_loss_sum))
    ci95_loss = float(1.96 * se_loss)

    se_net = float(np.sqrt(var_net_sum))
    ci95_net = float(1.96 * se_net)

    return {
        "strata_metrics": strata_metrics,
        "total_aoi_km2": round(total_aoi_km2, 4),
        "total_evaluated_points": len(df_eval),
        # Gross Gain
        "mapped_gain_km2": round(mapped_gain_km2, 4),
        "adjusted_gain_km2": round(adj_gain_sum, 4),
        "se_gain_km2": round(se_gain, 4),
        "ci95_gain_km2": round(ci95_gain, 4),
        "ci_lower_gain_km2": round(max(0.0, adj_gain_sum - ci95_gain), 4),
        "ci_upper_gain_km2": round(adj_gain_sum + ci95_gain, 4),
        # Gross Loss
        "mapped_loss_km2": round(mapped_loss_km2, 4),
        "adjusted_loss_km2": round(adj_loss_sum, 4),
        "se_loss_km2": round(se_loss, 4),
        "ci95_loss_km2": round(ci95_loss, 4),
        "ci_lower_loss_km2": round(max(0.0, adj_loss_sum - ci95_loss), 4),
        "ci_upper_loss_km2": round(adj_loss_sum + ci95_loss, 4),
        # Net Change
        "mapped_net_km2": round(mapped_net_km2, 4),
        "adjusted_net_km2": round(adj_net_sum, 4),
        "se_net_km2": round(se_net, 4),
        "ci95_net_km2": round(ci95_net, 4),
        "ci_lower_net_km2": round(adj_net_sum - ci95_net, 4),
        "ci_upper_net_km2": round(adj_net_sum + ci95_net, 4),
    }


def score_change_validation(
    sample_csv_path: str | Path,
    key_csv_path: str | Path | None = None,
    strata_areas_km2: dict[str, float] | None = None,
    city: str | None = None,
    data_dir: str | Path = PROJECT_ROOT / "data",
) -> dict[str, Any]:
    """
    Main entry point to score filled change validation sample CSV.
    Supports joining blind sample with separate change_sample_key.csv.
    """
    csv_file = Path(sample_csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Validation sample CSV not found: {csv_file.resolve()}")

    df = pd.read_csv(csv_file)

    # Check for stratum column or join key file
    if "stratum" not in df.columns:
        if key_csv_path is not None and Path(key_csv_path).exists():
            key_file = Path(key_csv_path)
        else:
            cand_key = csv_file.parent / "change_sample_key.csv"
            if cand_key.exists():
                key_file = cand_key
            else:
                raise ValueError(
                    f"Sample CSV '{csv_file.name}' is blind (no 'stratum' column), and key file "
                    f"not found at '{cand_key.resolve()}'."
                )
        df_key = pd.read_csv(key_file)
        if "id" not in df.columns or "id" not in df_key.columns:
            raise ValueError("Cannot join sample CSV and key CSV without 'id' column.")
        df = df.merge(df_key[["id", "stratum"]], on="id", how="left")

    for col in ["id", "stratum", "built_start", "built_end"]:
        if col not in df.columns:
            raise ValueError(f"Sample DataFrame missing required column: '{col}'")

    # Parse annotations
    df["ref_start"] = df["built_start"].apply(parse_boolean_label)
    df["ref_end"] = df["built_end"].apply(parse_boolean_label)

    # Exclude 'unclear' / incomplete annotations
    valid_mask = df["ref_start"].notna() & df["ref_end"].notna()
    df_eval = df[valid_mask].copy()
    n_excluded = len(df) - len(df_eval)

    # Resolve Strata Areas if not explicitly provided
    if strata_areas_km2 is None:
        meta_json = csv_file.parent / "sample_strata_metadata.json"
        if meta_json.exists():
            with open(meta_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
                strata_areas_km2 = meta.get("strata_areas_km2", {})
        else:
            strata_areas_km2 = {"A": 116.0, "B": 358.0, "C": 1635.0, "D": 58.0}

    # Format stratum keys (A, B, C, D)
    areas = {
        "A": float(strata_areas_km2.get("A", strata_areas_km2.get("stratum_a", 0.0))),
        "B": float(strata_areas_km2.get("B", strata_areas_km2.get("stratum_b", 0.0))),
        "C": float(strata_areas_km2.get("C", strata_areas_km2.get("stratum_c", 0.0))),
        "D": float(strata_areas_km2.get("D", strata_areas_km2.get("stratum_d", 0.0))),
    }

    scoring_results = calculate_stratified_change_metrics(df_eval, areas)
    scoring_results["total_samples"] = len(df)
    scoring_results["excluded_unclear_count"] = n_excluded
    scoring_results["csv_path"] = str(csv_file.resolve())

    # Print Formatted Report
    print("=" * 96)
    print(f"[*] UrbanPulse 4-Stratum Change Validation Report (Olofsson et al. 2014)")
    print(f"    - Input Sample CSV : {csv_file.resolve()}")
    print(f"    - Total Samples    : {len(df)}")
    print(f"    - Evaluated Points : {len(df_eval)} ({n_excluded} excluded as 'unclear' or blank)")
    print("=" * 96)

    strata_names = {
        "A": "Stratum A (Mapped Gain)",
        "B": "Stratum B (Persistent Built)",
        "C": "Stratum C (Persistent Non-built)",
        "D": "Stratum D (Mapped Loss)",
    }

    report_rows = []
    for st in ["A", "B", "C", "D"]:
        sm = scoring_results["strata_metrics"].get(st, {})
        report_rows.append({
            "Stratum": strata_names[st],
            "Area (km²)": f"{sm.get('mapped_area_km2', 0.0):.2f}",
            "N": sm.get("sample_size", 0),
            "Accuracy": f"{sm.get('accuracy', 0.0)*100:.2f}%",
            "True Gain (N)": sm.get("true_gain_count", 0),
            "True Loss (N)": sm.get("true_loss_count", 0),
            "Start Built (N)": sm.get("true_built_start_count", 0),
            "End Built (N)": sm.get("true_built_end_count", 0),
            "Est Gain (km²)": f"{sm.get('estimated_gain_km2', 0.0):.2f}",
            "Est Loss (km²)": f"{sm.get('estimated_loss_km2', 0.0):.2f}",
        })

    print("\n" + "-" * 96)
    print("PER-STRATUM EVALUATION & CHANGE MATRIX COUNTS:")
    print("-" * 96)
    print(pd.DataFrame(report_rows).to_string(index=False))

    # Mapped vs Adjusted Comparison Table
    comp_rows = [
        {
            "Metric": "Gross Built-up Gain",
            "Mapped (km²)": f"{scoring_results['mapped_gain_km2']:.2f}",
            "Adjusted Area (km²)": f"{scoring_results['adjusted_gain_km2']:.2f}",
            "Standard Error (SE)": f"±{scoring_results['se_gain_km2']:.2f}",
            "95% Confidence Interval": f"[{scoring_results['ci_lower_gain_km2']:.2f}, {scoring_results['ci_upper_gain_km2']:.2f}]",
        },
        {
            "Metric": "Gross Built-up Loss",
            "Mapped (km²)": f"{scoring_results['mapped_loss_km2']:.2f}",
            "Adjusted Area (km²)": f"{scoring_results['adjusted_loss_km2']:.2f}",
            "Standard Error (SE)": f"±{scoring_results['se_loss_km2']:.2f}",
            "95% Confidence Interval": f"[{scoring_results['ci_lower_loss_km2']:.2f}, {scoring_results['ci_upper_loss_km2']:.2f}]",
        },
        {
            "Metric": "Net Built-up Change",
            "Mapped (km²)": f"{scoring_results['mapped_net_km2']:+.2f}",
            "Adjusted Area (km²)": f"{scoring_results['adjusted_net_km2']:+.2f}",
            "Standard Error (SE)": f"±{scoring_results['se_net_km2']:.2f}",
            "95% Confidence Interval": f"[{scoring_results['ci_lower_net_km2']:+.2f}, {scoring_results['ci_upper_net_km2']:+.2f}]",
        },
    ]

    print("\n" + "-" * 96)
    print("MAPPED VS AREA-ADJUSTED CHANGE ESTIMATION (Olofsson et al. 2014):")
    print("-" * 96)
    print(pd.DataFrame(comp_rows).to_string(index=False))
    print("=" * 96 + "\n")

    return scoring_results


def main():
    parser = argparse.ArgumentParser(
        description="Score filled visual change validation CSV and compute area-adjusted built-up change with 95% CI."
    )
    parser.add_argument("--sample-csv", type=Path, required=True, help="Path to filled change_sample_blind.csv or change_sample.csv")
    parser.add_argument("--key-csv", type=Path, default=None, help="Path to change_sample_key.csv (optional if already in same dir or in sample csv)")
    parser.add_argument("--city", type=str, default=None, help="City key (optional)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Data directory")

    args = parser.parse_args()
    score_change_validation(
        sample_csv_path=args.sample_csv,
        key_csv_path=args.key_csv,
        city=args.city,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()

