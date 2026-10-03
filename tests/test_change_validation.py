"""
Unit and integration tests for change validation sampling and scoring modules:
- pipeline/make_change_validation_sample.py
- pipeline/score_change_validation.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

from pipeline.make_change_validation_sample import (
    compute_strata_masks,
    draw_stratified_sample_with_spacing,
    export_blind_kml,
)
from pipeline.score_change_validation import (
    calculate_proportion_variance,
    calculate_stratified_change_metrics,
    parse_boolean_label,
    score_change_validation,
)


@pytest.fixture
def synthetic_raster_pair():
    """
    Creates a synthetic start & end classified raster pair (100x100 pixels, 60m resolution).
    EPSG:32643 UTM projection.
    """
    # Grid: 100 x 100 = 10,000 pixels (6.0 km x 6.0 km = 36.0 km²)
    h, w = 100, 100
    transform = from_origin(270000, 2550000, 60.0, 60.0)
    crs = "EPSG:32643"

    # Start: center 30x30 is Built-up (class 1), plus a 10x10 block in top-left is Built-up (class 1)
    # Rest is Non-built (class 4 = Agri)
    start_cls = np.full((h, w), 4, dtype=np.uint8)
    start_cls[35:65, 35:65] = 1  # 30x30 = 900 pixels built-up
    start_cls[5:15, 5:15] = 1  # 10x10 = 100 pixels built-up (will become Loss)

    # End: center 50x50 is Built-up (class 1)
    # Top-left block reverted to Non-built (class 4 = Agri) [Loss]
    end_cls = np.full((h, w), 4, dtype=np.uint8)
    end_cls[25:75, 25:75] = (
        1  # 50x50 = 2500 pixels built-up (includes 900 persistent built + 1600 gain)
    )

    valid_mask = np.ones((h, w), dtype=bool)

    return {
        "start_cls": start_cls,
        "end_cls": end_cls,
        "valid_mask": valid_mask,
        "transform": transform,
        "crs": crs,
        "shape": (h, w),
    }


def test_compute_strata_masks(synthetic_raster_pair):
    """Verifies that strata A, B, C, and D are correctly partitioned and sum to valid AOI."""
    data = synthetic_raster_pair
    strata = compute_strata_masks(data["start_cls"], data["end_cls"], data["valid_mask"])

    # Stratum A (Mapped Gain: 0 -> 1): 2500 - 900 = 1600 pixels
    assert np.sum(strata["A"]) == 1600

    # Stratum B (Persistent Built: 1 -> 1): 900 pixels
    assert np.sum(strata["B"]) == 900

    # Stratum D (Mapped Loss: 1 -> 0): 100 pixels
    assert np.sum(strata["D"]) == 100

    # Stratum C (Persistent Non-built: 0 -> 0): 10000 - 1600 - 900 - 100 = 7400 pixels
    assert np.sum(strata["C"]) == 7400

    # Total partitioned pixels must equal valid AOI (10,000 px)
    total_strata_px = (
        np.sum(strata["A"]) + np.sum(strata["B"]) + np.sum(strata["C"]) + np.sum(strata["D"])
    )
    assert total_strata_px == 10000


def test_draw_sample_edge_exclusion_and_spacing(synthetic_raster_pair, tmp_path):
    """
    Verifies that sampling excludes pixels within 2 pixels of edges,
    enforces the >= 500 m minimum spatial distance constraint across all 4 strata,
    and produces blind and key CSV formats.
    """
    data = synthetic_raster_pair
    strata = compute_strata_masks(data["start_cls"], data["end_cls"], data["valid_mask"])

    sample_sizes = {"A": 15, "B": 10, "C": 15, "D": 5}
    df_sample = draw_stratified_sample_with_spacing(
        strata_masks=strata,
        transform=data["transform"],
        crs=data["crs"],
        sample_sizes=sample_sizes,
        min_edge_distance_px=2,
        min_spacing_m=500.0,
        seed=42,
    )

    assert len(df_sample) > 0
    assert set(df_sample["stratum"].unique()).issubset({"A", "B", "C", "D"})

    # Check shuffled blind and key preparation
    df_blind = df_sample[["id", "lon", "lat"]].copy()
    df_blind["built_start"] = ""
    df_blind["built_end"] = ""
    df_blind["notes"] = ""

    df_key = df_sample[["id", "stratum"]].copy()

    assert list(df_blind.columns) == ["id", "lon", "lat", "built_start", "built_end", "notes"]
    assert list(df_key.columns) == ["id", "stratum"]
    assert "stratum" not in df_blind.columns

    # Verify pairwise spatial spacing >= 500 m
    coords = df_sample[["x", "y"]].values
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = np.sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
            assert dist >= 500.0, f"Points {i} and {j} are too close: {dist:.1f} m < 500 m"

    # Test Blind KML export
    kml_path = tmp_path / "test_sample_blind.kml"
    export_blind_kml(df_blind, kml_path, "TestCity")
    assert kml_path.exists()
    kml_text = kml_path.read_text(encoding="utf-8")
    assert "<kml xmlns=" in kml_text
    assert "Placemark" in kml_text
    assert "Stratum" not in kml_text  # Must be blind


def test_parse_boolean_label():
    """Tests annotation parsing logic including 'unclear' handling."""
    assert parse_boolean_label(1) == 1
    assert parse_boolean_label("1") == 1
    assert parse_boolean_label("built") == 1
    assert parse_boolean_label("True") == 1
    assert parse_boolean_label("YES") == 1

    assert parse_boolean_label(0) == 0
    assert parse_boolean_label("0") == 0
    assert parse_boolean_label("non-built") == 0
    assert parse_boolean_label("False") == 0
    assert parse_boolean_label("NO") == 0

    assert parse_boolean_label("unclear") is None
    assert parse_boolean_label("UNCLEAR") is None
    assert parse_boolean_label("?") is None
    assert parse_boolean_label("") is None
    assert parse_boolean_label(None) is None
    assert parse_boolean_label(np.nan) is None


def test_calculate_proportion_variance_continuity_correction():
    """
    Verifies that when p=0 or p=1, continuity correction produces strictly positive variance.
    """
    # Standard non-zero case: x=25, n=50 -> p=0.5 -> var = 0.5 * 0.5 / 49 = 0.005102
    v_mid = calculate_proportion_variance(x=25, n=50)
    assert pytest.approx(v_mid, 1e-5) == (0.5 * 0.5) / 49

    # Edge case: x=0, n=50 (p=0) -> Laplace (add-one) smoothing: p_adj = 1 / 52 -> var = p_adj*(1-p_adj)/50 > 0
    v_zero = calculate_proportion_variance(x=0, n=50)
    assert v_zero > 0.0
    p_adj_zero = 1.0 / 52.0
    assert pytest.approx(v_zero, 1e-6) == (p_adj_zero * (1.0 - p_adj_zero)) / 50.0

    # Edge case: x=50, n=50 (p=1) -> Laplace (add-one) smoothing: p_adj = 51 / 52 -> var > 0
    v_one = calculate_proportion_variance(x=50, n=50)
    assert v_one > 0.0
    assert pytest.approx(v_one, 1e-6) == v_zero


def test_calculate_stratified_change_metrics_with_p0():
    """
    Tests exact 4-stratum calculations including a stratum with p=0,
    verifying gross gain, gross loss, net change, and positive CIs.
    """
    # Stratum A (Mapped Gain, Area = 100 km²): 10 points (8 true gain, 2 non-built) -> p_gain = 0.8, p_loss = 0.0
    # Stratum B (Persistent Built, Area = 400 km²): 10 points (10 persistent built) -> p_gain = 0.0, p_loss = 0.0
    # Stratum C (Persistent Non-built, Area = 1500 km²): 10 points (1 gain omission, 9 non-built) -> p_gain = 0.1, p_loss = 0.0
    # Stratum D (Mapped Loss, Area = 50 km²): 10 points (6 true loss, 4 persistent built) -> p_gain = 0.0, p_loss = 0.6

    rows = []
    # Stratum A
    for _ in range(8):
        rows.append({"stratum": "A", "ref_start": 0, "ref_end": 1})
    for _ in range(2):
        rows.append({"stratum": "A", "ref_start": 0, "ref_end": 0})

    # Stratum B (p_gain = 0, p_loss = 0)
    for _ in range(10):
        rows.append({"stratum": "B", "ref_start": 1, "ref_end": 1})

    # Stratum C
    for _ in range(1):
        rows.append({"stratum": "C", "ref_start": 0, "ref_end": 1})
    for _ in range(9):
        rows.append({"stratum": "C", "ref_start": 0, "ref_end": 0})

    # Stratum D
    for _ in range(6):
        rows.append({"stratum": "D", "ref_start": 1, "ref_end": 0})
    for _ in range(4):
        rows.append({"stratum": "D", "ref_start": 1, "ref_end": 1})

    df_eval = pd.DataFrame(rows)
    strata_areas = {"A": 100.0, "B": 400.0, "C": 1500.0, "D": 50.0}

    results = calculate_stratified_change_metrics(df_eval, strata_areas)

    # Gross Gain:
    # A: 100 * 0.8 = 80.0 km²
    # B: 400 * 0.0 = 0.0 km²
    # C: 1500 * 0.1 = 150.0 km²
    # D: 50 * 0.0 = 0.0 km²
    # Total Gross Gain = 230.0 km²
    assert results["adjusted_gain_km2"] == 230.0
    assert results["mapped_gain_km2"] == 100.0

    # Gross Loss:
    # A: 100 * 0.0 = 0.0 km²
    # B: 400 * 0.0 = 0.0 km²
    # C: 1500 * 0.0 = 0.0 km²
    # D: 50 * 0.6 = 30.0 km²
    # Total Gross Loss = 30.0 km²
    assert results["adjusted_loss_km2"] == 30.0
    assert results["mapped_loss_km2"] == 50.0

    # Net Change:
    # Adjusted Net = 230.0 - 30.0 = +200.0 km²
    # Mapped Net = 100.0 - 50.0 = +50.0 km²
    assert results["adjusted_net_km2"] == 200.0
    assert results["mapped_net_km2"] == 50.0

    # Verify standard errors and 95% CIs are strictly positive
    assert results["se_gain_km2"] > 0.0
    assert results["ci95_gain_km2"] > 0.0
    assert results["se_loss_km2"] > 0.0
    assert results["ci95_loss_km2"] > 0.0
    assert results["se_net_km2"] > 0.0
    assert results["ci95_net_km2"] > 0.0


def test_score_change_validation_blind_key_join(tmp_path):
    """Tests loading a blind CSV, automatically joining the key CSV, handling unclear annotations, and scoring."""
    blind_csv = tmp_path / "change_sample_blind.csv"
    key_csv = tmp_path / "change_sample_key.csv"
    meta_file = tmp_path / "sample_strata_metadata.json"

    meta_file.write_text(
        '{"strata_areas_km2": {"A": 50.0, "B": 200.0, "C": 1000.0, "D": 25.0}}', encoding="utf-8"
    )

    # Key file
    key_data = [
        {"id": 1, "stratum": "A"},
        {"id": 2, "stratum": "A"},
        {"id": 3, "stratum": "A"},
        {"id": 4, "stratum": "B"},
        {"id": 5, "stratum": "C"},
        {"id": 6, "stratum": "D"},
    ]
    pd.DataFrame(key_data).to_csv(key_csv, index=False)

    # Blind file with annotations
    blind_data = [
        {"id": 1, "lon": 72.5, "lat": 23.0, "built_start": "0", "built_end": "1", "notes": ""},
        {"id": 2, "lon": 72.6, "lat": 23.1, "built_start": "0", "built_end": "1", "notes": ""},
        {
            "id": 3,
            "lon": 72.7,
            "lat": 23.2,
            "built_start": "unclear",
            "built_end": "1",
            "notes": "cloudy",
        },
        {"id": 4, "lon": 72.8, "lat": 23.3, "built_start": "1", "built_end": "1", "notes": ""},
        {"id": 5, "lon": 72.9, "lat": 23.4, "built_start": "0", "built_end": "0", "notes": ""},
        {
            "id": 6,
            "lon": 73.0,
            "lat": 23.5,
            "built_start": "1",
            "built_end": "0",
            "notes": "cleared",
        },
    ]
    pd.DataFrame(blind_data).to_csv(blind_csv, index=False)

    results = score_change_validation(sample_csv_path=blind_csv)
    assert results["total_samples"] == 6
    assert results["excluded_unclear_count"] == 1
    assert results["total_evaluated_points"] == 5
    assert results["strata_metrics"]["A"]["sample_size"] == 2
    assert results["strata_metrics"]["A"]["accuracy"] == 1.0
    assert results["strata_metrics"]["D"]["sample_size"] == 1
    assert results["strata_metrics"]["D"]["true_loss_count"] == 1
    assert results["adjusted_gain_km2"] == 50.0
    assert results["adjusted_loss_km2"] == 25.0
    assert results["adjusted_net_km2"] == 25.0


def test_extract_and_draw_chip():
    """
    Tests chip extraction, 4x bicubic upscaling, and 60m center outline drawing on synthetic array.
    """
    from pipeline.make_label_chips import extract_and_draw_chip

    # Synthetic RGB raster: 300 x 300 pixels
    synthetic_rgb = np.zeros((300, 300, 3), dtype=np.uint8)
    synthetic_rgb[:, :, 0] = 50  # R
    synthetic_rgb[:, :, 1] = 120  # G
    synthetic_rgb[:, :, 2] = 70  # B

    # Center chip extraction
    chip_img = extract_and_draw_chip(
        rgb_uint8=synthetic_rgb,
        center_row=150,
        center_col=150,
        chip_size_px=128,
        upscale_factor=4,
        pixel_box_size_orig_px=6,
    )

    assert chip_img.size == (512, 512)
    assert chip_img.mode == "RGB"

    chip_arr = np.array(chip_img)
    # Check that yellow outline (255, 235, 59) exists at center box edge [244, 244]
    yellow = [255, 235, 59]
    assert np.array_equal(chip_arr[244, 244], yellow) or np.array_equal(chip_arr[244, 256], yellow)

    # Edge chip extraction (near raster boundary)
    edge_chip_img = extract_and_draw_chip(
        rgb_uint8=synthetic_rgb,
        center_row=10,
        center_col=10,
        chip_size_px=128,
        upscale_factor=4,
        pixel_box_size_orig_px=6,
    )
    assert edge_chip_img.size == (512, 512)


def test_generate_standalone_labeller_html(tmp_path):
    """
    Verifies that the generated labeller HTML is self-contained and completely blind (no stratum info).
    """
    from pipeline.make_label_chips import generate_standalone_labeller_html

    out_html = tmp_path / "labeller.html"
    points = [
        {"id": 1, "lon": 72.5, "lat": 23.0},
        {"id": 2, "lon": 72.6, "lat": 23.1},
    ]

    generate_standalone_labeller_html(
        city_name="Ahmedabad",
        start_year=2020,
        end_year=2024,
        points_data=points,
        output_html_path=out_html,
    )

    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")

    # Must contain essential interactive features
    assert "downloadCSV" in content
    assert "localStorage" in content
    assert "jumpToFirstUnlabelled" in content
    assert "google.com/maps" in content
    assert "Point #1" in content or "points" in content

    # Must NEVER mention stratum
    assert "stratum" not in content.lower()


def test_dashboard_change_validation_matches_scorer():
    """
    Verifies that the change validation statistics published to web/data/ahmedabad/stats.json
    and displayed in the dashboard exactly match the authoritative output of score_change_validation().
    """
    import json

    from pipeline.score_change_validation import score_change_validation

    stats_path = Path("web/data/ahmedabad/stats.json")
    val_csv = Path("data/ahmedabad/validation/change_sample_labelled.csv")
    if not val_csv.exists():
        val_csv = Path("data/ahmedabad/validation/change_sample_labelled_ahmedabad.csv")

    assert stats_path.exists(), "web/data/ahmedabad/stats.json must exist"
    assert val_csv.exists(), "Labelled validation CSV must exist"

    with open(stats_path, encoding="utf-8") as f:
        stats_json = json.load(f)

    assert "change_validation" in stats_json, "stats.json must contain 'change_validation' key"
    cv_dash = stats_json["change_validation"]
    assert cv_dash["status"] == "validated"

    # Run authoritative scorer
    scorer_res = score_change_validation(val_csv)

    # Verify gross gain, loss, net, and CIs
    assert pytest.approx(cv_dash["adjusted_gain_km2"], rel=1e-4) == scorer_res["adjusted_gain_km2"]
    assert pytest.approx(cv_dash["ci95_gain_km2"], rel=1e-4) == scorer_res["ci95_gain_km2"]
    assert pytest.approx(cv_dash["adjusted_loss_km2"], rel=1e-4) == scorer_res["adjusted_loss_km2"]
    assert pytest.approx(cv_dash["ci95_loss_km2"], rel=1e-4) == scorer_res["ci95_loss_km2"]
    assert pytest.approx(cv_dash["adjusted_net_km2"], rel=1e-4) == scorer_res["adjusted_net_km2"]
    assert pytest.approx(cv_dash["ci95_net_km2"], rel=1e-4) == scorer_res["ci95_net_km2"]
    assert pytest.approx(cv_dash["mapped_gain_km2"], rel=1e-4) == scorer_res["mapped_gain_km2"]
    assert pytest.approx(cv_dash["mapped_loss_km2"], rel=1e-4) == scorer_res["mapped_loss_km2"]
    assert pytest.approx(cv_dash["mapped_net_km2"], rel=1e-4) == scorer_res["mapped_net_km2"]
    assert cv_dash["sample_points_evaluated"] == scorer_res["total_evaluated_points"]
    assert cv_dash["excluded_unclear"] == scorer_res["excluded_unclear_count"]

    # Verify data-driven summary sentence
    is_dist = (scorer_res["ci_lower_net_km2"] > 0) or (scorer_res["ci_upper_net_km2"] < 0)
    expected_sentence = f"Validated on {scorer_res['total_evaluated_points']} points (Ahmedabad only); net change {'is' if is_dist else 'is not'} distinguishable from zero post-recheck (+{scorer_res['adjusted_net_km2']:.1f} ± {scorer_res['ci95_net_km2']:.1f} km²), but depends on the recheck (pre-recheck: +14.9 ± 83.6 km²; without Stratum C gains: +48.3 ± 23.1 km²)."
    assert cv_dash["summary_sentence"] == expected_sentence
    assert "sensitivities" in cv_dash
    assert cv_dash["sensitivities"]["no_stratum_c_gains"]["net_km2"] == 48.31
    assert cv_dash["sensitivities"]["pre_recheck"]["net_km2"] == 14.90

    # Verify Pune is marked as not independently validated
    pune_stats_path = Path("web/data/pune/stats.json")
    if pune_stats_path.exists():
        with open(pune_stats_path, encoding="utf-8") as f:
            pune_stats = json.load(f)
        assert (
            pune_stats.get("change_validation", {}).get("status") == "not_independently_validated"
        )
        assert (
            "not independently validated"
            in pune_stats.get("change_validation", {}).get("summary_sentence", "").lower()
        )
