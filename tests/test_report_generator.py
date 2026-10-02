"""
Unit tests for pipeline/generate_report.py
Tests synthetic CSV metric extraction, HTML generation, and missing section handling.
"""

from pathlib import Path
import pandas as pd
import pytest

from pipeline.generate_report import (
    extract_city_data,
    render_html_report,
    collect_test_stats,
    get_env_versions,
)


@pytest.fixture
def synthetic_workspace(tmp_path: Path):
    """Creates a minimal synthetic city data directory and outputs."""
    data_dir = tmp_path / "data"
    web_dir = tmp_path / "web_data"
    output_html = tmp_path / "docs" / "report" / "index.html"

    city_dir = data_dir / "testcity"
    city_dir.mkdir(parents=True)
    web_dir.mkdir(parents=True)

    # 1. Synthetic cleanup_summary.csv
    df_cleanup = pd.DataFrame([
        {
            "Year": 2018,
            "Raw_Builtup_km2": 150.50,
            "Clean_Builtup_km2": 140.25,
            "Net_Change_km2": -10.25,
            "Rule1_Spikes_Removed_px": 500,
            "Rule2_Persisted_Added_px": 0,
        },
        {
            "Year": 2024,
            "Raw_Builtup_km2": 210.80,
            "Clean_Builtup_km2": 205.75,
            "Net_Change_km2": -5.05,
            "Rule1_Spikes_Removed_px": 120,
            "Rule2_Persisted_Added_px": 340,
        },
    ])
    df_cleanup.to_csv(data_dir / "testcity_cleanup_summary.csv", index=False)

    # 2. Synthetic metrics.csv
    df_metrics = pd.DataFrame([
        {
            "city": "Testcity",
            "year": 2018,
            "builtup_km2": 140.25,
            "annual_growth_pct": 0.0,
            "cagr_from_start_pct": 0.0,
            "shannon_entropy": 0.8850,
            "core_builtup_0_6km_km2": 50.0,
            "core_share_0_6km_pct": 35.65,
            "periphery_builtup_gt_12km_km2": 30.0,
            "periphery_share_gt_12km_pct": 21.39,
        },
        {
            "city": "Testcity",
            "year": 2024,
            "builtup_km2": 205.75,
            "annual_growth_pct": 6.5,
            "cagr_from_start_pct": 6.58,
            "shannon_entropy": 0.9320,
            "core_builtup_0_6km_km2": 55.0,
            "core_share_0_6km_pct": 26.73,
            "periphery_builtup_gt_12km_km2": 65.0,
            "periphery_share_gt_12km_pct": 31.59,
        },
    ])
    df_metrics.to_csv(data_dir / "testcity_metrics.csv", index=False)

    # 3. Synthetic rings.csv
    df_rings = pd.DataFrame([
        {"ring_id": 0, "ring_label": "0-2 km", "distance_km": 2, "builtup_density_pct": 78.5, "year": 2018},
        {"ring_id": 1, "ring_label": "2-4 km", "distance_km": 4, "builtup_density_pct": 55.2, "year": 2018},
        {"ring_id": 0, "ring_label": "0-2 km", "distance_km": 2, "builtup_density_pct": 84.1, "year": 2024},
        {"ring_id": 1, "ring_label": "2-4 km", "distance_km": 4, "builtup_density_pct": 68.9, "year": 2024},
    ])
    df_rings.to_csv(data_dir / "testcity_rings.csv", index=False)

    # 4. Synthetic transition_2018_2024.csv
    df_trans = pd.DataFrame(
        [
            [140.0, 0.0, 0.0, 0.0, 0.25],
            [10.0, 80.0, 0.0, 2.0, 0.0],
            [0.5, 0.0, 25.0, 0.0, 0.0],
            [45.0, 0.0, 0.0, 110.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 40.0],
        ],
        index=["Built-up", "Vegetation", "Water", "Agriculture", "Open Land"],
        columns=["Built-up", "Vegetation", "Water", "Agriculture", "Open Land"],
    )
    df_trans.to_csv(data_dir / "testcity_transition_2018_2024.csv")

    # 5. Synthetic diagnostics.csv
    df_diag = pd.DataFrame([
        {
            "year": 2018,
            "scenes_used": 6,
            "scene_dates": "2018-11-10; 2018-12-05",
            "mgrs_tiles": "43QDG",
            "nodata_pct": 0.0,
            "mean_cloud_pct": 0.45,
            "stable_mean_red": 0.1245,
            "stable_mean_nir": 0.2104,
            "stable_mean_ndvi": 0.2568,
            "stable_mean_ndbi": 0.1842,
        },
        {
            "year": 2024,
            "scenes_used": 8,
            "scene_dates": "2024-11-12; 2024-12-02",
            "mgrs_tiles": "43QDG",
            "nodata_pct": 0.0,
            "mean_cloud_pct": 0.32,
            "stable_mean_red": 0.1251,
            "stable_mean_nir": 0.2098,
            "stable_mean_ndvi": 0.2531,
            "stable_mean_ndbi": 0.1855,
        },
    ])
    df_diag.to_csv(data_dir / "testcity_diagnostics.csv", index=False)

    return {
        "data_dir": data_dir,
        "web_dir": web_dir,
        "output_html": output_html,
        "city": "testcity",
    }


def test_synthetic_report_generation(synthetic_workspace):
    """Verifies that synthetic numbers appear in the generated HTML and no missing notices occur."""
    ws = synthetic_workspace
    city_data = extract_city_data(ws["city"], ws["data_dir"], ws["web_dir"])

    assert city_data["city_name"] == "Testcity"
    assert "cleanup_summary" in city_data
    assert "rings_pivot" in city_data
    assert "sprawl_metrics" in city_data
    assert "transition_matrix" in city_data
    assert "gain_loss" in city_data
    assert "diagnostics" in city_data

    # Render HTML
    render_html_report(
        cities_data=[city_data],
        output_html_path=ws["output_html"],
        git_commit="test_sha_12345",
        test_stats={"total_tests": 42},
    )

    assert ws["output_html"].exists()
    content = ws["output_html"].read_text(encoding="utf-8")

    # Verify synthetic numbers are in the HTML
    assert "140.25 km&sup2;" in content
    assert "205.75 km&sup2;" in content
    assert "0.8850" in content
    assert "0.9320" in content
    assert "78.5%" in content
    assert "test_sha" in content
    assert "42" in content
    assert "Testcity" in content


def test_missing_input_skips_gracefully_with_note(tmp_path: Path):
    """Verifies that missing files are skipped with a clear user-facing notice rather than crashing."""
    data_dir = tmp_path / "empty_data"
    web_dir = tmp_path / "empty_web"
    output_html = tmp_path / "docs" / "report" / "empty_report.html"
    data_dir.mkdir(parents=True)
    web_dir.mkdir(parents=True)

    city_data = extract_city_data("ghostcity", data_dir, web_dir)

    # Should register missing sections
    assert len(city_data["missing_sections"]) > 0
    assert any("cleanup_summary.csv missing" in s for s in city_data["missing_sections"])

    # Generating report should not crash
    render_html_report(
        cities_data=[city_data],
        output_html_path=output_html,
        git_commit="mock_sha",
        test_stats={"total_tests": 10},
    )

    assert output_html.exists()
    content = output_html.read_text(encoding="utf-8")
    assert "Ghostcity" in content
    assert "missing" in content.lower()
