"""
Unit and integration tests for PostGIS database loader.
Tests idempotent parsing, upsert logic, schema definitions, and pipeline execution logging.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd

from pipeline.load_db import (
    CLASS_NAME_TO_ID,
    PROJECT_CLASSES,
    load_city_metadata,
    load_lulc_stats,
    load_metrics,
    load_rings,
    log_pipeline_run,
)


def create_mock_connection():
    """Helper to create a fully configured mock psycopg2 connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.connection = mock_conn
    mock_conn.encoding = "UTF8"
    mock_cursor.mogrify.side_effect = lambda t, r: b"(1, 'a', 2.0)"
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn, mock_cursor


def test_project_classes_mapping():
    """Validates class dictionary mappings and completeness."""
    assert len(PROJECT_CLASSES) == 5
    assert PROJECT_CLASSES[1] == "Built-up"
    assert "built-up" in CLASS_NAME_TO_ID
    assert CLASS_NAME_TO_ID["built-up"] == 1


def test_load_city_metadata_mocked(tmp_path):
    """Validates loading city metadata and bounding box envelope."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    cfg_file = config_dir / "ahmedabad.yaml"
    cfg_file.write_text(
        """
city:
  name: "Ahmedabad"
  state: "Gujarat"
  country: "India"
  center:
    lat: 23.0225
    lon: 72.5714
spatial:
  bbox: [72.356712, 22.814991, 72.802746, 23.22784]
""",
        encoding="utf-8",
    )

    mock_conn, mock_cursor = create_mock_connection()

    res = load_city_metadata(mock_conn, city="ahmedabad", config_dir=config_dir)

    assert res["id"] == "ahmedabad"
    assert res["name"] == "Ahmedabad"
    assert len(res["bbox"]) == 4
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()


def test_load_lulc_stats_mocked(tmp_path):
    """Validates reading class areas CSV and executing batch upserts."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "ahmedabad_class_areas.csv"

    df = pd.DataFrame(
        [
            {
                "Year": 2018,
                "Built-up": 364.24,
                "Vegetation": 691.46,
                "Water": 58.43,
                "Agriculture": 1008.03,
                "Open land": 45.67,
                "Total_Area_km2": 2167.83,
            },
            {
                "Year": 2024,
                "Built-up": 440.88,
                "Vegetation": 459.99,
                "Water": 29.88,
                "Agriculture": 1148.55,
                "Open land": 84.82,
                "Total_Area_km2": 2164.12,
            },
        ]
    )
    df.to_csv(csv_file, index=False)

    mock_conn, mock_cursor = create_mock_connection()

    count = load_lulc_stats(mock_conn, city_id="ahmedabad", data_dir=data_dir)

    # 2 years * 5 classes = 10 rows
    assert count == 10
    mock_conn.commit.assert_called_once()


def test_load_rings_mocked(tmp_path):
    """Validates ring gradient data ingestion."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "ahmedabad_rings.csv"

    df = pd.DataFrame(
        [
            {
                "year": 2024,
                "ring_start_km": 0.0,
                "ring_end_km": 2.0,
                "builtup_km2": 11.1,
                "valid_km2": 12.6,
                "builtup_pct": 88.1,
            },
            {
                "year": 2024,
                "ring_start_km": 2.0,
                "ring_end_km": 4.0,
                "builtup_km2": 32.5,
                "valid_km2": 37.7,
                "builtup_pct": 86.2,
            },
        ]
    )
    df.to_csv(csv_file, index=False)

    mock_conn, mock_cursor = create_mock_connection()
    count = load_rings(mock_conn, city_id="ahmedabad", data_dir=data_dir)

    assert count == 2
    mock_conn.commit.assert_called_once()


def test_load_metrics_mocked(tmp_path):
    """Validates sprawl metrics ingestion."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    csv_file = data_dir / "ahmedabad_metrics.csv"

    df = pd.DataFrame(
        [
            {
                "city": "Ahmedabad",
                "year": 2024,
                "builtup_km2": 440.88,
                "annual_growth_pct": -0.67,
                "cagr_from_start_pct": 3.23,
                "shannon_entropy": 0.9516,
                "core_share_0_6km_pct": 22.59,
                "periphery_share_gt_12km_pct": 32.27,
            }
        ]
    )
    df.to_csv(csv_file, index=False)

    mock_conn, mock_cursor = create_mock_connection()
    count = load_metrics(mock_conn, city_id="ahmedabad", data_dir=data_dir)

    assert count == 1
    mock_conn.commit.assert_called_once()


def test_log_pipeline_run_mocked():
    """Validates recording pipeline audit log."""
    mock_conn, mock_cursor = create_mock_connection()
    mock_cursor.fetchone.return_value = [42]

    run_id = log_pipeline_run(
        mock_conn,
        city_id="ahmedabad",
        year=2024,
        status="SUCCESS",
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )

    assert run_id == 42
    mock_conn.commit.assert_called_once()
