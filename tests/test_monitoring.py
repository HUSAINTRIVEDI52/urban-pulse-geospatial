"""
Unit tests for UrbanPulse Prometheus & Grafana Monitoring stack.
Validates /metrics FastAPI endpoint, pipeline metrics exporter, alert rules, and Grafana dashboard JSON.
"""

import json
from pathlib import Path
import pytest
import yaml
from fastapi.testclient import TestClient

from api.main import app
from pipeline.metrics_exporter import export_pipeline_metrics


MONITORING_DIR = Path(__file__).resolve().parent.parent / "monitoring"


def test_api_metrics_endpoint():
    """Validates that /metrics exposes Prometheus formatted metrics for HTTP requests."""
    client = TestClient(app)
    # Generate requests
    client.get("/health")
    client.get("/cities")

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests" in response.text
    assert "python_info" in response.text


def test_export_pipeline_metrics_textfile(tmp_path):
    """Validates pipeline metrics export to Prometheus textfile format."""
    export_pipeline_metrics(
        city="testcity",
        status="SUCCESS",
        duration_seconds=123.45,
        nodata_pct=0.5,
        scenes_count=6,
        year=2024,
        textfile_dir=tmp_path,
    )
    prom_file = tmp_path / "testcity_metrics.prom"
    assert prom_file.exists()
    content = prom_file.read_text(encoding="utf-8")
    assert 'pipeline_run_duration_seconds{city="testcity"} 123.45' in content
    assert 'pipeline_run_status{city="testcity"} 1.0' in content
    assert 'pipeline_nodata_percentage{city="testcity",year="2024"} 0.5' in content
    assert 'pipeline_scenes_used_total{city="testcity",year="2024"} 6.0' in content


def test_prometheus_configs_and_alert_rules():
    """Validates Prometheus scrape config and 10-day pipeline alert rule."""
    prom_yaml = MONITORING_DIR / "prometheus" / "prometheus.yml"
    assert prom_yaml.exists()
    with open(prom_yaml, "r", encoding="utf-8") as f:
        prom_cfg = yaml.safe_load(f)
    assert "scrape_configs" in prom_cfg

    alert_yaml = MONITORING_DIR / "prometheus" / "alert_rules.yml"
    assert alert_yaml.exists()
    with open(alert_yaml, "r", encoding="utf-8") as f:
        alert_cfg = yaml.safe_load(f)

    rules = alert_cfg["groups"][0]["rules"]
    pipeline_alert = next((r for r in rules if r.get("alert") == "PipelineNotSucceededIn10Days"), None)
    assert pipeline_alert is not None
    assert "864000" in pipeline_alert["expr"] or "10" in pipeline_alert["expr"]


def test_grafana_dashboard_json_structure():
    """Validates Grafana dashboard JSON contains required panels and metrics queries."""
    dash_path = MONITORING_DIR / "grafana" / "dashboards" / "urbanpulse_overview.json"
    assert dash_path.exists()
    with open(dash_path, "r", encoding="utf-8") as f:
        dash = json.load(f)

    assert dash.get("title") == "UrbanPulse - System & Pipeline Overview"
    panels = dash.get("panels", [])
    assert len(panels) >= 5

    titles = [p.get("title", "") for p in panels]
    assert any("Status" in t for t in titles)
    assert any("Duration" in t for t in titles)
    assert any("NoData" in t for t in titles)
    assert any("Request Rate" in t for t in titles)
    assert any("p95" in t for t in titles)
