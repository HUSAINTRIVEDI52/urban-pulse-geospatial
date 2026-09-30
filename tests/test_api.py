"""
Unit and integration tests for FastAPI service endpoints.
Tests health checks, metadata/analytics payloads, PNG binary streams, and 404 error handling.
Uses synthetic test fixtures so tests run completely independent of local datasets.
"""

import json

import pytest
from fastapi.testclient import TestClient

import api.main
from api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def mock_web_data(tmp_path, monkeypatch):
    """
    Creates a synthetic web/data environment so tests do not rely on local pipeline artifacts.
    """
    web_data_dir = tmp_path / "web" / "data"
    city_dir = web_data_dir / "ahmedabad"
    city_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic meta.json
    synthetic_meta = {
        "city": "Ahmedabad",
        "state": "Gujarat",
        "country": "India",
        "center": [23.0225, 72.5714],
        "bounds": [72.356712, 22.814991, 72.802746, 23.22784],
        "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024],
        "classes": {
            "1": {"name": "Built-up", "color": "#e53e3e"},
            "2": {"name": "Vegetation", "color": "#38a169"},
            "3": {"name": "Water", "color": "#3182ce"},
            "4": {"name": "Agriculture", "color": "#d69e2e"},
            "5": {"name": "Open land", "color": "#d6bcfa"},
        },
    }
    with open(city_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(synthetic_meta, f)

    # 2. Synthetic stats.json
    synthetic_stats = {
        "city": "Ahmedabad",
        "years": [2018, 2024],
        "class_areas": {
            "2018": {"builtup_km2": 364.4, "vegetation_km2": 450.2},
            "2024": {"builtup_km2": 440.9, "vegetation_km2": 410.1},
        },
        "metrics": {
            "2018": {"shannon_entropy": 0.941, "builtup_km2": 364.4},
            "2024": {"shannon_entropy": 0.952, "builtup_km2": 440.9},
        },
        "rings": {
            "2018": [{"ring_start_km": 0, "ring_end_km": 2, "builtup_pct": 82.5}],
            "2024": [{"ring_start_km": 0, "ring_end_km": 2, "builtup_pct": 88.1}],
        },
        "transitions": {
            "2018_2024": {
                "gross_builtup_gain_km2": 82.1,
                "gross_builtup_loss_km2": 5.6,
                "net_builtup_change_km2": 76.5,
            }
        },
    }
    with open(city_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(synthetic_stats, f)

    # 3. Synthetic dummy PNG overlays (minimal valid 1x1 PNG or bytes)
    # Minimal 1x1 PNG transparent binary
    dummy_png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r"
        b"\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    with open(city_dir / "2024.png", "wb") as f:
        f.write(dummy_png_bytes)
    with open(city_dir / "change_2018_2024.png", "wb") as f:
        f.write(dummy_png_bytes)

    # Monkeypatch the API's WEB_DATA_DIR
    monkeypatch.setattr(api.main, "WEB_DATA_DIR", web_data_dir)

    yield web_data_dir


def test_health_endpoint():
    """Validates /health returns 200 with status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "app" in data


def test_cities_endpoint():
    """Validates /cities returns list containing 'ahmedabad'."""
    response = client.get("/cities")
    assert response.status_code == 200
    cities = response.json()
    assert isinstance(cities, list)
    city_ids = [c.get("id") for c in cities]
    assert "ahmedabad" in city_ids


def test_get_meta_ahmedabad():
    """Validates /meta/ahmedabad returns spatial bounds, center, and class schema."""
    response = client.get("/meta/ahmedabad")
    assert response.status_code == 200
    meta = response.json()
    assert meta["city"].lower() == "ahmedabad"
    assert len(meta["center"]) == 2
    assert len(meta["bounds"]) == 4
    assert 2024 in meta["years"]
    assert "classes" in meta


def test_get_meta_unknown_city_404():
    """Validates /meta/nonexistent_city returns 404 with clear message."""
    response = client.get("/meta/nonexistent_city")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_stats_ahmedabad():
    """Validates /stats/ahmedabad returns time-series analytics payload."""
    response = client.get("/stats/ahmedabad")
    assert response.status_code == 200
    stats = response.json()
    assert stats["city"].lower() == "ahmedabad"
    assert "class_areas" in stats
    assert "metrics" in stats
    assert "rings" in stats
    assert "transitions" in stats
    assert len(stats["class_areas"]) > 0


def test_get_stats_unknown_city_404():
    """Validates /stats/nonexistent_city returns 404."""
    response = client.get("/stats/nonexistent_city")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_overlay_valid_png():
    """Validates /overlay/ahmedabad/2024 returns 200 with image/png media type."""
    response = client.get("/overlay/ahmedabad/2024")
    assert response.status_code == 200
    assert "image/png" in response.headers.get("content-type", "")
    assert len(response.content) > 0


def test_get_overlay_unknown_year_404():
    """Validates /overlay/ahmedabad/1980 returns 404."""
    response = client.get("/overlay/ahmedabad/1980")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_change_overlay_valid():
    """Validates /change/ahmedabad/2018/2024 returns 200 with image/png media type."""
    response = client.get("/change/ahmedabad/2018/2024")
    assert response.status_code == 200
    assert "image/png" in response.headers.get("content-type", "")
    assert len(response.content) > 0


def test_get_change_overlay_unknown_years_404():
    """Validates /change/ahmedabad/1990/2000 returns 404."""
    response = client.get("/change/ahmedabad/1990/2000")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
