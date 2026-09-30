"""
Unit and integration tests for FastAPI service endpoints.
Tests health checks, metadata/analytics payloads, PNG binary streams, and 404 error handling.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


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
    assert len(response.content) > 1000


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
    assert len(response.content) > 1000


def test_get_change_overlay_unknown_years_404():
    """Validates /change/ahmedabad/1990/2000 returns 404."""
    response = client.get("/change/ahmedabad/1990/2000")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
