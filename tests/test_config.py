"""
Unit tests for YAML configuration schemas and spatial boundary integrity.
Validates required top-level keys, coordinate reference systems, and bounding box geometry.
"""

from pathlib import Path

import yaml


def test_ahmedabad_config_structure_and_bbox():
    """
    Validates configs/ahmedabad.yaml:
      1. File exists and is valid YAML.
      2. Required top-level keys exist: 'city', 'spatial', 'temporal', 'stac'.
      3. Bounding box exists and satisfies west < east (lon) and south < north (lat).
      4. Bounding box coordinates are within valid geographic ranges.
    """
    config_path = Path("configs/ahmedabad.yaml")
    assert config_path.exists(), f"Configuration file not found: {config_path.resolve()}"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), "Configuration must load as a dictionary."

    # 1. Check required top-level keys
    required_keys = ["city", "spatial", "temporal", "stac"]
    for key in required_keys:
        assert key in config, f"Missing required configuration section: '{key}'"

    # 2. Check spatial keys & bbox
    spatial = config["spatial"]
    assert "bbox" in spatial, "Missing 'spatial.bbox' in configuration."
    bbox = spatial["bbox"]
    assert isinstance(bbox, list), "spatial.bbox must be a list."
    assert (
        len(bbox) == 4
    ), f"spatial.bbox must have 4 elements [min_lon, min_lat, max_lon, max_lat], got {len(bbox)}."

    min_lon, min_lat, max_lon, max_lat = bbox

    # 3. Check bounding box geometric integrity
    assert min_lon < max_lon, f"Invalid bbox longitude: west ({min_lon}) must be < east ({max_lon})"
    assert (
        min_lat < max_lat
    ), f"Invalid bbox latitude: south ({min_lat}) must be < north ({max_lat})"

    # 4. Check coordinate range limits for Ahmedabad region (WGS84)
    assert -180.0 <= min_lon <= 180.0
    assert -180.0 <= max_lon <= 180.0
    assert -90.0 <= min_lat <= 90.0
    assert -90.0 <= max_lat <= 90.0

    # Ahmedabad specific sanity check (roughly 72E, 23N)
    assert 70.0 < min_lon < 75.0
    assert 20.0 < min_lat < 25.0
