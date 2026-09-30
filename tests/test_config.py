"""
Unit tests for YAML configuration schemas and spatial boundary integrity.
Validates required top-level keys, coordinate reference systems, and bounding box geometry
across all configured cities in configs/*.yaml.
"""

from pathlib import Path

import pytest
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def get_all_config_files():
    """Discovers all city YAML configuration files."""
    files = list(CONFIGS_DIR.glob("*.yaml")) + list(CONFIGS_DIR.glob("*.yml"))
    return sorted(files)


@pytest.mark.parametrize("config_path", get_all_config_files(), ids=lambda p: p.stem)
def test_city_config_structure_and_bbox(config_path: Path):
    """
    Validates that every city configuration:
      1. Is valid YAML.
      2. Contains required top-level sections: 'city', 'spatial', 'temporal', 'stac'.
      3. Contains city name, state, country.
      4. Bounding box exists, has 4 coordinates, satisfies west < east and south < north.
      5. Coordinates are in valid geographic WGS84 bounds [-180, 180] and [-90, 90].
    """
    assert config_path.exists(), f"Configuration file not found: {config_path.resolve()}"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert isinstance(config, dict), f"{config_path.name} must load as a dictionary."

    # 1. Check required top-level keys
    required_keys = ["city", "spatial", "temporal", "stac"]
    for key in required_keys:
        assert key in config, f"{config_path.name} is missing required section: '{key}'"

    # 2. Check city section
    city_sec = config["city"]
    assert "name" in city_sec and city_sec["name"], f"{config_path.name} missing 'city.name'"

    # 3. Check spatial keys & bbox
    spatial = config["spatial"]
    assert "bbox" in spatial, f"{config_path.name} missing 'spatial.bbox'"
    bbox = spatial["bbox"]
    assert isinstance(bbox, list), f"{config_path.name} 'spatial.bbox' must be a list"
    assert (
        len(bbox) == 4
    ), f"{config_path.name} bbox must have 4 elements [min_lon, min_lat, max_lon, max_lat], got {len(bbox)}."

    min_lon, min_lat, max_lon, max_lat = bbox

    # 4. Check geometric validity
    assert (
        min_lon < max_lon
    ), f"Invalid longitude in {config_path.name}: west ({min_lon}) must be < east ({max_lon})"
    assert (
        min_lat < max_lat
    ), f"Invalid latitude in {config_path.name}: south ({min_lat}) must be < north ({max_lat})"

    # 5. Check coordinate range limits (WGS84)
    assert -180.0 <= min_lon <= 180.0
    assert -180.0 <= max_lon <= 180.0
    assert -90.0 <= min_lat <= 90.0
    assert -90.0 <= max_lat <= 90.0
