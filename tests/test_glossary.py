import json
import re
from pathlib import Path

MANDATORY_TERMS = [
    "built-up",
    "land cover",
    "Sentinel-2",
    "composite",
    "dry season",
    "NDVI",
    "NDBI",
    "MNDWI",
    "Random Forest",
    "classification",
    "WorldCover",
    "normalisation (TLS)",
    "stratified sample",
    "Olofsson estimator",
    "confidence interval",
    "F1 score",
    "gross gain/loss",
    "net change",
    "Shannon entropy",
    "core vs periphery",
    "distance ring",
    "quality gate",
    "partial season",
    "false positive",
    "fallow land",
    "60 m pixel",
    "km2",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
GLOSSARY_PATH = REPO_ROOT / "web" / "data" / "glossary.json"


def test_glossary_json_exists_and_valid():
    assert GLOSSARY_PATH.exists(), f"Missing {GLOSSARY_PATH}"
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert "terms" in data
    assert len(data["terms"]) >= len(MANDATORY_TERMS)


def test_glossary_contains_all_mandatory_terms():
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        data = json.load(f)

    existing_terms = set()
    for item in data["terms"]:
        existing_terms.add(item["term"].lower())
        existing_terms.add(item["id"].lower())
        for a in item.get("aliases", []):
            existing_terms.add(a.lower())

    for req in MANDATORY_TERMS:
        assert (
            req.lower() in existing_terms
        ), f"Mandatory term '{req}' not found in glossary terms or aliases"


def test_ui_terms_are_defined_in_glossary():
    """Any term referenced via data-term='...' in the web UI must exist in glossary.json."""
    with open(GLOSSARY_PATH, encoding="utf-8") as f:
        glossary_data = json.load(f)

    glossary_ids = {t["id"] for t in glossary_data["terms"]}

    # Scan index.html and js files
    web_dir = REPO_ROOT / "web"
    data_term_regex = re.compile(r'data-term=["\']([^"\']+)["\']')

    found_terms = set()
    for file_path in web_dir.glob("**/*"):
        if file_path.suffix in [".html", ".js", ".json"]:
            # Skip the glossary itself
            if file_path.name == "glossary.json":
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = data_term_regex.findall(content)
                for m in matches:
                    found_terms.add(m)
            except Exception:
                pass

    missing = found_terms - glossary_ids
    assert not missing, f"Terms used in UI missing from glossary: {missing}"
