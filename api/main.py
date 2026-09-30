"""
UrbanPulse - Geospatial Analytics & Urban Sprawl FastAPI Service
Serves multi-temporal land cover overlays, spatial statistics, change trajectories,
and city metadata from web/data/ for interactive web applications.
"""

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DATA_DIR = PROJECT_ROOT / "web" / "data"

app = FastAPI(
    title="UrbanPulse Geospatial API",
    description="High-performance backend API serving satellite-derived urban land cover overlays, change analytics, and sprawl metrics.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    """Health check endpoint confirming API status."""
    return {"status": "ok", "app": "UrbanPulse API", "version": "1.0.0"}


@app.get("/cities", tags=["Catalog"])
def list_cities() -> list[dict[str, Any]]:
    """Lists all available cities with processed web data in the catalog."""
    if not WEB_DATA_DIR.exists():
        return []

    cities = []
    for city_dir in sorted(WEB_DATA_DIR.iterdir()):
        if city_dir.is_dir():
            meta_path = city_dir / "meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    cities.append(
                        {
                            "id": city_dir.name.lower(),
                            "name": meta.get("city", city_dir.name.capitalize()),
                            "state": meta.get("state", ""),
                            "country": meta.get("country", ""),
                            "years": meta.get("years", []),
                            "center": meta.get("center", []),
                            "bounds": meta.get("bounds", []),
                        }
                    )
                except Exception:
                    cities.append({"id": city_dir.name.lower(), "name": city_dir.name.capitalize()})
            else:
                cities.append({"id": city_dir.name.lower(), "name": city_dir.name.capitalize()})

    return cities


@app.get("/meta/{city}", tags=["Metadata"])
def get_city_meta(city: str) -> dict[str, Any]:
    """Retrieves spatial bounds, center coordinates, class colors, and available years for a city."""
    city_key = city.lower()
    meta_path = WEB_DATA_DIR / city_key / "meta.json"

    if not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"City metadata for '{city}' not found in web catalog ({meta_path.name}).",
        )

    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/stats/{city}", tags=["Analytics"])
def get_city_stats(city: str) -> dict[str, Any]:
    """Retrieves multi-year class areas, sprawl & Shannon entropy metrics, concentric ring data, and transition matrices."""
    city_key = city.lower()
    stats_path = WEB_DATA_DIR / city_key / "stats.json"

    if not stats_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Analytics dataset for '{city}' not found in web catalog ({stats_path.name}).",
        )

    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/overlay/{city}/{year}", tags=["Overlays"])
def get_annual_overlay(city: str, year: int) -> FileResponse:
    """Returns the EPSG:4326 transparent PNG classification overlay for the specified city and year."""
    city_key = city.lower()
    png_path = WEB_DATA_DIR / city_key / f"{year}.png"

    if not png_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Classification overlay for '{city}' in year {year} not found ({png_path.name}).",
        )

    return FileResponse(
        png_path,
        media_type="image/png",
        filename=f"{city_key}_{year}_overlay.png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/change/{city}/{start}/{end}", tags=["Overlays"])
def get_change_overlay(city: str, start: int, end: int) -> FileResponse:
    """Returns the EPSG:4326 transparent PNG change trajectory overlay between start and end year."""
    city_key = city.lower()
    png_path = WEB_DATA_DIR / city_key / f"change_{start}_{end}.png"

    if not png_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Change overlay for '{city}' ({start} -> {end}) not found ({png_path.name}).",
        )

    return FileResponse(
        png_path,
        media_type="image/png",
        filename=f"{city_key}_change_{start}_{end}.png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
