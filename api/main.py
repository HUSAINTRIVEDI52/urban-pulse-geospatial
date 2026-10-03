"""
UrbanPulse - Geospatial Analytics & Urban Sprawl FastAPI Service
Serves multi-temporal land cover overlays, spatial statistics, change trajectories,
and city metadata from PostGIS spatial database (with static JSON fallback).
"""

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

try:
    from prometheus_fastapi_instrumentator import Instrumentator
except ImportError:
    Instrumentator = None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None
    RealDictCursor = None

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

# Instrument Prometheus metrics (/metrics endpoint)
if Instrumentator is not None:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", tags=["Monitoring"])
else:

    @app.get("/metrics", tags=["Monitoring"], response_class=PlainTextResponse)
    def dummy_metrics():
        return '# HELP http_requests_total Total HTTP Requests\nhttp_requests_total 1\npython_info{version="3.14"} 1\nurbanpulse_api_requests_total 1\n'


def get_db_connection():
    """Returns a psycopg2 database connection if DATABASE_URL is configured and reachable."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        return None
    try:
        conn = psycopg2.connect(db_url, connect_timeout=3)
        return conn
    except Exception:
        return None


@app.get("/health", tags=["System"])
def health_check() -> dict[str, Any]:
    """Health check endpoint confirming API status and database connectivity."""
    db_status = "disabled"
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        conn = get_db_connection()
        if conn:
            db_status = "connected"
            conn.close()
        else:
            db_status = "unreachable"

    return {
        "status": "ok",
        "app": "UrbanPulse API",
        "version": "1.0.0",
        "database": db_status,
    }


@app.get("/cities", tags=["Catalog"])
def list_cities() -> list[dict[str, Any]]:
    """Lists all available cities from PostGIS (falling back to web/data/ catalog)."""
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, state, country, center_lat, center_lon
                    FROM cities
                    ORDER BY name ASC;
                    """)
                db_cities = cur.fetchall()

                # Get available years per city
                cities = []
                for c in db_cities:
                    cur.execute(
                        "SELECT DISTINCT year FROM lulc_stats WHERE city = %s ORDER BY year ASC;",
                        (c["id"],),
                    )
                    years = [r["year"] for r in cur.fetchall()]
                    cities.append(
                        {
                            "id": c["id"],
                            "name": c["name"],
                            "state": c["state"] or "",
                            "country": c["country"] or "",
                            "years": years,
                            "center": [c["center_lat"], c["center_lon"]],
                        }
                    )
                return cities
        except Exception:
            pass
        finally:
            conn.close()

    # Fallback to static catalog
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
    """Retrieves multi-year class areas, sprawl & Shannon entropy metrics, concentric rings, and transitions."""
    city_key = city.lower()

    # Try PostGIS first
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 1. Class areas
                cur.execute(
                    """
                    SELECT year, class, class_name, area_km2, pct
                    FROM lulc_stats
                    WHERE city = %s
                    ORDER BY year ASC, class ASC;
                    """,
                    (city_key,),
                )
                lulc_rows = cur.fetchall()

                # 2. Metrics
                cur.execute(
                    """
                    SELECT year, builtup_km2, growth_pct AS annual_growth_pct, cagr_pct AS cagr_from_start_pct,
                           entropy AS shannon_entropy, core_share_pct, periphery_share_pct
                    FROM metrics
                    WHERE city = %s
                    ORDER BY year ASC;
                    """,
                    (city_key,),
                )
                metric_rows = cur.fetchall()

                # 3. Rings
                cur.execute(
                    """
                    SELECT year, ring_start_km, ring_end_km, builtup_km2, valid_km2, builtup_pct
                    FROM rings
                    WHERE city = %s
                    ORDER BY year ASC, ring_start_km ASC;
                    """,
                    (city_key,),
                )
                ring_rows = cur.fetchall()

                if lulc_rows or metric_rows or ring_rows:
                    # Group class areas by year
                    class_areas_grouped = {}
                    for r in lulc_rows:
                        yr = r["year"]
                        if yr not in class_areas_grouped:
                            class_areas_grouped[yr] = {
                                "year": yr,
                                "built_up_km2": 0.0,
                                "vegetation_km2": 0.0,
                                "water_km2": 0.0,
                                "agriculture_km2": 0.0,
                                "open_land_km2": 0.0,
                                "total_area_km2": 0.0,
                            }
                        cid = r["class"]
                        km2 = float(r["area_km2"])
                        class_areas_grouped[yr]["total_area_km2"] += km2
                        if cid == 1:
                            class_areas_grouped[yr]["built_up_km2"] = km2
                        elif cid == 2:
                            class_areas_grouped[yr]["vegetation_km2"] = km2
                        elif cid == 3:
                            class_areas_grouped[yr]["water_km2"] = km2
                        elif cid == 4:
                            class_areas_grouped[yr]["agriculture_km2"] = km2
                        elif cid == 5:
                            class_areas_grouped[yr]["open_land_km2"] = km2

                    # Group rings by year
                    rings_by_year = {}
                    for r in ring_rows:
                        yr = r["year"]
                        if yr not in rings_by_year:
                            rings_by_year[yr] = []
                        rings_by_year[yr].append(
                            {
                                "ring_start_km": float(r["ring_start_km"]),
                                "ring_end_km": float(r["ring_end_km"]),
                                "builtup_km2": float(r["builtup_km2"]),
                                "valid_km2": float(r["valid_km2"]),
                                "builtup_pct": float(r["builtup_pct"]),
                            }
                        )

                    return {
                        "city": city_key,
                        "class_areas": list(class_areas_grouped.values()),
                        "metrics": [
                            {
                                "year": m["year"],
                                "builtup_km2": float(m["builtup_km2"]),
                                "annual_growth_pct": float(m["annual_growth_pct"] or 0.0),
                                "cagr_pct": float(m["cagr_from_start_pct"] or 0.0),
                                "shannon_entropy": float(m["shannon_entropy"]),
                                "core_share_pct": float(m["core_share_pct"] or 0.0),
                                "periphery_share_pct": float(m["periphery_share_pct"] or 0.0),
                            }
                            for m in metric_rows
                        ],
                        "rings": rings_by_year,
                        "transitions": {},
                    }
        except Exception:
            pass
        finally:
            conn.close()

    # Fallback to static JSON file
    stats_path = WEB_DATA_DIR / city_key / "stats.json"
    if not stats_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Analytics dataset for '{city}' not found in web catalog ({stats_path.name}).",
        )

    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/runs/{city}", tags=["Audit"])
def get_pipeline_runs(city: str) -> list[dict[str, Any]]:
    """Retrieves pipeline execution history and audit logs for a given city from PostGIS."""
    city_key = city.lower()
    conn = get_db_connection()
    if not conn:
        return [
            {
                "id": 1,
                "city": city_key,
                "year": 2024,
                "status": "SUCCESS",
                "started_at": "2026-09-30T00:00:00Z",
                "finished_at": "2026-09-30T00:01:00Z",
                "error": None,
                "note": "Static fallback - PostGIS not connected",
            }
        ]

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, city, year, status, started_at, finished_at, error
                FROM pipeline_runs
                WHERE city = %s
                ORDER BY started_at DESC
                LIMIT 50;
                """,
                (city_key,),
            )
            runs = cur.fetchall()
            return [
                {
                    "id": r["id"],
                    "city": r["city"],
                    "year": r["year"],
                    "status": r["status"],
                    "started_at": r["started_at"].isoformat() if r["started_at"] else None,
                    "finished_at": r["finished_at"].isoformat() if r["finished_at"] else None,
                    "error": r["error"],
                }
                for r in runs
            ]
    finally:
        conn.close()


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
