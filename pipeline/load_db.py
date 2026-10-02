"""
UrbanPulse - PostGIS Spatial Database Loader
Reads time-series satellite analytics CSV outputs and configurations,
executing idempotent upserts into PostGIS tables and logging pipeline execution history.
"""

import argparse
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    import psycopg2
    from psycopg2.extras import execute_values
except ImportError:
    psycopg2 = None
    execute_values = None


def _execute_values(cur, sql, rows):
    """Executes bulk inserts using execute_values or falls back to cursor execute."""
    if execute_values is not None:
        try:
            execute_values(cur, sql, rows)
            return
        except Exception:
            pass
    # Fallback for mocked test connections
    if hasattr(cur, "execute_values"):
        cur.execute_values(sql, rows)
    else:
        for r in rows:
            cur.execute(sql, r)

# Class mapping standard
PROJECT_CLASSES = {
    1: "Built-up",
    2: "Vegetation",
    3: "Water",
    4: "Agriculture",
    5: "Open land",
}

CLASS_NAME_TO_ID = {name.lower(): cid for cid, name in PROJECT_CLASSES.items()}


def get_db_connection(db_url: str):
    """Establishes and returns a psycopg2 connection."""
    if psycopg2 is None:
        raise ImportError(
            "psycopg2 is not installed in the local Python environment. "
            "To load into PostGIS, run within Docker or install psycopg2-binary: `pip install psycopg2-binary`"
        )
    return psycopg2.connect(db_url)


def init_db_schema(conn, schema_path: Path | None = None) -> None:
    """Initializes PostGIS extension and database tables if not already existing."""
    if schema_path is None:
        schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"

    if not schema_path.exists():
        return

    with open(schema_path, encoding="utf-8") as f:
        schema_sql = f.read()

    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.commit()


def load_city_metadata(conn, city: str, config_dir: Path) -> dict[str, Any]:
    """Loads city YAML configuration and upserts into cities table with Polygon geometry."""
    cfg_file = config_dir / f"{city.lower()}.yaml"
    if not cfg_file.exists():
        # Fallback minimal
        city_id = city.lower()
        city_name = city.capitalize()
        lat, lon = 23.0225, 72.5714
        bbox = [72.356712, 22.814991, 72.802746, 23.22784]
        state, country = "Gujarat", "India"
    else:
        with open(cfg_file, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        city_info = cfg.get("city", {})
        city_id = city_info.get("name", city).lower().replace(" ", "_")
        city_name = city_info.get("name", city.capitalize())
        state = city_info.get("state", "")
        country = city_info.get("country", "")
        lat = city_info.get("center", {}).get("lat", 0.0)
        lon = city_info.get("center", {}).get("lon", 0.0)
        bbox = cfg.get("spatial", {}).get("bbox", [0.0, 0.0, 0.0, 0.0])

    min_lon, min_lat, max_lon, max_lat = bbox

    upsert_city_sql = """
    INSERT INTO cities (id, name, state, country, center_lat, center_lon, geom, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, ST_SetSRID(ST_MakeEnvelope(%s, %s, %s, %s), 4326), NOW())
    ON CONFLICT (id) DO UPDATE SET
        name = EXCLUDED.name,
        state = EXCLUDED.state,
        country = EXCLUDED.country,
        center_lat = EXCLUDED.center_lat,
        center_lon = EXCLUDED.center_lon,
        geom = EXCLUDED.geom,
        updated_at = NOW();
    """

    with conn.cursor() as cur:
        cur.execute(
            upsert_city_sql,
            (
                city_id,
                city_name,
                state,
                country,
                lat,
                lon,
                min_lon,
                min_lat,
                max_lon,
                max_lat,
            ),
        )
    conn.commit()

    return {"id": city_id, "name": city_name, "bbox": bbox}


def resolve_data_csv(city_id: str, suffix: str, data_dir: Path) -> Path:
    """Finds CSV in data_dir, data_dir/city_id, etc."""
    candidates = [
        data_dir / f"{city_id}_{suffix}.csv",
        data_dir / city_id / f"{suffix}.csv",
        data_dir / city_id / f"{city_id}_{suffix}.csv",
        data_dir / f"{suffix}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def load_lulc_stats(conn, city_id: str, data_dir: Path) -> int:
    """Upserts annual land cover area statistics into lulc_stats."""
    csv_file = resolve_data_csv(city_id, "class_areas", data_dir)
    if not csv_file.exists():
        return 0

    df = pd.read_csv(csv_file)
    rows = []

    # Map column names
    col_map = {}
    for col in df.columns:
        clean = col.strip().lower()
        if "built" in clean:
            col_map[1] = col
        elif "veg" in clean:
            col_map[2] = col
        elif "water" in clean:
            col_map[3] = col
        elif "agri" in clean:
            col_map[4] = col
        elif "open" in clean:
            col_map[5] = col

    for _, r in df.iterrows():
        year = int(r["Year"])
        total_area = float(r.get("Total_Area_km2", r.get("total_area_km2", 0.0)))
        if total_area <= 0:
            total_area = sum(float(r[col_map[cid]]) for cid in col_map if cid in col_map)

        for cid, col_name in col_map.items():
            if col_name in r:
                area_km2 = float(r[col_name])
                pct = (area_km2 / total_area * 100.0) if total_area > 0 else 0.0
                class_name = PROJECT_CLASSES.get(cid, f"Class {cid}")
                rows.append((city_id, year, cid, class_name, area_km2, pct))

    if not rows:
        return 0

    upsert_sql = """
    INSERT INTO lulc_stats (city, year, class, class_name, area_km2, pct)
    VALUES %s
    ON CONFLICT (city, year, class) DO UPDATE SET
        class_name = EXCLUDED.class_name,
        area_km2 = EXCLUDED.area_km2,
        pct = EXCLUDED.pct;
    """

    with conn.cursor() as cur:
        _execute_values(cur, upsert_sql, rows)
    conn.commit()
    return len(rows)


def load_rings(conn, city_id: str, data_dir: Path) -> int:
    """Upserts concentric distance ring gradient analytics into rings table."""
    csv_file = resolve_data_csv(city_id, "rings", data_dir)
    if not csv_file.exists():
        return 0

    df = pd.read_csv(csv_file)
    rows = []

    for _, r in df.iterrows():
        year = int(r["year"])
        r_start = float(r["ring_start_km"])
        r_end = float(r["ring_end_km"])
        built_km2 = float(r["builtup_km2"])
        valid_km2 = float(r["valid_km2"])
        pct = float(r["builtup_pct"])
        rows.append((city_id, year, r_start, r_end, built_km2, valid_km2, pct))

    if not rows:
        return 0

    upsert_sql = """
    INSERT INTO rings (city, year, ring_start_km, ring_end_km, builtup_km2, valid_km2, builtup_pct)
    VALUES %s
    ON CONFLICT (city, year, ring_start_km, ring_end_km) DO UPDATE SET
        builtup_km2 = EXCLUDED.builtup_km2,
        valid_km2 = EXCLUDED.valid_km2,
        builtup_pct = EXCLUDED.builtup_pct;
    """

    with conn.cursor() as cur:
        _execute_values(cur, upsert_sql, rows)
    conn.commit()
    return len(rows)


def load_metrics(conn, city_id: str, data_dir: Path) -> int:
    """Upserts multi-year sprawl velocity and Shannon entropy metrics."""
    csv_file = resolve_data_csv(city_id, "metrics", data_dir)
    if not csv_file.exists():
        return 0

    df = pd.read_csv(csv_file)
    rows = []

    for _, r in df.iterrows():
        year = int(r["year"])
        builtup_km2 = float(r["builtup_km2"])
        growth_pct = (
            float(r.get("annual_growth_pct", r.get("growth_pct", 0.0)))
            if not pd.isna(r.get("annual_growth_pct", r.get("growth_pct")))
            else None
        )
        cagr_pct = (
            float(r.get("cagr_from_start_pct", r.get("cagr_pct", 0.0)))
            if not pd.isna(r.get("cagr_from_start_pct", r.get("cagr_pct")))
            else None
        )
        entropy = float(r.get("shannon_entropy", r.get("entropy", 0.0)))
        core_pct = (
            float(r.get("core_share_0_6km_pct", r.get("core_share_pct", 0.0)))
            if not pd.isna(r.get("core_share_0_6km_pct", r.get("core_share_pct")))
            else None
        )
        periph_pct = (
            float(r.get("periphery_share_gt_12km_pct", r.get("periphery_share_pct", 0.0)))
            if not pd.isna(r.get("periphery_share_gt_12km_pct", r.get("periphery_share_pct")))
            else None
        )

        rows.append(
            (
                city_id,
                year,
                builtup_km2,
                growth_pct,
                cagr_pct,
                entropy,
                core_pct,
                periph_pct,
            )
        )

    if not rows:
        return 0

    upsert_sql = """
    INSERT INTO metrics (city, year, builtup_km2, growth_pct, cagr_pct, entropy, core_share_pct, periphery_share_pct)
    VALUES %s
    ON CONFLICT (city, year) DO UPDATE SET
        builtup_km2 = EXCLUDED.builtup_km2,
        growth_pct = EXCLUDED.growth_pct,
        cagr_pct = EXCLUDED.cagr_pct,
        entropy = EXCLUDED.entropy,
        core_share_pct = EXCLUDED.core_share_pct,
        periphery_share_pct = EXCLUDED.periphery_share_pct;
    """

    with conn.cursor() as cur:
        _execute_values(cur, upsert_sql, rows)
    conn.commit()
    return len(rows)


def load_transitions(conn, city_id: str, data_dir: Path) -> int:
    """Upserts all land cover transition matrices found in data_dir or data_dir/city_id."""
    pattern = re.compile(rf"^(?:{re.escape(city_id)}_)?transition_(\d{{4}})_(\d{{4}})\.csv$")
    total_rows = 0

    search_dirs = [data_dir / city_id, data_dir]
    seen_files = set()

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for file_path in s_dir.glob("*transition_*.csv"):
            if file_path.name in seen_files:
                continue
            seen_files.add(file_path.name)
            m = pattern.match(file_path.name)
            if not m:
                continue

            start_yr = int(m.group(1))
            end_yr = int(m.group(2))

            df = pd.read_csv(file_path, index_col=0)
            rows = []

            for row_label, row in df.iterrows():
                row_str = str(row_label).lower()
                # Determine from_class
                from_cid = None
                for name, cid in CLASS_NAME_TO_ID.items():
                    if name in row_str:
                        from_cid = cid
                        break

                if from_cid is None:
                    continue

                for col_label, val in row.items():
                    col_str = str(col_label).lower()
                    to_cid = None
                    for name, cid in CLASS_NAME_TO_ID.items():
                        if name in col_str:
                            to_cid = cid
                            break

                    if to_cid is not None:
                        area_km2 = float(val)
                        rows.append((city_id, start_yr, end_yr, from_cid, to_cid, area_km2))

            if rows:
                upsert_sql = """
                INSERT INTO transitions (city, start_year, end_year, from_class, to_class, area_km2)
                VALUES %s
                ON CONFLICT (city, start_year, end_year, from_class, to_class) DO UPDATE SET
                    area_km2 = EXCLUDED.area_km2;
                """
                with conn.cursor() as cur:
                    _execute_values(cur, upsert_sql, rows)
                conn.commit()
                total_rows += len(rows)

    return total_rows


def log_pipeline_run(
    conn,
    city_id: str,
    year: int,
    status: str,
    started_at: datetime,
    finished_at: datetime | None = None,
    error: str | None = None,
) -> int:
    """Records pipeline run execution in the pipeline_runs table."""
    insert_sql = """
    INSERT INTO pipeline_runs (city, year, status, started_at, finished_at, error)
    VALUES (%s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(insert_sql, (city_id, year, status, started_at, finished_at, error))
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def load_city_data_to_db(
    city: str = "ahmedabad",
    db_url: str | None = None,
    data_dir: Path | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Main entry point for loading city datasets into PostGIS.
    """
    if db_url is None:
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/urbanpulse",
        )

    project_root = Path(__file__).resolve().parent.parent
    if data_dir is None:
        data_dir = project_root / "data"
    if config_dir is None:
        config_dir = project_root / "configs"

    started_at = datetime.now(UTC)
    city_key = city.lower()

    conn = get_db_connection(db_url)
    try:
        # Initialize tables
        init_db_schema(conn)

        # 1. City Metadata
        city_meta = load_city_metadata(conn, city=city_key, config_dir=config_dir)
        city_id = city_meta["id"]

        # 2. LULC Stats
        lulc_count = load_lulc_stats(conn, city_id=city_id, data_dir=data_dir)

        # 3. Concentric Rings
        rings_count = load_rings(conn, city_id=city_id, data_dir=data_dir)

        # 4. Sprawl Metrics
        metrics_count = load_metrics(conn, city_id=city_id, data_dir=data_dir)

        # 5. Transitions
        trans_count = load_transitions(conn, city_id=city_id, data_dir=data_dir)

        finished_at = datetime.now(UTC)

        # 6. Audit run log
        run_id = log_pipeline_run(
            conn,
            city_id=city_id,
            year=2024,
            status="SUCCESS",
            started_at=started_at,
            finished_at=finished_at,
            error=None,
        )

        return {
            "status": "success",
            "city": city_id,
            "run_id": run_id,
            "lulc_stats_rows": lulc_count,
            "rings_rows": rings_count,
            "metrics_rows": metrics_count,
            "transitions_rows": trans_count,
            "elapsed_s": (finished_at - started_at).total_seconds(),
        }

    except Exception as e:
        finished_at = datetime.now(UTC)
        try:
            log_pipeline_run(
                conn,
                city_id=city_key,
                year=2024,
                status="FAILED",
                started_at=started_at,
                finished_at=finished_at,
                error=str(e),
            )
        except Exception:
            pass
        raise e
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="UrbanPulse PostGIS Ingestion Loader",
    )
    parser.add_argument(
        "--city",
        type=str,
        default="ahmedabad",
        help="Target city key (default: ahmedabad)",
    )
    parser.add_argument(
        "--db-url",
        type=str,
        default=None,
        help="PostGIS connection string (default: DATABASE_URL env)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to data/ directory containing CSVs",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Path to configs/ directory containing city YAML",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw uncleaned datasets instead of clean/",
    )

    args = parser.parse_args()

    print(f"[*] Ingesting {args.city.upper()} data into PostGIS database...")
    res = load_city_data_to_db(
        city=args.city,
        db_url=args.db_url,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
    )

    print("\n" + "=" * 60)
    print(f"[+] PostGIS Ingestion Complete for {args.city.capitalize()}:")
    print("    - City Record       : 1 (with ST_MakeEnvelope geometry)")
    print(f"    - LULC Stats Rows   : {res['lulc_stats_rows']}")
    print(f"    - Ring Profile Rows : {res['rings_rows']}")
    print(f"    - Sprawl Metric Rows: {res['metrics_rows']}")
    print(f"    - Transition Rows   : {res['transitions_rows']}")
    print(f"    - Pipeline Run ID   : #{res['run_id']} (SUCCESS)")
    print(f"    - Elapsed Time      : {res['elapsed_s']:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
