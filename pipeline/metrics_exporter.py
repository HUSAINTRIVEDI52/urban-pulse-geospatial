"""
UrbanPulse - Prometheus Pipeline Metrics Exporter
Records pipeline run execution metrics (duration, status, nodata %, scene counts)
and exports them via Prometheus Pushgateway (if reachable) and local textfile metrics (.prom).
"""

import os
import time
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, write_to_textfile


def export_pipeline_metrics(
    city: str,
    status: str,
    duration_seconds: float,
    nodata_pct: float | None = None,
    scenes_count: int | None = None,
    year: int | None = None,
    pushgateway_url: str | None = None,
    textfile_dir: Path | None = None,
) -> None:
    """
    Exports pipeline execution metrics to Prometheus Pushgateway and/or textfile collector.
    """
    city_key = city.lower()
    registry = CollectorRegistry()

    # Define Gauges
    g_duration = Gauge(
        "pipeline_run_duration_seconds",
        "Duration of the latest satellite analytics pipeline run in seconds",
        ["city"],
        registry=registry,
    )
    g_status = Gauge(
        "pipeline_run_status",
        "Status of the latest pipeline run (1 = SUCCESS, 0 = FAILED)",
        ["city"],
        registry=registry,
    )
    g_last_run = Gauge(
        "pipeline_last_run_timestamp_seconds",
        "Unix timestamp of the most recent pipeline execution",
        ["city"],
        registry=registry,
    )
    g_nodata = Gauge(
        "pipeline_nodata_percentage",
        "Percentage of invalid or missing NoData pixels in satellite composite",
        ["city", "year"],
        registry=registry,
    )
    g_scenes = Gauge(
        "pipeline_scenes_used_total",
        "Number of Sentinel-2/Landsat scenes used in multi-temporal composite",
        ["city", "year"],
        registry=registry,
    )

    # Set metric values
    g_duration.labels(city=city_key).set(duration_seconds)
    g_status.labels(city=city_key).set(1.0 if status.upper() == "SUCCESS" else 0.0)
    g_last_run.labels(city=city_key).set(time.time())

    yr_label = str(year) if year else "all"
    if nodata_pct is not None:
        g_nodata.labels(city=city_key, year=yr_label).set(nodata_pct)
    if scenes_count is not None:
        g_scenes.labels(city=city_key, year=yr_label).set(scenes_count)

    # 1. Export to local textfile (.prom) for node-exporter / prometheus scrapers
    if textfile_dir is None:
        textfile_dir = Path(__file__).resolve().parent.parent / "data"

    try:
        textfile_dir.mkdir(parents=True, exist_ok=True)
        prom_file = textfile_dir / f"{city_key}_metrics.prom"
        write_to_textfile(str(prom_file), registry)
    except Exception as e:
        print(f"[!] Warning: Could not write metrics textfile: {e}")

    # 2. Push to Prometheus Pushgateway if configured
    if pushgateway_url is None:
        pushgateway_url = os.getenv("PUSHGATEWAY_URL")

    if pushgateway_url:
        try:
            push_to_gateway(
                gateway=pushgateway_url,
                job="urbanpulse_pipeline",
                registry=registry,
                grouping_key={"city": city_key},
                timeout=5,
            )
            print(f"[+] Pushed pipeline metrics to Pushgateway: {pushgateway_url}")
        except Exception as e:
            print(
                f"[*] Note: Pushgateway at {pushgateway_url} not reachable ({e}). Textfile metrics preserved."
            )
