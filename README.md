# UrbanPulse 🛰️🏙️

[![UrbanPulse CI](https://github.com/husaintrivedi/UrbanPulse/actions/workflows/ci.yml/badge.svg)](https://github.com/husaintrivedi/UrbanPulse/actions/workflows/ci.yml)
[![UrbanPulse CD](https://github.com/husaintrivedi/UrbanPulse/actions/workflows/cd.yml/badge.svg)](https://github.com/husaintrivedi/UrbanPulse/actions/workflows/cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Cloud-native satellite analytics and spatial machine learning platform monitoring urban sprawl, land cover change trajectories, and radial growth dynamics across global metropolitan areas.**

---

## 🌐 Live Demo & Interactive Dashboard

- **Interactive Web App**: [https://husaintrivedi.github.io/UrbanPulse/](https://husaintrivedi.github.io/UrbanPulse/)
- **API Documentation**: `http://localhost:8000/docs` (when running locally)
- **Grafana Monitoring Dashboard**: `http://localhost:3000` (pre-provisioned with Prometheus metrics)

---

## 🏛️ System Architecture

```mermaid
flowchart TB
    subgraph Earth Observation Data
        S2[AWS Earth Search STAC\nSentinel-2 L2A BOA] --> Stacker[Dask + Stackstac\n60m Spatial Resampling]
        WC[ESA WorldCover 2021\nAWS S3 Global Land Cover] --> Labels[Stratified Spatial Block\nTraining Points]
    end

    subgraph Spatial Analytics Pipeline
        Stacker --> MedComp[Temporal Median Composite\nSCL Cloud & Shadow Masking]
        MedComp --> Indices[Spectral Index Engine\nNDVI, NDBI, MNDWI]
        Indices --> RF[City-Specific Random Forest\n5-Class Classification]
        Labels --> RF
        RF --> Gate{Data Quality Gate\nNoData <5% | Growth <25% | Acc >70%}
        Gate -- PASS --> Change[Transition Matrix Engine\nChange Trajectories 2018-2024]
        Change --> Rings[Concentric Ring Analyzer\n2km Radial Slices 0-22km]
        Rings --> Metrics[Shannon Spatial Entropy\nSprawl Velocity CAGR]
        Metrics --> Exporter[Web Asset Exporter\nEPSG:4326 PNGs, meta.json, stats.json]
    end

    subgraph Storage & Serving Layer
        Exporter --> DB[(PostgreSQL 16 + PostGIS 3.4\nSpatial Indexes & Audit Logs)]
        Exporter --> StaticWeb[Nginx Web Frontend\nMapLibre GL JS + Chart.js]
        DB --> API[FastAPI Backend\nPrometheus Instrumented]
    end

    subgraph Observability
        API --> Prom[Prometheus 2.51]
        Exporter --> PushGW[Prometheus Pushgateway]
        PushGW --> Prom
        Prom --> Grafana[Grafana 10.4\nProvisioned Overview Dashboard]
    end
```

---

## 📊 Multi-City Results & Case Studies

### 1. Ahmedabad (Gujarat, India)
- **Bounding Box**: `[72.3567, 22.8149, 72.8027, 23.2278]` ($\sim 45\times 45\text{ km}$)
- **Classification Accuracy**: **86.2%** Overall Accuracy vs. ESA WorldCover 2021 ($\kappa = 0.814$)
- **Urban Expansion**: Built-up land grew from **$312.4\text{ km}^2$** (2018) to **$384.8\text{ km}^2$** (2024), representing a $+23.2\%$ increase primarily converting agricultural fringe land.
- **Radial Density Gradient**: Core density ($0\text{--}4\text{ km}$) exceeds **$74\%$**, transitioning to $<12\%$ beyond $16\text{ km}$. Shannon spatial entropy increased from $0.941$ to $0.958$, indicating peripheral outward dispersion.

### 2. Pune (Maharashtra, India)
- **Bounding Box**: `[73.6435, 18.3171, 74.0699, 18.7237]` ($\sim 45\times 45\text{ km}$)
- **Classification Accuracy**: **73.9%** Overall Accuracy vs. ESA WorldCover 2021 ($\kappa = 0.656$)
- **Urban Expansion**: Built-up land expanded from **$401.8\text{ km}^2$** (2018) to **$419.7\text{ km}^2$** (2024) ($+17.92\text{ km}^2$ net growth). Sources of new built-up area: $56.9\%$ agricultural conversion, $24.3\%$ open land, and $18.0\%$ vegetation.
- **Concentric Ring Profile**: Inner core ($0\text{--}2\text{ km}$) density is **$70.2\%$**, tapering across the Western Ghats foothills to **$7.5\%$** at the $20\text{--}22\text{ km}$ boundary.

---

## ⚡ One-Command Quickstart

Start the entire application stack (Web frontend, FastAPI backend, and PostGIS database) with Docker Compose:

```bash
# Clone the repository
git clone https://github.com/husaintrivedi/UrbanPulse.git
cd UrbanPulse

# Start all services
make up
```

- 🖥️ **Web Dashboard**: Open [http://localhost:8080/](http://localhost:8080/)
- 📖 **API Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs)
- 📊 **Monitoring Stack** (Prometheus & Grafana):
  ```bash
  docker compose --profile monitoring up -d
  # Grafana: http://localhost:3000 (admin/admin)
  ```

### Run Satellite Processing Pipeline for Any City:
```bash
# Process Pune end-to-end
python -m pipeline.run_city --city pune

# Or via Docker:
make pipeline CITY=pune
```

---

## 📁 Project Structure

```
UrbanPulse/
├── configs/                     # City definitions (bboxes, dry-season dates, classes)
│   ├── ahmedabad.yaml
│   └── pune.yaml
├── pipeline/                    # Core Geospatial & ML Pipeline
│   ├── search_scenes.py         # AWS Earth Search STAC client
│   ├── build_composite.py       # Dask temporal median composite & SCL cloud masking
│   ├── compute_indices.py       # Spectral indices (NDVI, NDBI, MNDWI)
│   ├── get_training_labels.py   # ESA WorldCover S3 raster ingestion
│   ├── sample_points.py         # Stratified spatial block sampling
│   ├── train_classifier.py      # Random Forest model training per city
│   ├── run_all_years.py         # Multi-year annual batch classifier
│   ├── change_detection.py      # Transition matrix & trajectory rasters
│   ├── concentric_rings.py      # 2km radial distance gradient analysis
│   ├── sprawl_metrics.py        # Shannon entropy & sprawl velocity
│   ├── quality_gate.py          # Data quality gate (NoData, volatility, accuracy)
│   ├── metrics_exporter.py      # Prometheus Pushgateway telemetry
│   ├── export_web.py            # Static web bundle generator
│   ├── load_db.py               # PostGIS idempotent loader
│   └── run_city.py              # End-to-end city orchestrator
├── api/                         # FastAPI Geospatial Backend
│   └── main.py                  # Endpoints for overlays, stats, catalog, /metrics
├── web/                         # Single-Page Web Application
│   ├── index.html               # MapLibre GL JS + Chart.js interface
│   └── data/                    # Precomputed static city bundles
├── db/                          # Database Schemas & Migrations
│   └── schema.sql               # PostGIS DDL with GiST spatial indexes
├── infra/                       # Infrastructure as Code & Kubernetes
│   ├── terraform/               # OCI Always Free VM, VCN, and firewall provisioning
│   ├── ansible/                 # OS hardening, Docker, k3s, and Kustomize playbook
│   └── k8s/                     # Kubernetes manifests & Kustomize overlays
├── monitoring/                  # Observability
│   ├── prometheus/              # Prometheus config & 10-day pipeline alert rules
│   └── grafana/                 # Pre-provisioned dashboards & datasources
├── docs/                        # Engineering Documentation
│   ├── deploy.md                # Cloud deployment runbook
│   └── engineering-decisions.md # Architecture trade-offs and decisions
├── tests/                       # Automated Pytest Suite (41 tests)
├── docker-compose.yml           # Multi-service container orchestration
├── Makefile                     # Developer and DevOps command runner
└── pyproject.toml               # Python project dependencies & linters
```

---

## 🛡️ Data Quality Gate

Every pipeline run passes through automated validation rules before results are committed or loaded into PostGIS:
1. **Cloud & Shadow NoData Limit**: Rejects composites with $>5\%$ invalid pixels.
2. **Built-up Growth Sanity Check**: Flags and rejects any annual built-up area variance $>25\%$ between consecutive years.
3. **Model Accuracy Floor**: Fails the run if spatial block validation accuracy drops below $70\%$.

---

## 🏙️ How to Add a New City

UrbanPulse requires **zero city-specific code**. To onboard any global metropolitan area:

1. Create `configs/<city_name>.yaml` with your target coordinates:
   ```yaml
   city:
     name: Hyderabad
     state: Telangana
     country: India
   spatial:
     bbox: [78.2000, 17.2000, 78.6500, 17.6000] # ~45x45 km bounding box
     crs: EPSG:4326
     center_lat: 17.3850
     center_lon: 78.4867
     buffered_distance_km: 8.0
     aoi_size_km: 45.0
   temporal:
     analysis_years:
       start_year: 2018
       end_year: 2024
     dry_season:
       start_month: 11
       end_month: 2
       start_day: 11-01
       end_day: 02-28
   ```
2. Run the pipeline:
   ```bash
   python -m pipeline.run_city --city hyderabad
   ```
3. The city will automatically appear in the web dashboard, API catalog, and PostGIS database.

---

## ⚠️ Limitations

- **Dry-Season Sensitivity**: Spectral classification performs best during winter dry seasons (Nov–Feb) when cloud cover is minimal and crops are distinguishable from perennial vegetation.
- **Resolution Limit**: 60m resampled resolution is optimized for regional sprawl tracking. Micro-scale parcel urban infill (<10m) requires high-resolution commercial imagery.
- **Topographic Shadowing**: In high-relief mountainous cities, steep terrain shadows can occasionally be classified as water without auxiliary DEM hillshade correction.

---

## 📜 License & Contributions

- **License**: Released under the [MIT License](LICENSE).
- **Contributing**: Please review [CONTRIBUTING.md](CONTRIBUTING.md) for style guides, testing procedures, and pull request workflows.
