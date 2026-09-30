# UrbanPulse: Satellite Urban Sprawl & Land Cover Analytics

[![UrbanPulse CI](https://github.com/HUSAINTRIVEDI52/urban-pulse-geospatial/actions/workflows/ci.yml/badge.svg)](https://github.com/HUSAINTRIVEDI52/urban-pulse-geospatial/actions/workflows/ci.yml)
[![GitHub Pages](https://github.com/HUSAINTRIVEDI52/urban-pulse-geospatial/actions/workflows/pages.yml/badge.svg)](https://husaintrivedi52.github.io/urban-pulse-geospatial/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

UrbanPulse is an end-to-end cloud-native geospatial analytics engine and interactive web system for monitoring, quantifying, and visualizing multi-temporal urban sprawl and land cover transitions using Sentinel-2 Level-2A satellite imagery.

---

## 🏗️ Architecture Overview

```
UrbanPulse/
├── configs/                  # City AOI bounding boxes and configuration files
├── pipeline/                 # Satellite compositing, spectral indices, RF ML classifier
├── api/                      # FastAPI REST service delivering overlays and spatial statistics
├── web/                      # MapLibre GL JS + Chart.js zero-dependency single-file web client
├── docker/                   # Dockerfiles for pipeline, api, and web services
├── tests/                    # Pytest test suite with synthetic in-memory fixtures
├── .github/workflows/ci.yml  # GitHub Actions CI (Lint, Test, Docker Build, Trivy Security)
├── docker-compose.yml        # Multi-container orchestration (Web, API, on-demand Pipeline)
└── Makefile                  # Developer automation targets
```

---

## 🚀 Quick Start (Docker & Compose)

### 1. Build and Run All Services
```bash
# Build all images
make build

# Start Web (port 8080) and API (port 8000)
make up
```

- **Interactive Map Client**: [http://localhost:8080](http://localhost:8080)
- **FastAPI OpenAPI Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Probe**: [http://localhost:8000/health](http://localhost:8000/health)

### 2. Run Satellite Processing On-Demand
```bash
# Process a city for a specific year
make pipeline CITY=ahmedabad YEAR=2024
```

---

## 🧪 Testing & CI Workflow

All unit tests use synthetic data fixtures and do not depend on external data downloads or local pipeline outputs.

```bash
# Run test suite
pytest -v

# Run formatters & linters
ruff check .
black --check .
```

The GitHub Actions CI pipeline automatically executes:
1. **Lint**: Code style validation via `ruff` and `black`.
2. **Test**: Isolated test execution on Python 3.11 with pip dependency caching.
3. **Docker**: Multi-image build (`api`, `web`, `pipeline`) with GitHub Actions Layer Caching (`type=gha`).
4. **Scan**: Trivy container vulnerability scanner checking for `CRITICAL` CVEs.

---

## 📜 Land Cover Classification Schema

| Class ID | Land Cover Class | Color | Description |
| :---: | :--- | :---: | :--- |
| **1** | Built-up | `#ef4444` | Residential, commercial, industrial, impervious pavement |
| **2** | Vegetation | `#10b981` | Forests, tree cover, parks, green spaces |
| **3** | Water | `#3b82f6` | Rivers, lakes, reservoirs, canals |
| **4** | Agriculture | `#f59e0b` | Cropland, irrigated fields, seasonal agriculture |
| **5** | Open land | `#8b5cf6` | Bare soil, fallow fields, arid terrain |
