# Architecture & Engineering Decisions

This document records the architectural decisions, alternatives evaluated, and trade-offs made in the **UrbanPulse** project.

---

## 1. Static PNG Overlays vs. Dynamic Tile Servers

- **Decision**: Precompute transparent `EPSG:4326` PNG overlays (`<year>.png` and `change_2018_2024.png`) served as static CDN assets alongside Cloud-Optimized GeoTIFFs (COGs) for backend analytical extraction.
- **Alternatives Considered**: Dynamic raster tile servers (TiTiler, GeoServer, MapServer, pg_tileserv).
- **Trade-off**: Dynamic tile servers provide on-the-fly multi-scale band arithmetic and custom colormaps but require continuous CPU/GPU instances and incur high cloud compute costs. Static PNG overlays (~100–160 KB per city) render instantaneously via WebGL in MapLibre GL JS, scale to thousands of users for $0 hosting cost on GitHub Pages/CDN, but require re-exporting if colormaps or bounding boxes change.

---

## 2. PostGIS as the Spatial Source of Truth

- **Decision**: Persist all spatial envelopes, annual land cover class statistics, transition matrices, concentric ring densities, Shannon entropy metrics, and `pipeline_runs` audit logs in **PostgreSQL 16 + PostGIS 3.4**.
- **Alternatives Considered**: Pure flat file storage (GeoJSON / Parquet on S3) or Document databases (MongoDB, DynamoDB).
- **Trade-off**: Flat files avoid running a database daemon but lack transactional ACID guarantees, relational integrity, and fast spatial intersection indexes (`GiST`). PostGIS enables fast bounding envelope queries, time-series aggregations, and idempotent pipeline re-runs via composite primary keys `(city, year, class)`.

---

## 3. Lightweight k3s over Managed Kubernetes

- **Decision**: Deploy infrastructure on a single-node **k3s** cluster orchestrated with Helm and Kustomize overlays.
- **Alternatives Considered**: Managed Kubernetes services (AWS EKS, GCP GKE, Azure AKS) or plain Docker Compose.
- **Trade-off**: Managed cloud Kubernetes costs $70+/month for control planes alone. k3s runs with embedded SQLite/etcd consuming <512 MB memory on Oracle Cloud Always Free Ampere A1 (4 OCPU, 24 GB RAM) while providing full Kubernetes API compatibility, Traefik Ingress, and seamless local reproduction via `k3d`.

---

## 4. Per-City Models and Pooled Multi-Year Training

- **Decision**: Train city-specific Random Forest classifiers pooled across three representative dry-season years (2018, 2021, 2024) at identical sample point locations with strict spatial-block partitioning.
- **Alternatives Considered**: A single global/universal classifier, or independent year-by-year models without pooling.
- **Trade-off**: Universal classifiers suffer from severe regional soil/phenology domain shift (e.g., basaltic Deccan traps in Pune vs. alluvial soils in Ahmedabad). Single-year models suffer from inter-annual decision boundary flicker. The pooled per-city model stabilizes class definitions across the 7-year timeline while respecting local biome spectral distributions.

---

## 5. Sentinel-2-Only Scope (2017–Present)

- **Decision**: Scope the automated production pipeline to Sentinel-2 Level-2A surface reflectance (MSI BOA) from 2017/2018 to 2024.
- **Alternatives Considered**: Combining historical Landsat 5/7/8 (1984–present) with Sentinel-2.
- **Trade-off**: Including Landsat extends time series to 40 years but introduces significant cross-sensor resolution mismatches (30m vs. 10m/20m), band wavelength shifts, and Landsat 7 Scan Line Corrector (SLC-off) data gaps. Sentinel-2 provides consistent 10–20m multispectral bands with standardized SCL cloud masking for clean, high-precision urban boundary tracking.

---

## 6. Temporal Cleanup Rules (Majority & Persistence)

- **Decision**: Apply post-classification temporal rules (2-of-3 year majority filter and 2-consecutive-year urban persistence) on predicted rasters to eliminate spurious inter-annual classification flips.
- **Alternatives Considered**: Raw pixel predictions without temporal smoothing, or HMM / CRFs temporal graph modeling.
- **Trade-off**: Raw satellite classifications contain seasonal phenological noise where fallow dry agricultural fields transiently flip to built-up land and back. The persistence rule codifies the physical reality that urbanized land rarely reverts to natural cover, eliminating false de-urbanization speckle at the cost of assuming irreversible urban development.

---

## 7. What I Would Change at Scale (100+ Cities)

- **Decision**: Current architecture processes cities sequentially via Dask on local/VM cores.
- **Alternatives Considered / Scale Architecture**:
  1. **Distributed Compute**: Replace single-node processing with **Ray on Kubernetes (KubeRay)** or **Apache Sedona** for massively parallel STAC fetching and spatial indexing.
  2. **Vector Tile Serving**: Precompute Mapbox Vector Tiles (MVT) for parcel-level sprawl vectors using Martin or Tippecanoe.
  3. **Deep Learning Foundation Models**: Transition from pixel-level Random Forest to geospatial vision transformers (e.g., Prithvi / SatMAE) running on GPU inference pools.
  4. **Serverless Compositing**: Deploy event-driven AWS Lambda workers running `cog-mosaic` to synthesize cloud-free composites on object storage directly.
