# Architecture & Engineering Decisions

This document details the core engineering and design trade-offs made in the **UrbanPulse** project.

---

## 1. Static PNG Overlays vs. Dynamic Tile Servers (TiTiler / GeoServer)

### Decision:
Precompute transparent, reprojected `EPSG:4326` PNG overlays (`<year>.png` and `change_<start>_<end>.png`) served directly as static assets from Nginx / GitHub Pages CDN alongside Cloud-Optimized GeoTIFFs (COGs) for analytical backend queries.

### Rationale:
- **Cost & Zero-Backend Reliability**: Dynamic raster tile servers (such as TiTiler, MapServer, or GeoServer) require continuous CPU/GPU resources to resample, color-map, and compress raster tiles on every pan/zoom interaction. Static PNG overlays allow the frontend to run entirely as static files on GitHub Pages or edge CDNs with zero backend cost.
- **Client Latency**: Transparent PNG overlays for a 45×45 km metropolitan AOI are approximately 100–140 KB in size. MapLibre GL JS renders them instantly in WebGL as an `ImageSource` bounding box with smooth opacity blending and hardware-accelerated swipe transitions.
- **Scalability**: Overlays can be cached indefinitely on CDNs (`Cache-Control: public, max-age=86400`) and served to thousands of simultaneous users without generating cloud compute spikes.

---

## 2. PostGIS as the Spatial Source of Truth

### Decision:
Store all spatial metadata (bounding box geometries, annual LULC class statistics, transition matrices, concentric ring densities, and Shannon entropy metrics) in a **PostgreSQL 16 + PostGIS 3.4** spatial database with fallback to static JSON files.

### Rationale:
- **Spatial Geometry Operations**: Storing city AOI bounding envelopes as `geometry(Polygon, 4326)` with GiST spatial indexes allows fast spatial intersections, proximity queries, and integration with GIS tools (QGIS, ArcGIS).
- **Relational Integrity**: Foreign key constraints and composite primary keys `(city, year, class)` guarantee idempotent re-runs of the processing pipeline without duplicating historical records.
- **Rich Analytics**: PostGIS enables SQL-level spatial aggregations, time-series joins, and rapid audit logging via the `pipeline_runs` table.

---

## 3. Lightweight k3s over Managed Kubernetes (EKS / GKE / AKS)

### Decision:
Use **k3s** packaged in a lightweight containerized distribution with built-in Traefik Ingress and local storage provisioner, rather than managed cloud Kubernetes offerings.

### Rationale:
- **Cost Efficiency on Free-Tier Cloud**: Managed Kubernetes control planes (EKS, GKE) cost ~$70+/month for the control plane alone. k3s runs comfortably inside a single Oracle Cloud Always Free Ampere A1 instance (4 OCPUs, 24 GB RAM).
- **Developer Parity (k3d)**: Developers can run the exact same Kubernetes manifests locally using `k3d` (k3s in Docker) in seconds without differences in Ingress controllers or storage classes.
- **Low Footprint**: k3s replaces heavyweight Kubernetes components with sqlite/embedded etcd, reducing idle memory overhead to under 512 MB.

---

## 4. City-Specific ML Models vs. Generalized Global Classifiers

### Decision:
Train independent Random Forest classifiers for each city (`data/{city}_rf_model_{year}.pkl`) using that city's local ESA WorldCover 2021 ground truth, rather than applying a single universal classifier.

### Rationale:
- **Spectral Variations Across Biomes**: Soil spectral reflectance, vegetation phenology, and building materials vary significantly between geographic zones (e.g., arid alluvial soils in Ahmedabad vs. basaltic Deccan trap terrain and hill topography in Pune).
- **Reduced Spectral Confusion**: Local models avoid confusion between dry fallow agricultural fields and bare urban soil by learning city-specific dry-season spectral index distributions (`NDVI`, `NDBI`, `MNDWI`).
- **Targeted Accuracy**: Local training achieves high overall accuracy (>73–85%) and robust Cohen's Kappa without requiring massive multi-gigabyte training sets.

---

## 5. Sentinel-2 L2A Scope (2017–Present)

### Decision:
Scope the primary automated pipeline to Sentinel-2 Level-2A surface reflectance (Bottom-of-Atmosphere / BOA) from 2017/2018 to 2024.

### Rationale:
- **Consistent Atmospheric Correction**: Sentinel-2 L2A provides harmonized 10m–20m multispectral bands with standardized Scene Classification Layer (SCL) cloud/shadow masking on AWS Earth Search STAC.
- **Avoiding Cross-Sensor Radiometric Gaps**: Merging older Landsat 5/7 data (30m) with Sentinel-2 (10m) introduces spatial resolution artifacts and sensor calibration biases in automated change detection.

---

## 6. What We Would Change at Continental Scale (100+ Cities)

If scaling UrbanPulse to monitor hundreds or thousands of global cities continuously:

1. **Distributed Raster Engine**: Replace local single-node Dask execution with **Apache Sedona** or **Ray on KubeRay** running on a scalable cluster for parallel STAC ingestion.
2. **Dynamic Vector Tile Pipeline**: Pre-generate Mapbox Vector Tiles (MVT) for street-level urban sprawl boundaries and building footprints using Martin or pg_tileserv.
3. **Deep Learning Vision Models**: Transition from pixel-level Random Forest to a lightweight geospatial vision transformer (e.g., Prithvi / SatMAE) running on GPU inference workers for improved semantic feature extraction.
4. **Serverless STAC Compositing**: Deploy serverless AWS Lambda workers running `cog-mosaic` to create temporal median composites on-demand in cloud object storage.
