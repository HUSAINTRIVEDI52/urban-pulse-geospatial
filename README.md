# UrbanPulse 🛰️🏙️

[![CI](https://img.shields.io/github/actions/workflow/status/HUSAINTRIVEDI52/urban-pulse-geospatial/ci.yml?branch=main&label=CI&logo=github&style=flat-square)](https://github.com/HUSAINTRIVEDI52/urban-pulse-geospatial/actions/workflows/ci.yml)
[![CD](https://img.shields.io/github/actions/workflow/status/HUSAINTRIVEDI52/urban-pulse-geospatial/cd.yml?branch=main&label=CD&logo=github&style=flat-square)](https://github.com/HUSAINTRIVEDI52/urban-pulse-geospatial/actions/workflows/cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Shows how Ahmedabad and Pune grew from 2020 to 2024, measured from free Sentinel-2 satellite images. What the satellite data shows, and how sure we are.**

---

## 🌐 Live Demo & Interactive App

- **Interactive Web Application**: [https://HUSAINTRIVEDI52.github.io/urban-pulse-geospatial/](https://HUSAINTRIVEDI52.github.io/urban-pulse-geospatial/)
- **FastAPI Documentation**: `http://localhost:8000/docs` (local deployment)
- **Grafana Monitoring**: `http://localhost:3000` (provisioned with Prometheus metrics)

---

## 📸 Interface & Spatial Visualizations

|           Interactive Web Map & Multi-Series Growth           |                   Classified Land Cover Output                   |
| :-----------------------------------------------------------: | :--------------------------------------------------------------: |
| ![UrbanPulse Web Interface](data/preview_2024_classified.png) | ![Land Cover Map](data/ahmedabad/ahmedabad_change_2018_2024.png) |

---

## 📊 Key Results (2020–2024 Analysis Window)

> **Framing Note**: All core analytics are evaluated strictly over the **2020–2024** window. 2018-2019 excluded: too few clear scenes in the Nov-Feb window. 2022 covers October-December 2021 only (no January-February) (query window Oct 1, 2021 – Mar 31, 2022; last acquisition date Dec 23, 2021 for Ahmedabad; query window Nov 1, 2021 – Feb 28, 2022; last acquisition date Dec 25, 2021 for Pune; scale and offset are derived directly from STAC item metadata).
>
> **Real Composite Observation Windows & Scene Counts**:
>
> - **Ahmedabad**:
>   - **2020**: Dec 1, 2019 – Feb 15, 2020 (9 dates, 14 scenes, 0 outside Nov–Feb).
>   - **2021**: Oct 1, 2020 – Mar 31, 2021 (12 dates [9 in-window], 20-scene cap, 5 scenes outside Nov–Feb on Oct 29, Mar 23, Mar 28).
>   - **2022**: Oct 1, 2021 – Mar 31, 2022 (query window; last scene Dec 23, 2021; 9 dates [6 in-window], 18 scenes, 6 scenes outside Nov–Feb on Oct 14, Oct 19, Oct 29).
>   - **2023**: Oct 1, 2022 – Mar 31, 2023 (12 dates [9 in-window], 20-scene cap, 5 scenes outside Nov–Feb on Mar 3, Mar 18, Mar 28).
>   - **2024**: Oct 1, 2023 – Mar 31, 2024 (11 dates [5 in-window], 20-scene cap, 11 scenes outside Nov–Feb on Oct 4, Oct 29, Mar 7, Mar 12, Mar 17, Mar 22).
> - **Pune**:
>   - **2020**: Nov 1, 2019 – Feb 29, 2020 (12 dates, 20-scene cap, 0 outside Nov–Feb).
>   - **2021**: Nov 1, 2020 – Feb 28, 2021 (12 dates, 20-scene cap, 0 outside Nov–Feb).
>   - **2022**: Nov 1, 2021 – Feb 28, 2022 (query window; last scene Dec 25, 2021, Nov-Dec 2021, Collection 1; 5 dates, 10 scenes, 0 outside Nov–Feb).
>   - **2023**: Nov 1, 2022 – Feb 28, 2023 (12 dates, 20-scene cap, 0 outside Nov–Feb).
>   - **2024**: Nov 1, 2023 – Feb 29, 2024 (12 dates, 20-scene cap, 0 outside Nov–Feb).

---

### 1. Ahmedabad (Gujarat, India) — 46.6 × 46.6 km AOI (2,167.83 km²)

- **Headline 2020–2024 Expansion Range**: **+53.0 to +84.0 km²** across three ways of measuring: raw, cleaned and normalised (Main series: **+58.91 km²** / +14.2%).
- **2021 Benchmark Anchor**: ESA WorldCover 2021 built-up ground truth = **393.73 km²** (18.19% of AOI) vs. 2021 Main series (normalised) estimate of **413.61 km²**.
- **2024 Footprint**: Main series (normalised) built-up area of **474.88 km²** (21.91% of AOI; Cleaned sensitivity: 468.54 km²).
- **Radial Dispersion & Entropy**: Core (0–6 km) built-up share: **21.9%**, Peripheral (>12 km) share: **32.5%**. Shannon spatial entropy: **0.9501** (how evenly built-up land is spread across distance rings from the centre; values near 1 mean it is not concentrated near the centre, not measuring leapfrog development).

#### Multi-Series Growth & Method Sensitivity Band (Ahmedabad)

| Year                            | Cleaned (sensitivity, km²) | Raw Classified (km²) | Main series (normalised) (km²) |    Method Sensitivity Spread    | WorldCover 2021 Anchor |
| :------------------------------ | :-------------------------: | :-------------------: | :-----------------------------: | :-----------------------------: | :--------------------: |
| **2020**                  |           384.51           |        421.06        |             415.97             |      [384.5 – 421.1 km²]      |           —           |
| **2021**                  |           414.07           |        413.61        |             413.61             |      [413.6 – 414.1 km²]      | **393.73 km²** |
| **2022 (partial season)** |           441.91           |        423.91        |             426.94             |      [423.9 – 441.9 km²]      |           —           |
| **2023**                  |           474.54           |        504.42        |             451.61             |      [451.6 – 504.4 km²]      |           —           |
| **2024**                  |      **468.54**      |   **474.02**   |        **474.88**        | **[468.5 – 474.9 km²]** |           —           |

---

### 2. Pune (Maharashtra, India) — 45.4 × 45.4 km AOI (2,057.53 km²)

- **Headline 2020–2024 Expansion Range**: **+62.7 to +92.1 km²** across three ways of measuring: raw, cleaned and normalised (Main series: **+62.68 km²** / +15.4%).
- **2021 Benchmark Anchor**: ESA WorldCover 2021 built-up ground truth = **378.08 km²** (18.43% of AOI) vs. 2021 Main series (normalised) estimate of **386.55 km²**.
- **2024 Footprint**: Main series (normalised) built-up area of **469.80 km²** (22.83% of AOI; Cleaned sensitivity: 448.31 km²).
- **Radial Dispersion & Entropy**: Core (0–6 km) built-up share: **11.0%**, Peripheral (>12 km) share: **51.7%**. Shannon spatial entropy: **0.9544**.

#### Multi-Series Growth & Method Sensitivity Band (Pune)

| Year                            | Cleaned (sensitivity, km²) | Raw Classified (km²) | Main series (normalised) (km²) |    Method Sensitivity Spread    | WorldCover 2021 Anchor |
| :------------------------------ | :-------------------------: | :-------------------: | :-----------------------------: | :-----------------------------: | :--------------------: |
| **2020**                  |           356.19           |        332.13        |             407.12             |      [332.1 – 407.1 km²]      |           —           |
| **2021**                  |           421.99           |        386.55        |             386.55             |      [386.6 – 422.0 km²]      | **378.08 km²** |
| **2022 (partial season)** |           443.75           |        692.99*        |             393.07             |      [393.1 – 443.8 km²]      |           —           |
| **2023**                  |           479.78           |        359.65        |             440.15             |      [359.7 – 479.8 km²]      |           —           |
| **2024**                  |      **448.31**      |   **406.65**   |        **469.80**        | **[406.7 – 469.8 km²]** |           —           |

*\*2022 raw point (692.99 km²) is excluded from sensitivity bounds (5-date composite containing an anomalous scene, 2021-12-05).*

---

## 🧪 Validation & Negative Result

### Leave-One-Year-Out (LOYO) Validation (All Held-Out Points)

To test temporal generalization and prevent data leakage, spatial classifiers were trained with one year completely held out. Performance was evaluated on **all held-out points** ($N = 400\text{ to }414$ points per fold), with area estimation and 95% confidence intervals computed via stratified area-weighted adjustment (Olofsson et al. 2014):

| City                |      Held-Out Year      | Test Points ($N$) | Raw Built-up F1 | Raw Adjusted Area (95% CI) | Main series Built-up F1 | Norm Adjusted Area (95% CI) |
| :------------------ | :----------------------: | :-----------------: | :-------------: | :------------------------: | :---------------------: | :-------------------------: |
| **Ahmedabad** | 2018*(outside window)* |         400         |     0.7079     |     393.6 ± 64.9 km²     |         0.7543         |     403.8 ± 61.5 km²     |
| **Ahmedabad** |           2021           |         400         |     0.7953     |     414.9 ± 65.3 km²     |         0.7791         |     378.3 ± 55.5 km²     |
| **Ahmedabad** |           2024           |         400         |     0.7513     |     415.4 ± 67.8 km²     |         0.7213         |     407.3 ± 69.0 km²     |
| **Pune**      | 2018*(outside window)* |         414         |     0.3191     |     177.4 ± 61.0 km²     |         0.3226         |     199.7 ± 65.0 km²     |
| **Pune**      |           2021           |         414         |     0.6173     |     339.6 ± 77.4 km²     |         0.6582         |     353.1 ± 76.0 km²     |
| **Pune**      |           2024           |         414         |     0.5591     |     291.3 ± 72.9 km²     |         0.5800         |     306.6 ± 74.6 km²     |

> **Note on LOYO Validation**: Reference labels are WorldCover 2021 for every fold, so adjusted areas for 2018 and 2024 are not comparable with the change-validation estimates.

### 🔍 4-Stratum Change Validation (2020–2024) & Area Adjustment

To independently validate multi-temporal land cover transitions and compute error-adjusted area estimates, a probability sample of $N=300$ verification points across 4 spatial strata was visually audited following Olofsson et al. (2014).

#### Labelling Protocol & History

- **Interpreter**: Single interpreter (the project author); first pass blind to strata, recheck of 10 discordant points was unblinded.
- **Imagery Sources**: Paired Sentinel-2 10m dry-season RGB surface reflectance chips (2020 vs 2024) corroborated against high-resolution Google Earth Pro historical satellite imagery.
- **Ambiguity Rule**: Points exhibiting mixed-pixel ambiguity or low visual contrast were flagged as `unclear` (3 points) and excluded from primary estimation; sensitivity bounds were evaluated treating all unclear points as built vs. non-built.
- **Labelling History**:
  1. *First Pass*: Conducted on initial 2024 Sentinel-2 chips.
  2. *Second Pass*: Conducted with corrected reflectance stretch chips (26 of 291 start labels changed: 12 built to not built, 14 not built to built, establishing the labeller's change rate at ~9%, and 30 end labels changed from not built to built).
  3. *Third Pass (Google Earth Pro Recheck)*: Rechecked 10 discordant points (apparent losses and gains) using Google Earth Pro historical timeline imagery, updating 6 labels (IDs 81, 91, 94, 123 in Stratum C; IDs 151, 226 in Stratum D). The 6 corrected labels persisted through the second pass, and concordant points were not rechecked. These initial discrepancies were labelling errors in the first pass (bare or ploughed soil read as built-up on 2020 Sentinel-2 chips), corrected using Google Earth historical imagery (not classifier errors).

> **Disclaimer**: *Intervals reflect sampling error only; labelling inconsistency (about 9% between passes) is not included.*

#### Per-Stratum Evaluation & Area Adjustment (Ahmedabad)

| Stratum                                    | Mapped Area (km²) | Evaluated ($N$) | $(0,0)$ Persistent Non-built | $(0,1)$ True Gain | $(1,0)$ True Loss | $(1,1)$ Persistent Built | Unclear | Stratum Accuracy |
| :----------------------------------------- | :----------------: | :---------------: | :----------------------------: | :-----------------: | :-----------------: | :------------------------: | :-----: | :--------------: |
| **Stratum A (Mapped Gain)**          |       116.65       |        99        |               40               |         41         |          0          |             18             |    1    | **41.41%** |
| **Stratum B (Persistent Built)**     |       358.23       |        50        |               2               |          1          |          1          |             46             |    0    | **92.00%** |
| **Stratum C (Persistent Non-built)** |      1635.22      |        98        |               94               |          2          |          0          |             2             |    2    | **95.92%** |
| **Stratum D (Mapped Loss)**          |       57.74       |        50        |               45               |          0          |          0          |             5             |    0    | **0.00%** |

#### Olofsson Adjusted Change & Extent Estimates (95% Confidence Intervals)

| Metric                              | Mapped (km²) | Area-Adjusted Estimate (km²) | Standard Error (SE) |     95% Confidence Interval     |
| :---------------------------------- | :-----------: | :---------------------------: | :-----------------: | :------------------------------: |
| **Gross Built-up Gain**       |    116.65    |        **88.85**        |       ±25.25       |  **[39.36, 138.33] km²**  |
| **Gross Built-up Loss**       |     57.74     |        **7.16**        |       ±18.00       |   **[0.00, 42.45] km²**   |
| **Net Built-up Change**       |    +58.91    |       **+81.68**       |       ±26.31       | **[+30.12, +133.24] km²** |
| **Built-up Footprint (2020)** |    415.97    |       **397.09**       |       ±26.94       | **[344.29, 449.88] km²** |
| **Built-up Footprint (2024)** |    474.88    |       **478.77**       |       ±35.59       | **[409.02, 548.52] km²** |

#### Sensitivity & Baseline Scenarios

- **Pre-recheck Baseline**: Net Change = `+14.9 ± 83.6 km²` (dominated by apparent loss variance in Stratum C).
- **Without Stratum C Gains (IDs 50 & 172 as errors)**: Net Change = `+48.3 ± 23.1 km²`.
- **Post-recheck Adjusted Net (Primary)**: Net Change = `+81.7 ± 51.6 km²` ($95\%\text{ CI}: [+30.12, +133.24]\text{ km}^2$).
- **Adjusted 2024 Built-up Footprint**: **478.8 km² (95% CI 409.0-549.0)**.

> **Conclusion**: *Validated on 297 points (Ahmedabad only); net change is distinguishable from zero post-recheck (+81.7 ± 51.6 km²), but depends on the recheck (pre-recheck: +14.9 ± 83.6 km²; without Stratum C gains: +48.3 ± 23.1 km²). Adjusted 2024 built-up footprint: 478.8 km² (95% CI 409.0-549.0).* (Pune: *not independently validated*).

*(Note: "±" in text and sensitivity scenarios denotes the 95% CI half-width, i.e., $1.96 \times \text{SE}$).*

*Methodology Notes*: Strata were built from annual TLS-normalised classifications (`ahmedabad_2020_classified.tif` and `ahmedabad_2024_classified.tif`) with a 3x3 majority filter (no temporal consistency cleanup rules). Validation mapped areas (415.97 and 474.88 km²) match the dashboard TLS series (415.97 and 474.88 km²), both produced by the 3x3 majority filtered TLS classification pipeline without temporal filtering. Gross loss standard error applies Laplace (add-one) smoothing for zero-count sample proportions. Validated series is the TLS-normalised classification.

### ⚠️ Negative Result: Cross-Year Radiometric Normalisation

Cross-year Total Least Squares (TLS) pseudo-invariant feature (PIF) radiometric normalisation was implemented and systematically benchmarked against raw surface reflectance composites.

**Finding**: Radiometric normalisation **did not reduce year-to-year classification drift** across held-out evaluation folds; the displayed series has no temporal filtering, and the cleaned series is a sensitivity comparison. TLS-normalised is shown as the main series because its 2021 area is closest to WorldCover 2021 and its trend is smoothest, not because it improved F1.

> **Methodological Disclosure**: Ahmedabad's 2021, 2023 and 2024 composites use the 20 lowest-cloud scenes between October and March, so they include October and March scenes (11 of 20 in 2024). Pune's composites use November-February only. Ahmedabad's 2020 composite was built earlier with a strict Dec 1-Feb 15 window; current code does not reproduce its exact scene list. How much the window affects Ahmedabad's built-up area has not been tested.

---

## 🛡️ Data Quality Gate & CI Status

Quality gates and CI health are computed dynamically from pipeline execution outputs:

- **Ahmedabad Quality Gate**: FLAGGED (4/5 PASS) — Composite NoData: PASS (0.0%), Scenes in Window: PASS (min 5 dates), YoY Volatility: PASS (5.2%), Agreement with WorldCover: PASS (min 72.1% F1), Loss-to-Gain Ratio: FAIL (0.495 mapped; 0.081 adjusted).
- **Pune Quality Gate**: FLAGGED (3/5 PASS) — Composite NoData: PASS (1.1%), Scenes in Window: PASS (min 5 dates), YoY Volatility: PASS (6.7%), Agreement with WorldCover: FAIL (min 58.0% F1), Loss-to-Gain Ratio: FAIL (0.469).
- **Continuous Integration**: 97/97 automated unit and integration tests passing (17% unit line coverage across analytical modules).

---

## ⚡ Quickstart

Start the entire local application stack (Web frontend, FastAPI backend, and PostGIS database) with Docker Compose:

```bash
# Clone the repository
git clone https://github.com/HUSAINTRIVEDI52/urban-pulse-geospatial.git
cd urban-pulse-geospatial

# Spin up services
make up
```

- 🖥️ **Web Dashboard**: [http://localhost:8080/](http://localhost:8080/)
- 📖 **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 📊 **Monitoring Stack**: `docker compose --profile monitoring up -d` (Grafana at `http://localhost:3000`, admin/admin)

### Run Pipeline & Generate Reports:

```bash
# Execute end-to-end pipeline for any city
python -m pipeline.run_city --city pune

# Generate HTML report and PDF export
make report
make report-pdf
```

---

## 🔍 Visual Change Validation & Stratified Estimation

UrbanPulse provides an end-to-end blind visual accuracy validation and area-adjustment suite based on **Olofsson et al. (2014)**:

1. **Stratified Sampling** (A: Gain, B: Persistent Built, C: Persistent Non-Built, D: Loss):

   ```bash
   python pipeline/make_change_validation_sample.py --city ahmedabad --start 2020 --end 2024 --seed 42
   ```

   - Filters out pixels within 2 pixels of stratum edges.
   - Enforces pairwise spacing $\ge 500\text{ m}$.
   - Exports `change_sample_blind.csv`, `change_sample_key.csv`, and Google Earth `change_sample.kml`.
2. **Generate 10m True Colour Chips & Standalone Labeller**:

   ```bash
   python pipeline/make_label_chips.py --city ahmedabad --start 2020 --end 2024
   ```

   - Builds 10 m true colour (B04, B03, B02) composites with SCL cloud mask and 2%-98% fixed percentile stretch.
   - Cuts $128\times 128$ chips, upscales 4x with bicubic resampling, and overlays the 60m center classifier pixel.
   - Generates `data/{city}/validation/labeller.html` (single self-contained file with offline shortcuts and Google Maps satellite integration).
3. **Label Samples with Keyboard Shortcuts**:
   Open `data/{city}/validation/labeller.html` in any web browser:

   - `Q` / `W` / `E` = Start year Built (`Y`) / Not built (`N`) / Unclear (`unclear`)
   - `I` / `O` / `P` = End year Built (`Y`) / Not built (`N`) / Unclear (`unclear`)
   - `←` / `→` = Previous / Next point
   - `S` = Edit notes
   - Click **Download CSV** to export labelled answers.
4. **Score Accuracy & Compute Area-Adjusted Change**:

   ```bash
   python pipeline/score_change_validation.py --sample-csv data/ahmedabad/validation/change_sample_labelled_ahmedabad.csv
   ```

   - Computes per-stratum accuracy matrix.
   - Estimates area-adjusted **Gross Gain**, **Gross Loss**, and **NET Change** with 95% CIs and continuity correction for boundary proportions ($p=0$ or $1$).

---

## 📁 Project Tree

```
UrbanPulse/
├── configs/                     # City YAML configurations (bbox, dry-season dates, classes)
├── pipeline/                    # Earth Observation & ML Pipeline
│   ├── build_composite.py       # SCL cloud-masked median compositing & radiometric calibration
│   ├── normalize_radiometry.py  # TLS pseudo-invariant feature radiometric normalisation
│   ├── train_classifier.py      # Spatial-block Random Forest model training
│   ├── temporal_cleanup.py      # Majority rule & urban persistence smoothing
│   ├── change_detection.py      # Land cover transition matrix & trajectories
│   ├── make_change_validation_sample.py # 4-stratum sampling with edge buffer & 500m spacing
│   ├── make_label_chips.py      # 10m true colour Sentinel-2 chip cutter & labeller generator
│   ├── score_change_validation.py       # Olofsson et al. (2014) area-adjusted change & 95% CIs
│   ├── ring_analysis.py         # Concentric radial distance density profiling

│   ├── sprawl_metrics.py        # Shannon entropy & sprawl velocity computation
│   ├── quality_gate.py          # Data quality gate validator
│   ├── generate_report.py       # Professional self-contained HTML report generator
│   ├── export_report_pdf.py     # Playwright Chromium headless PDF exporter
│   └── run_city.py              # City pipeline orchestrator
├── api/                         # FastAPI Geospatial Backend (FastAPI, PostGIS queries, /metrics)
├── web/                         # MapLibre GL JS + Chart.js Dashboard (static CDN bundle)
├── db/                          # PostgreSQL 16 + PostGIS 3.4 spatial schemas & migrations
├── infra/                       # Infrastructure as Code (Terraform, Ansible, k3s manifests)
├── monitoring/                  # Observability (Prometheus alerts, Grafana dashboards)
├── docs/                        # Architecture & Technical Documentation
│   ├── report/                  # Generated HTML & PDF reports
│   └── engineering-decisions.md # Rationale and trade-offs
├── tests/                       # Automated Pytest suite
├── Makefile                     # Developer and automation workflows
└── pyproject.toml               # Python package configuration
```

---

## 🏙️ How to Add a New City

UrbanPulse requires **zero code changes** to onboard a new metropolitan area:

1. Create `configs/<city_name>.yaml`:
   ```yaml
   city:
     name: Hyderabad
     state: Telangana
     country: India
   spatial:
     bbox: [78.2000, 17.2000, 78.6500, 17.6000] # ~45x45 km bounding envelope
     crs: EPSG:4326
     center_lat: 17.3850
     center_lon: 78.4867
     buffered_distance_km: 8.0
     aoi_size_km: 45.0
   temporal:
     analysis_years:
       start_year: 2020
       end_year: 2024
     strict_window:
       start_month: 11
       start_day: 11-01
       end_month: 2
       end_day: 02-28
     dry_season:
       start_month: 11
       start_day: 11-01
       end_month: 2
       end_day: 02-28
   ```
2. Execute the pipeline:
   ```bash
   python -m pipeline.run_city --city hyderabad
   ```
3. The new city immediately populates the web app, PostGIS database, and documentation reports.

---

## ⚠️ Limitations

- **Fallow & Bare Land Confusion (Stratum A)**: About 40% of mapped gain is fallow or bare land (computed from Stratum A: 40 of 99 sampled gain points were persistent non-built).
- **2020 Commission Errors (Stratum D)**: The 2020 map falsely marks about 52 km² as built-up (45 of 50 mapped-loss points were never built-up, yielding a 90% false-loss rate across the 57.74 km² mapped loss stratum).
- **Ahmedabad Observation Windows**: Ahmedabad's windows differ between 2020 (built earlier with strict Dec 1–Feb 15) and 2021–2024 (Oct 1–Mar 31).
- **Partial 2022 Season**: 2022 covers October–December 2021 only (partial season with no January–February scenes).
- **Pune Validation Status**: Pune is not independently validated with reference stratified points; metrics reflect WorldCover agreement only.
- **Pune 2021 Trajectory Dip**: Pune's mapped built-up series dips in 2021 (from 407.12 km² in 2020 to 386.55 km² in 2021); cause not established; consistent with classifier noise (Pune built-up F1 0.58) before recovering to 469.80 km² in 2024.
- **Sensor Resolution**: 10–60 m Sentinel-2 pixel resolution limits detection of narrow roads, informal settlements, and sub-pixel urban canopy.
- **Model Transferability**: Machine learning classifiers are sensor-specific and trained on Sentinel-2 MSI surface reflectance; cross-sensor transfer (e.g. Landsat) requires separate radiometric harmonisation.

---

## 📜 License & Contributions

- **License**: Released under the [MIT License](LICENSE).
- **Contributing**: Please review [CONTRIBUTING.md](docs/CONTRIBUTING.md) for code formatting, tests, and pull request guidelines.
