// web/js/methodology-diagram.js
// Visual Methodology & Architecture Diagram for UrbanPulse
// Displays full data lineage, Python libraries, algorithms, and interactive fullscreen modal.

export const PIPELINE_STAGES = [
  {
    id: 'stage-1',
    category: 'ingestion',
    colorClass: 'stage-ingest',
    number: '01',
    icon: '🛰️',
    name: 'Satellite Ingestion & AOI',
    categoryName: 'Data Acquisition',
    headline: 'Copernicus Sentinel-2 MSI Level-2A Query & Cataloging',
    desc: 'Automated retrieval of multi-spectral Bottom-Of-Atmosphere (BOA) surface reflectance tiles covering 45x45 km Metropolitan AOIs.',
    inputs: 'ESA Copernicus S2 L2A Archive, AOI Bounding Box GeoJSON',
    outputs: 'Raw Multi-band Surface Reflectance Tiles (10m & 20m)',
    libraries: [
      { name: 'pystac-client', type: 'geo' },
      { name: 'planetary-computer', type: 'geo' },
      { name: 'requests', type: 'primary' }
    ],
    details: `
      <ul>
        <li><strong>Query Engine:</strong> Element84 Earth Search STAC API endpoint queried for Sentinel-2 Level-2A assets.</li>
        <li><strong>Atmospheric Correction:</strong> Utilizes Bottom-Of-Atmosphere (BOA) reflectances to normalize for ozone, water vapor, and aerosols across multi-year comparisons.</li>
        <li><strong>Metropolitan Bounding Boxes:</strong> Standardized 45x45 km UTM projected boundaries (Ahmedabad: UTM Zone 43N / Pune: UTM Zone 43N).</li>
        <li><strong>Reference Baseline:</strong> ESA WorldCover 2021 global 10m land cover map ingested as the macro-calibration anchor.</li>
      </ul>
    `
  },
  {
    id: 'stage-2',
    category: 'preprocessing',
    colorClass: 'stage-cloud',
    number: '02',
    icon: '☁️',
    name: 'Cloud Masking & Compositing',
    categoryName: 'Raster Preprocessing',
    headline: 'Scene Classification Bitmasking & Annual Median Mosaic',
    desc: 'Filters atmospheric artifacts and blends the cleanest dry-season observations into a cloud-free annual surface composite.',
    inputs: 'Raw S2 L2A bands + Scene Classification Layer (SCL)',
    outputs: 'Cloud-free 13-band surface reflectance GeoTIFF mosaic',
    libraries: [
      { name: 'rasterio', type: 'geo' },
      { name: 'numpy', type: 'math' }
    ],
    details: `
      <ul>
        <li><strong>SCL Quality Masking:</strong> Pixels flagged as cloud shadows (3), cloud high/medium probability (8, 9), thin cirrus (10), or snow/ice (11) are masked to NaN.</li>
        <li><strong>Dry-Season Observation Window:</strong> Constrained to post-monsoon winter months (Nov–Feb / Dec–Feb) to minimize agricultural crop phenology distortion.</li>
        <li><strong>20-Scene Ranking:</strong> The 20 least-cloudy scenes in the target window are selected and stacked.</li>
        <li><strong>Temporal Pixel Median:</strong> Computes the median reflectance across valid non-masked pixels per coordinate, eliminating transient vehicles, shadows, and smoke.</li>
      </ul>
    `
  },
  {
    id: 'stage-3',
    category: 'features',
    colorClass: 'stage-features',
    number: '03',
    icon: '📊',
    name: 'Spectral Feature Engineering',
    categoryName: 'Feature Extraction',
    headline: 'Resampling & Multispectral Index Calculation',
    desc: 'Harmonizes bands to a 60m analysis grid and derives specialized indices highlighting concrete, vegetation, water, and bare soil.',
    inputs: '13 Sentinel-2 bands (B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12)',
    outputs: '13-feature stacked multidimensional raster matrix',
    libraries: [
      { name: 'numpy', type: 'math' },
      { name: 'rasterio', type: 'geo' }
    ],
    details: `
      <ul>
        <li><strong>Spatial Resampling:</strong> 10m visible/NIR and 20m SWIR bands harmonized to a 60m x 60m spatial pixel grid (0.0036 km² per pixel) for regional scalability.</li>
        <li><strong>NDVI (Normalized Difference Vegetation Index):</strong> (B08 - B04) / (B08 + B04) — isolates healthy canopy and tree cover.</li>
        <li><strong>NDBI (Normalized Difference Built-up Index):</strong> (B11 - B08) / (B11 + B08) — enhances impervious concrete, asphalt, and rooftops.</li>
        <li><strong>MNDWI (Modified ND Water Index):</strong> (B03 - B11) / (B03 + B11) — sharply demarcates lakes, reservoirs, and rivers.</li>
        <li><strong>BSI (Bare Soil Index):</strong> ((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02)) — distinguishes fallow fields from built structures.</li>
        <li><strong>UI (Urban Index):</strong> (B12 - B08) / (B12 + B08) — separates dense urban fabric from industrial bare ground.</li>
      </ul>
    `
  },
  {
    id: 'stage-4',
    category: 'ml',
    colorClass: 'stage-ml',
    number: '04',
    icon: '🤖',
    name: 'Machine Learning Classification',
    categoryName: 'Supervised ML',
    headline: 'Random Forest Multi-Class Land Cover Classifier',
    desc: 'Supervised classification allocating pixels to 5 discrete land cover classes with spatial cross-validation.',
    inputs: '13-channel feature raster + Ground reference training vectors',
    outputs: 'Annual 5-class classified raster (*_classified.tif)',
    libraries: [
      { name: 'scikit-learn', type: 'ml' },
      { name: 'joblib', type: 'primary' }
    ],
    details: `
      <ul>
        <li><strong>Classification Schema:</strong> 5 distinct land cover categories: Built-up (Red), Vegetation (Green), Water (Blue), Agriculture (Yellow), Open Land / Fallow (Brown).</li>
        <li><strong>Model Architecture:</strong> Scikit-learn <code>RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)</code>.</li>
        <li><strong>Spatial Block Cross-Validation:</strong> Held-out spatial blocks used to prevent spatial autocorrelation leakage between training and validation points.</li>
        <li><strong>Feature Importance:</strong> SWIR-1 (B11), NDBI, and BSI exhibit highest Gini importance for resolving semi-arid soil vs. concrete confusion.</li>
      </ul>
    `
  },
  {
    id: 'stage-5',
    category: 'postprocessing',
    colorClass: 'stage-tls',
    number: '05',
    icon: '🔄',
    name: 'Temporal Smoothing & Filter',
    categoryName: 'Signal Processing',
    headline: 'Temporal Logistic Smoothing (TLS) & 3x3 Spatial Mode',
    desc: 'Removes annual spectral flicker and false urbanization cycles using multi-year trajectory modeling.',
    inputs: 'Multi-year classified time series (2020–2024)',
    outputs: 'Temporal Logistic Smoothed annual raster series (*_tls.tif)',
    libraries: [
      { name: 'scipy.ndimage', type: 'math' },
      { name: 'numpy', type: 'math' }
    ],
    details: `
      <ul>
        <li><strong>The Problem:</strong> Atmospheric variation and seasonal fallow fields can cause false reversals (e.g., built-up in 2021, farmland in 2022, built-up in 2023).</li>
        <li><strong>TLS Normalization:</strong> Fits a monotonic logistic trajectory across annual built-up class probabilities, enforcing that true permanent urban structures do not revert to vacant desert.</li>
        <li><strong>3x3 Majority Filter:</strong> Spatial mode filter removes isolated single-pixel raster noise ('salt-and-pepper' artifacts) across homogeneous urban blocks.</li>
      </ul>
    `
  },
  {
    id: 'stage-6',
    category: 'metrics',
    colorClass: 'stage-rings',
    number: '06',
    icon: '🎯',
    name: 'Urban Sprawl & Morphological Metrics',
    categoryName: 'Spatial Analysis',
    headline: 'Concentric Ring Buffers & Shannon Entropy Calculation',
    desc: 'Quantifies whether metropolitan growth is densifying historical cores or sprawling radially outward into rural peripheries.',
    inputs: 'TLS classified built-up raster + Municipal centroid coordinates',
    outputs: 'stats.json concentric ring gradients & Shannon Entropy indices',
    libraries: [
      { name: 'geopandas', type: 'geo' },
      { name: 'shapely', type: 'geo' },
      { name: 'pyproj', type: 'geo' }
    ],
    details: `
      <ul>
        <li><strong>Concentric Ring Buffering:</strong> 2 km radial zones generated outward from the city center (0–2 km, 2–4 km ... out to 16–18 km).</li>
        <li><strong>Shannon Entropy ($H_n$):</strong> Derived from information theory: <code>H_n = -sum(p_i * ln(p_i)) / ln(k)</code>, measuring spatial dispersion. A value near 1.0 (e.g. 0.9501) indicates high sprawl and leapfrog outward expansion.</li>
        <li><strong>Core vs. Periphery Breakdown:</strong> Tallying built-up share in the historic core (0–6 km) versus expanding outer rural rings (>12 km).</li>
      </ul>
    `
  },
  {
    id: 'stage-7',
    category: 'validation',
    colorClass: 'stage-olofsson',
    number: '07',
    icon: '📐',
    name: 'Stratified Ground Validation',
    categoryName: 'Statistical Audit',
    headline: 'Olofsson (2014) Error Matrix & 95% Confidence Intervals',
    desc: 'Gold-standard statistical auditing correcting for map bias using hand-verified human ground-truth control points.',
    inputs: 'Mapped change map, 297 human-labelled ground-truth audit points',
    outputs: 'Unbiased error-adjusted net area change ± 95% Confidence Bounds',
    libraries: [
      { name: 'scipy.stats', type: 'math' },
      { name: 'pandas', type: 'primary' },
      { name: 'numpy', type: 'math' }
    ],
    details: `
      <ul>
        <li><strong>Olofsson (2014) 4-Stratum Design:</strong> Samples allocated across Stratum A (mapped gain), B (mapped loss), C (stable built-up), and D (stable non-built-up).</li>
        <li><strong>Audit Sample:</strong> N=297 points independently inspected against high-resolution Google Earth / Planet historical imagery.</li>
        <li><strong>Error-Adjusted Net Change:</strong> Corrects raw pixel tallies for false positives. In Ahmedabad, confirmed net growth of +81.7 ± 51.6 km² at 95% confidence.</li>
        <li><strong>Fallow Soil Disclosure:</strong> Discloses that ~4 in 10 raw mapped gain points were seasonal dry bare soil rather than permanent buildings.</li>
      </ul>
    `
  },
  {
    id: 'stage-8',
    category: 'ops',
    colorClass: 'stage-ops',
    number: '08',
    icon: '🚀',
    name: 'CI/CD Quality Gates & Web App',
    categoryName: 'DevOps & Frontend',
    headline: 'Automated Pipeline Tests & High-Performance Visual Dashboard',
    desc: 'Ensures data integrity with 5 automated quality gates and serves interactive vector/raster exploration.',
    inputs: 'Classified GeoTIFFs, stats.json, metadata.json',
    outputs: 'Interactive MapLibre GL web client & published metrics',
    libraries: [
      { name: 'pytest', type: 'primary' },
      { name: 'Docker', type: 'primary' },
      { name: 'Kubernetes (K3d)', type: 'primary' },
      { name: 'Prometheus', type: 'primary' },
      { name: 'FastAPI', type: 'primary' },
      { name: 'MapLibre GL', type: 'geo' },
      { name: 'Chart.js', type: 'primary' }
    ],
    details: `
      <ul>
        <li><strong>5 Automated Quality Gates:</strong> Validates cloud cover <20%, bounds match, pixel counts positive, area conservation, and zero NaN values.</li>
        <li><strong>Infrastructure:</strong> Dockerized reproducible pipeline orchestrated via Kubernetes (K3d) with Prometheus metric telemetry.</li>
        <li><strong>Frontend Architecture:</strong> Vanilla JavaScript, MapLibre GL JS GPU raster tile compositing, swipe-divider compare mode, and Chart.js responsive analytics.</li>
      </ul>
    `
  }
];

/**
 * Renders the Visual Methodology Diagram into container
 */
export function renderMethodologyDiagram(containerId = 'how-diagram-root') {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = `
    <div class="methodology-hero-card">
      <div class="methodology-header">
        <div class="methodology-title-wrap">
          <h2>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
            </svg>
            Analytical Pipeline & Engineering Methodology
          </h2>
          <p>Complete data lineage from raw Sentinel-2 satellite pixels to validated sprawl metrics using open-source Python libraries.</p>
        </div>
        
        <div class="methodology-controls">
          <button id="btn-fullscreen-diagram" class="method-btn active" title="Open Fullscreen Zoomable View">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>
            </svg>
            <span>Expand Fullscreen</span>
          </button>
          <button id="btn-toggle-diagram-detail" class="method-btn" title="Toggle technical implementation notes">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>
            </svg>
            <span>Technical Spec</span>
          </button>
        </div>
      </div>

      <!-- Category Filter Pills -->
      <div class="methodology-filters" role="tablist" aria-label="Filter pipeline stages">
        <button class="filter-chip active" data-filter="all">All Stages (8)</button>
        <button class="filter-chip" data-filter="ingestion">1. Data Ingestion</button>
        <button class="filter-chip" data-filter="preprocessing">2. Cloud Masking</button>
        <button class="filter-chip" data-filter="features">3. Spectral Indices</button>
        <button class="filter-chip" data-filter="ml">4. Machine Learning</button>
        <button class="filter-chip" data-filter="postprocessing">5. Temporal Smoothing</button>
        <button class="filter-chip" data-filter="metrics">6. Spatial Metrics</button>
        <button class="filter-chip" data-filter="validation">7. Ground Audit</button>
        <button class="filter-chip" data-filter="ops">8. Cloud Stack & UI</button>
      </div>

      <!-- Flowchart Visual Diagram Viewport -->
      <div class="diagram-viewport" id="diagram-viewport">
        <div class="diagram-flow" id="diagram-flow-container">
          ${renderStageCards(PIPELINE_STAGES)}
        </div>
      </div>

      <!-- Stage Detail Inspection Drawer -->
      <div id="stage-detail-drawer" class="stage-detail-panel">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
          <div>
            <div id="drawer-stage-badge" class="stage-num-badge" style="display: inline-block; margin-bottom: 4px;">Stage 01</div>
            <h3 id="drawer-stage-title" style="margin: 0; font-family: var(--font-display); font-size: 16px;">Stage Title</h3>
          </div>
          <button id="drawer-close-btn" class="tool-btn" style="padding: 2px 8px;" aria-label="Close details">&times;</button>
        </div>
        <div id="drawer-stage-content" style="font-size: 13px; line-height: 1.6; color: var(--color-text-secondary);"></div>
      </div>
    </div>
  `;

  attachDiagramEvents();
}

function renderStageCards(stages) {
  return stages.map(s => `
    <div class="stage-card ${s.colorClass}" data-stage-id="${s.id}" data-category="${s.category}" tabindex="0" role="button" aria-label="Inspect ${s.name}">
      <div>
        <div class="stage-top">
          <span class="stage-num-badge">STAGE ${s.number}</span>
          <div class="stage-icon">${s.icon}</div>
        </div>
        <div class="stage-name">${s.name}</div>
        <div class="stage-desc">${s.desc}</div>
      </div>

      <div>
        <div class="stage-io-box">
          <div class="io-row">
            <span class="io-tag">IN</span>
            <span class="io-val">${s.inputs}</span>
          </div>
          <div class="io-row">
            <span class="io-tag">OUT</span>
            <span class="io-val">${s.outputs}</span>
          </div>
        </div>

        <div class="stage-libs-title">
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
          Python Libraries & Tools
        </div>
        <div class="stage-libs">
          ${s.libraries.map(lib => `
            <span class="lib-badge ${lib.type}"><code>${lib.name}</code></span>
          `).join('')}
        </div>
      </div>
    </div>
  `).join('');
}

function attachDiagramEvents() {
  const container = document.getElementById('diagram-flow-container');
  const drawer = document.getElementById('stage-detail-drawer');
  const drawerTitle = document.getElementById('drawer-stage-title');
  const drawerBadge = document.getElementById('drawer-stage-badge');
  const drawerContent = document.getElementById('drawer-stage-content');
  const drawerClose = document.getElementById('drawer-close-btn');

  // Filter chips
  document.querySelectorAll('.filter-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-chip').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filter = btn.getAttribute('data-filter');

      document.querySelectorAll('.stage-card').forEach(card => {
        if (filter === 'all' || card.getAttribute('data-category') === filter) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });

  // Stage card click -> show drawer
  document.querySelectorAll('.stage-card').forEach(card => {
    const handleSelect = () => {
      document.querySelectorAll('.stage-card').forEach(c => c.classList.remove('highlighted'));
      card.classList.add('highlighted');

      const stageId = card.getAttribute('data-stage-id');
      const stage = PIPELINE_STAGES.find(s => s.id === stageId);
      if (stage && drawer) {
        drawerBadge.textContent = `Stage ${stage.number} • ${stage.categoryName}`;
        drawerTitle.textContent = `${stage.icon} ${stage.name}: ${stage.headline}`;
        drawerContent.innerHTML = `
          <div style="margin-bottom: 12px; font-weight: 500; color: var(--color-text-main);">${stage.desc}</div>
          <div style="margin-bottom: 12px;">${stage.details}</div>
          <div style="background: var(--color-bg-subtle); padding: 10px 14px; border-radius: var(--radius-sm); border: 1px solid var(--color-border); font-size: 12px;">
            <strong>Python Libraries & Toolchain:</strong>
            ${stage.libraries.map(l => `<span class="lib-badge ${l.type}" style="margin: 2px 4px;">${l.name}</span>`).join('')}
          </div>
        `;
        drawer.classList.add('visible');
        drawer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    };

    card.addEventListener('click', handleSelect);
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        handleSelect();
      }
    });
  });

  if (drawerClose && drawer) {
    drawerClose.addEventListener('click', () => {
      drawer.classList.remove('visible');
      document.querySelectorAll('.stage-card').forEach(c => c.classList.remove('highlighted'));
    });
  }

  // Toggle detail button
  const toggleDetailBtn = document.getElementById('btn-toggle-diagram-detail');
  if (toggleDetailBtn) {
    toggleDetailBtn.addEventListener('click', () => {
      if (drawer.classList.contains('visible')) {
        drawer.classList.remove('visible');
        toggleDetailBtn.classList.remove('active');
      } else {
        const firstCard = document.querySelector('.stage-card');
        if (firstCard) firstCard.click();
        toggleDetailBtn.classList.add('active');
      }
    });
  }

  // Fullscreen Modal Viewer
  const fullscreenBtn = document.getElementById('btn-fullscreen-diagram');
  if (fullscreenBtn) {
    fullscreenBtn.addEventListener('click', () => {
      openFullscreenModal();
    });
  }
}

/**
 * Creates and opens the fullscreen modal diagram viewer
 */
function openFullscreenModal() {
  let modal = document.getElementById('diagram-fullscreen-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'diagram-fullscreen-modal';
    modal.className = 'diagram-modal-backdrop';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.innerHTML = `
      <div class="diagram-modal-card">
        <div class="diagram-modal-header">
          <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 20px;">🛰️</span>
            <div>
              <h3 style="margin: 0; font-family: var(--font-display); font-size: 16px;">UrbanPulse Analytical Architecture & Methodology</h3>
              <p style="margin: 0; font-size: 12px; color: var(--color-text-secondary);">High-definition interactive diagram of the geospatial processing pipeline</p>
            </div>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <button id="modal-zoom-out" class="method-btn" title="Zoom Out">&minus;</button>
            <button id="modal-zoom-reset" class="method-btn" title="Reset Zoom">100%</button>
            <button id="modal-zoom-in" class="method-btn" title="Zoom In">&plus;</button>
            <button id="modal-close-btn" class="tool-btn" style="padding: 4px 10px; font-size: 18px;" aria-label="Close dialog">&times;</button>
          </div>
        </div>

        <div class="diagram-modal-body" id="modal-diagram-scroll">
          <div id="modal-zoom-canvas" style="transform-origin: top left; transition: transform 0.2s ease;">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; min-width: 1200px;">
              ${renderStageCards(PIPELINE_STAGES)}
            </div>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(modal);

    let currentZoom = 1;
    const canvas = modal.querySelector('#modal-zoom-canvas');
    const resetBtn = modal.querySelector('#modal-zoom-reset');

    modal.querySelector('#modal-zoom-in').addEventListener('click', () => {
      currentZoom = Math.min(currentZoom + 0.15, 1.8);
      canvas.style.transform = `scale(${currentZoom})`;
      resetBtn.textContent = `${Math.round(currentZoom * 100)}%`;
    });

    modal.querySelector('#modal-zoom-out').addEventListener('click', () => {
      currentZoom = Math.max(currentZoom - 0.15, 0.7);
      canvas.style.transform = `scale(${currentZoom})`;
      resetBtn.textContent = `${Math.round(currentZoom * 100)}%`;
    });

    resetBtn.addEventListener('click', () => {
      currentZoom = 1;
      canvas.style.transform = 'scale(1)';
      resetBtn.textContent = '100%';
    });

    const closeModal = () => {
      modal.style.display = 'none';
      document.body.style.overflow = '';
    };

    modal.querySelector('#modal-close-btn').addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modal.style.display !== 'none') closeModal();
    });
  }

  modal.style.display = 'flex';
  document.body.style.overflow = 'hidden';
}
