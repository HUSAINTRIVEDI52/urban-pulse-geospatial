/**
 * UrbanPulse Main Application Bootstrap (web/js/main.js)
 * Coordinates data ingestion, view modules, map interactions, and controls.
 */

import { state } from './state.js';
import { initTheme } from './theme.js';
import { initNavigation } from './navigation.js';
import { 
  initMap, 
  setupMapLayers, 
  updateMapYear, 
  toggleLayerVisibility, 
  toggleChangeLayer, 
  recenterMap,
  initCompareMode,
  exitCompareMode
} from './map.js';
import { initCharts, updateChartHighlights } from './charts.js';
import { 
  updateMetricCards, 
  updateClassTable, 
  updateQualityGate 
} from './panels.js';
import { initGlossary } from './glossary.js';
import { initTour } from './tour.js';
import { initDepthToggle } from './depth.js';
import { initContentPages, updateKeyFindings } from './content.js';

// Base path helper (supports GitHub Pages subpaths)
const DATA_BASE = 'data';

async function fetchJsonVerified(url, desc) {
  try {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} (${res.statusText}) when fetching ${desc} from ${url}`);
    }
    return await res.json();
  } catch (err) {
    showError(`Fetch ${desc}`, err);
    throw err;
  }
}

function showError(context, err) {
  console.error(`[UrbanPulse Error - ${context}]:`, err);
  const banner = document.getElementById('error-banner');
  const title = document.getElementById('error-title');
  const content = document.getElementById('error-content');
  if (banner && title && content) {
    title.textContent = `⚠️ Error in ${context}`;
    content.textContent = err.message || String(err);
    banner.classList.add('visible');
  }
}

function clearError() {
  const banner = document.getElementById('error-banner');
  if (banner) banner.classList.remove('visible');
}

async function loadCityData(city) {
  clearError();
  state.currentCity = city;

  // Sync city selector dropdown
  const citySelect = document.getElementById('city-select');
  if (citySelect && citySelect.value !== city) {
    citySelect.value = city;
  }

  // Update URL search parameter ?city= without full reload
  try {
    const url = new URL(window.location.href);
    if (url.searchParams.get('city') !== city) {
      url.searchParams.set('city', city);
      window.history.replaceState({ city }, '', url.toString());
    }
  } catch (e) {
    // History API blocked or iframe
  }

  try {
    state.metaData = await fetchJsonVerified(`${DATA_BASE}/${city}/meta.json`, `${city} metadata`);
    state.statsData = await fetchJsonVerified(`${DATA_BASE}/${city}/stats.json`, `${city} statistics`);

    // Available years
    if (state.metaData.years && state.metaData.years.length > 0) {
      state.yearsList = state.metaData.years;
    } else if (state.statsData.years && state.statsData.years.length > 0) {
      state.yearsList = state.statsData.years;
    } else {
      state.yearsList = [2020, 2021, 2022, 2023, 2024];
    }

    // Default to last year
    state.currentYear = state.yearsList[state.yearsList.length - 1];

    // Setup map layers
    setupMapLayers(state.metaData);

    // Setup compare dropdowns
    setupCompareDropdowns();

    // If compare mode is active, refresh compare map for the newly selected city
    if (state.isCompareMode) {
      const selA = document.getElementById('compare-year-a');
      const selB = document.getElementById('compare-year-b');
      const yrA = selA ? parseInt(selA.value, 10) : state.yearsList[0];
      const yrB = selB ? parseInt(selB.value, 10) : state.yearsList[state.yearsList.length - 1];
      initCompareMode(yrA, yrB);
    }

    // Setup slider ticks
    setupTimelineTicks();

    // Update displays
    updateAllViews();

  } catch (err) {
    showError(`Load City Data (${city})`, err);
  }
}

function setupTimelineTicks() {
  const ticksContainer = document.getElementById('timeline-ticks');
  const slider = document.getElementById('year-slider');
  if (!ticksContainer || !slider) return;

  slider.min = 0;
  slider.max = state.yearsList.length - 1;
  slider.value = state.yearsList.indexOf(state.currentYear);

  ticksContainer.innerHTML = state.yearsList.map((yr, idx) => `
    <span class="${yr === state.currentYear ? 'tick-active' : ''}" style="cursor: pointer;" data-index="${idx}">${yr}</span>
  `).join('');

  ticksContainer.querySelectorAll('span').forEach(el => {
    el.addEventListener('click', () => {
      const idx = parseInt(el.getAttribute('data-index'), 10);
      slider.value = idx;
      onYearIndexChange(idx);
    });
  });
}

function setupCompareDropdowns() {
  const selA = document.getElementById('compare-year-a');
  const selB = document.getElementById('compare-year-b');
  if (!selA || !selB) return;

  const options = state.yearsList.map(y => `<option value="${y}">${y}</option>`).join('');
  selA.innerHTML = options;
  selB.innerHTML = options;

  selA.value = state.yearsList[0];
  selB.value = state.yearsList[state.yearsList.length - 1];
}

function onYearIndexChange(idx) {
  const year = state.yearsList[idx];
  state.currentYear = year;

  const yearDisplay = document.getElementById('year-display');
  const focusYearDisplay = document.getElementById('focus-year-display');
  if (yearDisplay) yearDisplay.textContent = year;
  if (focusYearDisplay) focusYearDisplay.textContent = year;

  // Update tick classes
  document.querySelectorAll('#timeline-ticks span').forEach((el, i) => {
    if (i === idx) el.classList.add('tick-active');
    else el.classList.remove('tick-active');
  });

  // Update Map
  updateMapYear(year);

  // Update panels and charts
  updateMetricCards(state.statsData, year);
  updateClassTable(state.statsData, year);
  updateChartHighlights(state.statsData, year);
}

function updateAllViews() {
  const year = state.currentYear;
  const idx = state.yearsList.indexOf(year);

  const yearDisplay = document.getElementById('year-display');
  const focusYearDisplay = document.getElementById('focus-year-display');
  const slider = document.getElementById('year-slider');
  if (yearDisplay) yearDisplay.textContent = year;
  if (focusYearDisplay) focusYearDisplay.textContent = year;
  if (slider) slider.value = idx >= 0 ? idx : state.yearsList.length - 1;

  // Analysis note
  const noteEl = document.getElementById('analysis-note-text');
  if (noteEl && state.statsData?.analysis_window?.excluded_years_note) {
    noteEl.textContent = state.statsData.analysis_window.excluded_years_note;
  }

  // City metadata badge & header title
  const cityDisplay = state.currentCity === 'ahmedabad' ? 'Ahmedabad' : (state.currentCity === 'pune' ? 'Pune' : state.currentCity);
  const cityTitleEl = document.getElementById('dashboard-city-title');
  if (cityTitleEl) {
    cityTitleEl.textContent = `${cityDisplay} Growth Analytics`;
  }
  document.title = `UrbanPulse | ${cityDisplay} Metropolitan Satellite Observatory`;

  const aoiBadge = document.getElementById('aoi-meta-badge');
  if (aoiBadge && state.statsData?.aoi_area_km2) {
    aoiBadge.textContent = `${state.statsData.aoi_area_km2.toFixed(1)} km² AOI`;
  }

  // Render cards & charts
  updateMetricCards(state.statsData, year);
  updateClassTable(state.statsData, year);
  updateQualityGate(state.statsData);
  initCharts(state.statsData, year);

  // Render Legend Items
  renderLegendItems();

  // Update Key Findings page
  updateKeyFindings(state.statsData, state.metaData, state.currentCity);

  // Update Download data link
  const downloadDataBtn = document.getElementById('download-data-btn');
  if (downloadDataBtn) {
    downloadDataBtn.href = `data/${state.currentCity}/stats.json`;
    downloadDataBtn.setAttribute('download', `${state.currentCity}_urbanpulse_stats.json`);
  }
}

function renderLegendItems() {
  const container = document.getElementById('legend-items-container');
  if (!container) return;

  const classDefs = [
    { name: 'Built-up', color: '#d9533f' },
    { name: 'Vegetation', color: '#2e7d32' },
    { name: 'Water', color: '#0288d1' },
    { name: 'Agriculture', color: '#d97706' },
    { name: 'Open Land', color: '#8d6e63' }
  ];

  container.innerHTML = classDefs.map(c => `
    <div class="legend-item">
      <div style="display: flex; align-items: center; gap: 6px;">
        <span class="legend-patch" style="background: ${c.color};"></span>
        <span>${c.name}</span>
      </div>
    </div>
  `).join('');
}

// Play / Pause Animation
function togglePlayPause() {
  const playIcon = document.getElementById('play-icon');
  const pauseIcon = document.getElementById('pause-icon');

  if (state.isPlaying) {
    clearInterval(state.playTimer);
    state.isPlaying = false;
    if (playIcon) playIcon.style.display = 'block';
    if (pauseIcon) pauseIcon.style.display = 'none';
  } else {
    state.isPlaying = true;
    if (playIcon) playIcon.style.display = 'none';
    if (pauseIcon) pauseIcon.style.display = 'block';

    state.playTimer = setInterval(() => {
      let nextIdx = state.yearsList.indexOf(state.currentYear) + 1;
      if (nextIdx >= state.yearsList.length) nextIdx = 0;
      const slider = document.getElementById('year-slider');
      if (slider) slider.value = nextIdx;
      onYearIndexChange(nextIdx);
    }, 1500);
  }
}

// Event Listeners Wiring
function setupEventListeners() {
  // City Select
  const citySelect = document.getElementById('city-select');
  if (citySelect) {
    citySelect.value = state.currentCity;
    citySelect.addEventListener('change', (e) => {
      loadCityData(e.target.value);
    });
  }

  // Year Slider
  const yearSlider = document.getElementById('year-slider');
  if (yearSlider) {
    yearSlider.addEventListener('input', (e) => {
      onYearIndexChange(parseInt(e.target.value, 10));
    });
  }

  // Play button
  const playBtn = document.getElementById('play-pause-btn');
  if (playBtn) playBtn.addEventListener('click', togglePlayPause);

  // Overlay toggle
  const toggleOverlayBtn = document.getElementById('toggle-overlay-btn');
  if (toggleOverlayBtn) {
    toggleOverlayBtn.addEventListener('click', () => {
      const active = !state.isLayerActive;
      toggleLayerVisibility(active);
      toggleOverlayBtn.classList.toggle('active', active);
      toggleOverlayBtn.setAttribute('aria-pressed', String(active));
    });
  }

  // Change Layer toggle
  const toggleChangeBtn = document.getElementById('toggle-change-btn');
  if (toggleChangeBtn) {
    toggleChangeBtn.addEventListener('click', () => {
      const active = !state.isChangeActive;
      toggleChangeLayer(active);
      toggleChangeBtn.classList.toggle('active', active);
      toggleChangeBtn.setAttribute('aria-pressed', String(active));
    });
  }

  // Compare Mode
  const toggleCompareBtn = document.getElementById('toggle-compare-btn');
  const closeCompareBtn = document.getElementById('close-compare-btn');
  const compareBar = document.getElementById('compare-bar');
  const selA = document.getElementById('compare-year-a');
  const selB = document.getElementById('compare-year-b');

  if (toggleCompareBtn) {
    toggleCompareBtn.addEventListener('click', () => {
      const active = !state.isCompareMode;
      if (active) {
        initCompareMode(parseInt(selA.value, 10), parseInt(selB.value, 10));
        if (compareBar) compareBar.classList.add('active');
      } else {
        exitCompareMode();
        if (compareBar) compareBar.classList.remove('active');
      }
      toggleCompareBtn.classList.toggle('active', active);
      toggleCompareBtn.setAttribute('aria-pressed', String(active));
    });
  }

  if (closeCompareBtn) {
    closeCompareBtn.addEventListener('click', () => {
      exitCompareMode();
      if (compareBar) compareBar.classList.remove('active');
      if (toggleCompareBtn) {
        toggleCompareBtn.classList.remove('active');
        toggleCompareBtn.setAttribute('aria-pressed', 'false');
      }
    });
  }

  if (selA && selB) {
    selA.addEventListener('change', () => {
      if (state.isCompareMode) initCompareMode(parseInt(selA.value, 10), parseInt(selB.value, 10));
    });
    selB.addEventListener('change', () => {
      if (state.isCompareMode) initCompareMode(parseInt(selA.value, 10), parseInt(selB.value, 10));
    });
  }

  // Recenter button
  const recenterBtn = document.getElementById('recenter-btn');
  if (recenterBtn) recenterBtn.addEventListener('click', recenterMap);

  // Focus Mode
  const toggleFocusBtn = document.getElementById('toggle-focus-btn');
  const focusRestoreBar = document.getElementById('focus-restore-bar');
  const restoreFocusBtn = document.getElementById('restore-focus-btn');
  const dashboard = document.getElementById('dashboard');

  function enterFocusMode() {
    state.isFocusMode = true;
    if (dashboard) dashboard.style.display = 'none';
    if (focusRestoreBar) focusRestoreBar.classList.add('active');
    if (toggleFocusBtn) toggleFocusBtn.classList.add('active');
    window.dispatchEvent(new CustomEvent('urbanpulse:mapresize'));
  }

  function exitFocusMode() {
    state.isFocusMode = false;
    if (dashboard) dashboard.style.display = '';
    if (focusRestoreBar) focusRestoreBar.classList.remove('active');
    if (toggleFocusBtn) toggleFocusBtn.classList.remove('active');
    window.dispatchEvent(new CustomEvent('urbanpulse:mapresize'));
  }

  if (toggleFocusBtn) toggleFocusBtn.addEventListener('click', enterFocusMode);
  if (restoreFocusBtn) restoreFocusBtn.addEventListener('click', exitFocusMode);
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.isFocusMode) exitFocusMode();
  });

  // Mobile Bottom Sheet Drawer Toggle
  const dashToggle = document.getElementById('dashToggle');
  const closeDashBtn = document.getElementById('close-dashboard-btn');
  const sheetHandleBar = document.getElementById('sheet-handle-bar');

  function toggleBottomSheet() {
    if (!dashboard) return;
    const isExpanded = dashboard.classList.contains('expanded');
    dashboard.classList.toggle('expanded', !isExpanded);
    if (dashToggle) dashToggle.setAttribute('aria-expanded', String(!isExpanded));
  }

  if (dashToggle) dashToggle.addEventListener('click', toggleBottomSheet);
  if (closeDashBtn) closeDashBtn.addEventListener('click', toggleBottomSheet);
  if (sheetHandleBar) sheetHandleBar.addEventListener('click', toggleBottomSheet);

  // Error Close
  const errClose = document.getElementById('error-close-btn');
  if (errClose) errClose.addEventListener('click', clearError);

  // Theme changes update charts
  window.addEventListener('urbanpulse:themechange', () => {
    if (state.statsData) {
      initCharts(state.statsData, state.currentYear);
    }
  });
}

// Bootstrap
document.addEventListener('DOMContentLoaded', () => {
  window.UrbanPulseState = state;
  initTheme();
  initNavigation();
  initDepthToggle();
  initGlossary();
  initContentPages();
  setupEventListeners();

  // Support back/forward navigation for ?city=
  window.addEventListener('popstate', (e) => {
    const params = new URLSearchParams(window.location.search);
    const city = params.get('city') || (e.state && e.state.city) || 'ahmedabad';
    if (city !== state.currentCity) {
      loadCityData(city);
    }
  });

  const urlParams = new URLSearchParams(window.location.search);
  const initialCity = urlParams.get('city') || 'ahmedabad';

  initMap(() => {
    loadCityData(initialCity);
    initTour();
  });
});
