/**
 * UrbanPulse Map Controller (web/js/map.js)
 * Manages MapLibre GL instance, raster overlays, compare swipe, basemap switching, and recentering.
 */

import { state } from './state.js';

let map = null;
let mapBefore = null;
let mapAfter = null;
let syncingMaps = false;

export function getBasemapStyle(type = 'osm') {
  return {
    version: 8,
    sources: {
      'osm-basemap': {
        type: 'raster',
        tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
        tileSize: 256,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors',
        maxzoom: 19
      },
      'satellite-basemap': {
        type: 'raster',
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
        tileSize: 256,
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and GIS Community',
        maxzoom: 19
      }
    },
    layers: [
      {
        id: 'osm-basemap-layer',
        type: 'raster',
        source: 'osm-basemap',
        minzoom: 0,
        maxzoom: 19,
        layout: { visibility: type === 'osm' ? 'visible' : 'none' }
      },
      {
        id: 'satellite-basemap-layer',
        type: 'raster',
        source: 'satellite-basemap',
        minzoom: 0,
        maxzoom: 19,
        layout: { visibility: type === 'satellite' ? 'visible' : 'none' }
      }
    ]
  };
}

export function initMap(onMapLoaded) {
  const MapLibreClass = window.maplibregl || window.maplibreGl || (typeof maplibregl !== 'undefined' ? maplibregl : null);
  if (!MapLibreClass) {
    console.error("MapLibre GL library not found in vendor.");
    return null;
  }

  map = new MapLibreClass.Map({
    container: 'map',
    style: getBasemapStyle(state.basemapType),
    center: [72.5714, 23.0225],
    zoom: 10.5,
    pitch: 0,
    attributionControl: true
  });

  map.addControl(new MapLibreClass.NavigationControl({ showCompass: false }), 'bottom-right');

  map.on('load', () => {
    if (onMapLoaded) onMapLoaded();
  });

  window.UrbanPulseMap = map;

  window.addEventListener('urbanpulse:mapresize', () => {
    if (map) {
      setTimeout(() => map.resize(), 100);
    }
  });

  return map;
}

export function setupMapLayers(metaData) {
  if (!map || !metaData || !metaData.bounds) return;

  const [west, south, east, north] = metaData.bounds;
  const coordinates = [
    [west, north], // Top-Left (NW)
    [east, north], // Top-Right (NE)
    [east, south], // Bottom-Right (SE)
    [west, south]  // Bottom-Left (SW)
  ];

  // Remove existing
  ['annual-overlay-layer', 'change-overlay-layer'].forEach(id => {
    if (map.getLayer(id)) map.removeLayer(id);
  });
  ['annual-overlay-source', 'change-overlay-source'].forEach(id => {
    if (map.getSource(id)) map.removeSource(id);
  });

  const annualPngUrl = `data/${state.currentCity}/${state.currentYear}.png`;
  const changePngUrl = `data/${state.currentCity}/change_2020_2024.png`;

  map.addSource('annual-overlay-source', {
    type: 'image',
    url: annualPngUrl,
    coordinates: coordinates
  });

  map.addLayer({
    id: 'annual-overlay-layer',
    type: 'raster',
    source: 'annual-overlay-source',
    layout: {
      visibility: (state.isLayerActive && !state.isChangeActive) ? 'visible' : 'none'
    },
    paint: {
      'raster-opacity': state.layerOpacity,
      'raster-resampling': 'nearest'
    }
  });

  map.addSource('change-overlay-source', {
    type: 'image',
    url: changePngUrl,
    coordinates: coordinates
  });

  map.addLayer({
    id: 'change-overlay-layer',
    type: 'raster',
    source: 'change-overlay-source',
    layout: {
      visibility: (state.isLayerActive && state.isChangeActive) ? 'visible' : 'none'
    },
    paint: {
      'raster-opacity': state.layerOpacity,
      'raster-resampling': 'nearest'
    }
  });

  map.fitBounds([[west, south], [east, north]], { padding: 40, duration: 800 });
}

export function updateMapYear(year) {
  state.currentYear = year;
  if (!map || !map.getSource('annual-overlay-source') || !state.metaData) return;

  const [west, south, east, north] = state.metaData.bounds;
  const coordinates = [
    [west, north],
    [east, north],
    [east, south],
    [west, south]
  ];
  const url = `data/${state.currentCity}/${year}.png`;
  map.getSource('annual-overlay-source').updateImage({
    url: url,
    coordinates: coordinates
  });
}

export function toggleLayerVisibility(visible) {
  state.isLayerActive = visible;
  if (!map) return;
  if (map.getLayer('annual-overlay-layer')) {
    map.setLayoutProperty('annual-overlay-layer', 'visibility', (visible && !state.isChangeActive) ? 'visible' : 'none');
  }
  if (map.getLayer('change-overlay-layer')) {
    map.setLayoutProperty('change-overlay-layer', 'visibility', (visible && state.isChangeActive) ? 'visible' : 'none');
  }
}

export function toggleChangeLayer(active) {
  state.isChangeActive = active;
  if (!map) return;
  if (map.getLayer('annual-overlay-layer')) {
    map.setLayoutProperty('annual-overlay-layer', 'visibility', (state.isLayerActive && !active) ? 'visible' : 'none');
  }
  if (map.getLayer('change-overlay-layer')) {
    map.setLayoutProperty('change-overlay-layer', 'visibility', (state.isLayerActive && active) ? 'visible' : 'none');
  }
}

export function setLayerOpacity(opacity) {
  state.layerOpacity = opacity;
  if (!map) return;
  if (map.getLayer('annual-overlay-layer')) {
    map.setPaintProperty('annual-overlay-layer', 'raster-opacity', opacity);
  }
  if (map.getLayer('change-overlay-layer')) {
    map.setPaintProperty('change-overlay-layer', 'raster-opacity', opacity);
  }
}

export function switchBasemap(type) {
  state.basemapType = type;
  if (!map) return;
  if (map.getLayer('osm-basemap-layer')) {
    map.setLayoutProperty('osm-basemap-layer', 'visibility', type === 'osm' ? 'visible' : 'none');
  }
  if (map.getLayer('satellite-basemap-layer')) {
    map.setLayoutProperty('satellite-basemap-layer', 'visibility', type === 'satellite' ? 'visible' : 'none');
  }
}

export function recenterMap() {
  if (!map || !state.metaData || !state.metaData.bounds) return;
  const [west, south, east, north] = state.metaData.bounds;
  map.fitBounds([[west, south], [east, north]], { padding: 40, duration: 600 });
}

export function initCompareMode(yearA, yearB) {
  state.isCompareMode = true;
  state.compareYearA = yearA;
  state.compareYearB = yearB;

  const MapLibreClass = window.maplibregl || window.maplibreGl || (typeof maplibregl !== 'undefined' ? maplibregl : null);
  const container = document.getElementById('compare-container');
  if (!container || !MapLibreClass) return;

  container.classList.add('active');

  // Update comparison tags
  const tagLeft = document.getElementById('compare-tag-left');
  const tagRight = document.getElementById('compare-tag-right');
  if (tagLeft) tagLeft.textContent = yearA;
  if (tagRight) tagRight.textContent = yearB;

  const [west, south, east, north] = state.metaData.bounds;
  const coordinates = [
    [west, north],
    [east, north],
    [east, south],
    [west, south]
  ];
  const cityBounds = [[west, south], [east, north]];

  if (!mapBefore) {
    mapBefore = new MapLibreClass.Map({
      container: 'map-before',
      style: getBasemapStyle(state.basemapType),
      bounds: cityBounds,
      fitBoundsOptions: { padding: 40 },
      attributionControl: false
    });
    mapBefore.on('load', () => {
      mapBefore.addSource('cmp-left-source', {
        type: 'image',
        url: `data/${state.currentCity}/${yearA}.png`,
        coordinates: coordinates
      });
      mapBefore.addLayer({
        id: 'cmp-left-layer',
        type: 'raster',
        source: 'cmp-left-source',
        paint: { 'raster-opacity': state.layerOpacity, 'raster-resampling': 'nearest' }
      });
      mapBefore.resize();
    });
  } else {
    mapBefore.fitBounds(cityBounds, { padding: 40, duration: 600 });
    if (mapBefore.getSource('cmp-left-source')) {
      mapBefore.getSource('cmp-left-source').updateImage({
        url: `data/${state.currentCity}/${yearA}.png`,
        coordinates: coordinates
      });
    }
    setTimeout(() => mapBefore && mapBefore.resize(), 50);
  }

  if (!mapAfter) {
    mapAfter = new MapLibreClass.Map({
      container: 'map-after',
      style: getBasemapStyle(state.basemapType),
      bounds: cityBounds,
      fitBoundsOptions: { padding: 40 },
      attributionControl: false
    });
    mapAfter.on('load', () => {
      mapAfter.addSource('cmp-right-source', {
        type: 'image',
        url: `data/${state.currentCity}/${yearB}.png`,
        coordinates: coordinates
      });
      mapAfter.addLayer({
        id: 'cmp-right-layer',
        type: 'raster',
        source: 'cmp-right-source',
        paint: { 'raster-opacity': state.layerOpacity, 'raster-resampling': 'nearest' }
      });
      mapAfter.resize();
    });
  } else {
    mapAfter.fitBounds(cityBounds, { padding: 40, duration: 600 });
    if (mapAfter.getSource('cmp-right-source')) {
      mapAfter.getSource('cmp-right-source').updateImage({
        url: `data/${state.currentCity}/${yearB}.png`,
        coordinates: coordinates
      });
    }
    setTimeout(() => mapAfter && mapAfter.resize(), 50);
  }

  // Setup clip path & swipe handling
  setupSwipeDivider();
}

function syncMove(src, dst) {
  if (syncingMaps || !src || !dst) return;
  syncingMaps = true;
  dst.jumpTo({
    center: src.getCenter(),
    zoom: src.getZoom(),
    bearing: src.getBearing(),
    pitch: src.getPitch()
  });
  syncingMaps = false;
}

let swipeListenersAttached = false;
let isSwipeDragging = false;

function setupSwipeDivider() {
  const divider = document.getElementById('swipe-divider');
  const container = document.getElementById('compare-container');
  const mapBeforeEl = document.getElementById('map-before');
  if (!divider || !container || !mapBeforeEl) return;

  function setSliderPosition(clientX) {
    const rect = container.getBoundingClientRect();
    const pos = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const pct = (pos / rect.width) * 100;
    divider.style.left = `${pct}%`;
    mapBeforeEl.style.clipPath = `polygon(0 0, ${pct}% 0, ${pct}% 100%, 0 100%)`;
  }

  // Set initial position at center (50%)
  setSliderPosition(container.getBoundingClientRect().left + container.clientWidth / 2);

  if (!swipeListenersAttached) {
    divider.addEventListener('mousedown', (e) => {
      e.preventDefault();
      isSwipeDragging = true;
      document.body.style.cursor = 'ew-resize';
    });

    window.addEventListener('mouseup', () => {
      if (isSwipeDragging) {
        isSwipeDragging = false;
        document.body.style.cursor = '';
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (isSwipeDragging) {
        e.preventDefault();
        setSliderPosition(e.clientX);
      }
    });

    divider.addEventListener('touchstart', (e) => {
      isSwipeDragging = true;
    }, { passive: true });

    window.addEventListener('touchend', () => {
      isSwipeDragging = false;
    });

    window.addEventListener('touchmove', (e) => {
      if (isSwipeDragging && e.touches[0]) {
        setSliderPosition(e.touches[0].clientX);
      }
    }, { passive: true });

    if (mapBefore && mapAfter) {
      mapBefore.on('move', () => syncMove(mapBefore, mapAfter));
      mapAfter.on('move', () => syncMove(mapAfter, mapBefore));
    }

    swipeListenersAttached = true;
  }
}

export function exitCompareMode() {
  state.isCompareMode = false;
  isSwipeDragging = false;
  document.body.style.cursor = '';
  const container = document.getElementById('compare-container');
  if (container) container.classList.remove('active');
  if (map) map.resize();
}
