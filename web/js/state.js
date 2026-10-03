/**
 * UrbanPulse Application State (web/js/state.js)
 * Single source of truth for runtime data loaded from stats.json & meta.json.
 */

export const state = {
  currentCity: 'ahmedabad',
  currentYear: 2024,
  yearsList: [2020, 2021, 2022, 2023, 2024],
  metaData: null,
  statsData: null,
  isLayerActive: true,
  isChangeActive: false,
  isCompareMode: false,
  isFocusMode: false,
  basemapType: 'osm', // 'osm' or 'satellite'
  layerOpacity: 0.90,
  detailMode: 'detailed', // 'simple' or 'detailed'
  isPlaying: false,
  playTimer: null,
  compareYearA: 2020,
  compareYearB: 2024
};

export function getSavedDetailMode() {
  try {
    return localStorage.getItem('urbanpulse_detail_mode') || 'detailed';
  } catch {
    return 'detailed';
  }
}

export function saveDetailMode(mode) {
  state.detailMode = mode;
  try {
    localStorage.setItem('urbanpulse_detail_mode', mode);
  } catch {
    // Ignore private browsing storage restriction
  }
}
