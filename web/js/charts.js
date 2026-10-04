/**
 * UrbanPulse Charts Controller (web/js/charts.js)
 * Manages Chart.js instances for multi-series growth and concentric ring density profiles.
 */

let timeSeriesChart = null;
let ringDensityChart = null;

export function initCharts(statsData, currentYear) {
  if (!window.Chart || !statsData) return;

  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  const textColor = isDark ? '#94a3b8' : '#64748b';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(20, 26, 36, 0.08)';

  // 1. Time Series Chart
  const tsCanvas = document.getElementById('timeSeriesChart');
  if (tsCanvas) {
    if (timeSeriesChart) timeSeriesChart.destroy();

    const gs = statsData.growth_series || [];
    const labels = gs.map(d => d.display_year || String(d.year));
    const cleanData = gs.map(d => d.clean_builtup_km2);
    const rawData = gs.map(d => d.raw_builtup_km2);
    const normData = gs.map(d => d.norm_builtup_km2);
    const bandMin = gs.map(d => d.band_min_km2 != null ? d.band_min_km2 : Math.min(d.clean_builtup_km2, d.raw_builtup_km2, d.norm_builtup_km2));
    const bandMax = gs.map(d => d.band_max_km2 != null ? d.band_max_km2 : Math.max(d.clean_builtup_km2, d.raw_builtup_km2, d.norm_builtup_km2));
    const wc2021 = statsData.headline_2020_2024_expansion?.worldcover_2021_anchor_km2 || 393.73;

    timeSeriesChart = new window.Chart(tsCanvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Method Sensitivity Upper',
            data: bandMax,
            borderColor: 'transparent',
            backgroundColor: isDark ? 'rgba(240, 110, 91, 0.12)' : 'rgba(217, 83, 63, 0.10)',
            fill: '+1',
            pointRadius: 0
          },
          {
            label: 'Method Sensitivity Lower',
            data: bandMin,
            borderColor: 'transparent',
            backgroundColor: 'transparent',
            fill: false,
            pointRadius: 0
          },
          {
            label: 'TLS-Normalised (Main)',
            data: normData,
            borderColor: isDark ? '#f06e5b' : '#d9533f',
            borderWidth: 2.5,
            backgroundColor: isDark ? '#f06e5b' : '#d9533f',
            pointRadius: gs.map(d => d.year === currentYear ? 6 : 3),
            pointHoverRadius: 7,
            tension: 0.15
          },
          {
            label: 'Cleaned (sensitivity)',
            data: cleanData,
            borderColor: '#10b981',
            borderWidth: 1.8,
            borderDash: [4, 4],
            pointRadius: 2,
            tension: 0.15
          },
          {
            label: 'Raw Classified',
            data: rawData,
            borderColor: '#f59e0b',
            borderWidth: 1.8,
            borderDash: [2, 2],
            pointRadius: 2,
            tension: 0.15
          },
          {
            label: 'WorldCover 2021',
            data: gs.map(d => d.year === 2021 ? wc2021 : null),
            borderColor: '#a855f7',
            backgroundColor: '#a855f7',
            pointRadius: 6,
            pointStyle: 'rectRot',
            showLine: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
            callbacks: {
              label: (ctx) => {
                const item = gs[ctx.dataIndex];
                if (ctx.dataset.label === 'Raw Classified' && item && item.raw_excluded) {
                  return `${ctx.dataset.label}: ${Number(ctx.raw).toFixed(1)} km² (Excluded: ${item.raw_excluded_reason || '5-date composite containing an anomalous scene, 2021-12-05'})`;
                }
                return `${ctx.dataset.label}: ${ctx.raw != null ? Number(ctx.raw).toFixed(1) + ' km²' : 'N/A'}`;
              }
            }
        },
        scales: {
          x: { grid: { color: gridColor }, ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } } },
          y: { grid: { color: gridColor }, ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } } }
        }
      }
    });
  }

function getRingsForYear(ringsData, year) {
  if (!ringsData) return [];
  if (Array.isArray(ringsData)) {
    return ringsData.filter(r => String(r.year) === String(year));
  }
  return ringsData[String(year)] || ringsData[year] || [];
}

  // 2. Concentric Rings Chart
  const ringCanvas = document.getElementById('ringDensityChart');
  if (ringCanvas && statsData.rings) {
    if (ringDensityChart) ringDensityChart.destroy();

    const rings = getRingsForYear(statsData.rings, currentYear);
    const labels = rings.map(r => r.ring_label || `${r.ring_start_km || r.min_km}-${r.ring_end_km || r.max_km} km`);
    const densities = rings.map(r => r.builtup_pct != null ? r.builtup_pct : (r.builtup_density_pct != null ? r.builtup_density_pct : r.density_pct));

    ringDensityChart = new window.Chart(ringCanvas.getContext('2d'), {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Built-up Density %',
          data: densities,
          backgroundColor: isDark ? 'rgba(240, 110, 91, 0.75)' : 'rgba(217, 83, 63, 0.75)',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { color: textColor, font: { size: 10 } } },
          y: { min: 0, max: 100, grid: { color: gridColor }, ticks: { color: textColor, font: { family: 'JetBrains Mono', size: 10 } } }
        }
      }
    });
  }
}

export function updateChartHighlights(statsData, currentYear) {
  if (timeSeriesChart && statsData?.growth_series) {
    const gs = statsData.growth_series;
    const tlsDataset = timeSeriesChart.data.datasets.find(ds => ds.label.includes('TLS-Normalised'));
    if (tlsDataset) {
      tlsDataset.pointRadius = gs.map(d => d.year === currentYear ? 6 : 3);
      timeSeriesChart.update('none');
    }
  }

  if (ringDensityChart && statsData?.rings) {
    const rings = getRingsForYear(statsData.rings, currentYear);
    if (rings.length > 0) {
      ringDensityChart.data.labels = rings.map(r => r.ring_label || `${r.ring_start_km || r.min_km}-${r.ring_end_km || r.max_km} km`);
      ringDensityChart.data.datasets[0].data = rings.map(r => r.builtup_pct != null ? r.builtup_pct : (r.builtup_density_pct != null ? r.builtup_density_pct : r.density_pct));
      ringDensityChart.update('none');
    }
  }
}
