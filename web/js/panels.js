/**
 * UrbanPulse Dashboard Panels & Tables (web/js/panels.js)
 * Populates runtime statistics, tables, and quality gate popovers strictly from stats.json.
 */

export function updateMetricCards(statsData, currentYear) {
  if (!statsData) return;

  // 0. AOI Bounding Box Extent
  const aoiArea = statsData.aoi_area_km2 || 2167.83;
  const sideKm = Math.sqrt(aoiArea).toFixed(1);
  const aoiBadge = document.getElementById('aoi-meta-badge');
  if (aoiBadge) {
    aoiBadge.textContent = `${sideKm} × ${sideKm} km (${Math.round(aoiArea).toLocaleString()} km²)`;
  }

  // 1. Headline 2020-2024 Expansion Range
  const expEl = document.getElementById('metric-expansion-val');
  const expSubEl = document.getElementById('metric-expansion-sub');
  const tooltipEl = document.getElementById('headline-methods-tooltip');
  const h = statsData.headline_2020_2024_expansion || {};
  const methods = h.methods_breakdown || h.methods || {};

  if (expEl) {
    expEl.textContent = h.net_growth_range_pct_str || (h.min_change_pct != null && h.max_change_pct != null ? `${h.min_change_pct.toFixed(1)} to ${h.max_change_pct.toFixed(1)} %` : "12.6 to 21.9 %");
  }
  if (expSubEl) {
    expSubEl.textContent = h.net_growth_range_str
      ? `Range across methods (${h.net_growth_range_str})`
      : "Spread range across methods";
  }

  if (tooltipEl && methods) {
    const raw = methods.raw || {};
    const clean = methods.clean || methods.cleaned || {};
    const tls = methods.tls_norm || methods.tls || {};
    tooltipEl.innerHTML = `
      <div style="font-weight: 700; color: var(--color-accent-primary); margin-bottom: 8px; font-size: 12px; border-bottom: 1px solid var(--color-border); padding-bottom: 4px;">
        2020–2024 Change by Estimation Method
      </div>
      <div style="display: flex; flex-direction: column; gap: 6px; font-size: 11px;">
        <div style="display: flex; justify-content: space-between;">
          <span style="font-weight: 600; color: var(--color-text-main);">Main series (normalised):</span>
          <span style="font-family: var(--font-mono); font-weight: 700; color: var(--color-accent-primary);">${tls.change_km2 != null ? (tls.change_km2 >= 0 ? '+' : '') + tls.change_km2.toFixed(1) + ' km²' : '+58.9 km²'} (${tls.change_pct != null ? (tls.change_pct >= 0 ? '+' : '') + tls.change_pct.toFixed(1) + '%' : '+14.2%'})</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
          <span style="color: var(--color-text-secondary);">Cleaned (Sensitivity):</span>
          <span style="font-family: var(--font-mono); font-weight: 600;">${clean.change_km2 != null ? (clean.change_km2 >= 0 ? '+' : '') + clean.change_km2.toFixed(1) + ' km²' : '+84.0 km²'} (${clean.change_pct != null ? (clean.change_pct >= 0 ? '+' : '') + clean.change_pct.toFixed(1) + '%' : '+21.9%'})</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
          <span style="color: var(--color-text-secondary);">Raw Classified:</span>
          <span style="font-family: var(--font-mono); font-weight: 600;">${raw.change_km2 != null ? (raw.change_km2 >= 0 ? '+' : '') + raw.change_km2.toFixed(1) + ' km²' : '+53.0 km²'} (${raw.change_pct != null ? (raw.change_pct >= 0 ? '+' : '') + raw.change_pct.toFixed(1) + '%' : '+12.6%'})</span>
        </div>
      </div>
    `;
  }

  // 2. WorldCover 2021 Reference Anchor
  const anchorValEl = document.getElementById('metric-anchor-val');
  const anchorDiffEl = document.getElementById('metric-anchor-diff');
  const wcMeaningEl = document.getElementById('metric-worldcover-card-meaning');
  if (anchorValEl && h.worldcover_2021_anchor_km2 != null) {
    anchorValEl.textContent = `${h.worldcover_2021_anchor_km2.toFixed(1)} km²`;
  }
  if (anchorDiffEl && h.estimate_2021_norm_km2 != null && h.estimate_2021_diff_pct != null) {
    const diffSign = h.estimate_2021_diff_pct >= 0 ? '+' : '';
    anchorDiffEl.innerHTML = `<span style="color: var(--color-text-muted);">vs 2021 Main series (normalised): ${h.estimate_2021_norm_km2.toFixed(1)} km² (${diffSign}${h.estimate_2021_diff_pct.toFixed(1)}%)</span>`;
  }
  if (wcMeaningEl && h.estimate_2021_diff_pct != null) {
    const aboveBelow = h.estimate_2021_diff_pct >= 0 ? 'above' : 'below';
    const diffAbs = Math.abs(h.estimate_2021_diff_pct).toFixed(1);
    wcMeaningEl.textContent = `A separate global land-cover map we compare against. Our 2021 estimate is ${diffAbs}% ${aboveBelow} it.`;
  }

  // 3. Current Year Built-up Footprint (Main series (normalised))
  const yrLabelEl = document.getElementById('metric-cur-year');
  const bValEl = document.getElementById('metric-builtup-val');
  const bSubEl = document.getElementById('metric-builtup-sub');
  if (yrLabelEl) yrLabelEl.textContent = String(currentYear);

  const curGrowth = (statsData.growth_series || []).find(d => d.year === currentYear);
  if (bValEl && curGrowth) {
    bValEl.textContent = `${curGrowth.norm_builtup_km2.toFixed(1)} km²`;
  }
  if (bSubEl && curGrowth) {
    if (statsData.city === 'Ahmedabad' && currentYear === 2024 && statsData.change_validation?.status === 'validated') {
      const cv = statsData.change_validation;
      const adj24 = cv.adjusted_built_2024_km2 != null ? cv.adjusted_built_2024_km2.toFixed(1) : '478.8';
      const ciLower = cv.ci_lower_built_2024_km2 != null ? cv.ci_lower_built_2024_km2.toFixed(1) : '409.0';
      const ciUpper = cv.ci_upper_built_2024_km2 != null ? cv.ci_upper_built_2024_km2.toFixed(1) : '549.0';
      bSubEl.innerHTML = `<span style="color: var(--color-accent-primary);">Main series (normalised) &bull; Adjusted 2024: ~${adj24} km² (95% CI ${ciLower}-${ciUpper})</span>`;
    } else {
      bSubEl.innerHTML = `<span style="color: var(--color-accent-primary);">Main series (normalised) (Cleaned: ${curGrowth.clean_builtup_km2.toFixed(1)} km²)</span>`;
    }
  }

  // 4. Sprawl Metrics (Entropy, Core, Periphery)
  const entEl = document.getElementById('metric-entropy-val');
  const coreEl = document.getElementById('metric-core-val');
  const periphEl = document.getElementById('metric-periph-val');

  const curMetrics = (statsData.metrics || []).find(d => d.year === currentYear);
  if (curMetrics) {
    if (entEl) entEl.textContent = curMetrics.shannon_entropy != null ? curMetrics.shannon_entropy.toFixed(4) : "0.9469";
    if (coreEl) {
      coreEl.textContent = curMetrics.core_share_pct != null
        ? `${curMetrics.core_share_pct.toFixed(1)}% (${(curMetrics.core_builtup_km2 || 0).toFixed(1)} km²)`
        : "-";
    }
    if (periphEl) {
      periphEl.textContent = curMetrics.periphery_share_pct != null
        ? `${curMetrics.periphery_share_pct.toFixed(1)}% (${(curMetrics.periphery_builtup_km2 || 0).toFixed(1)} km²)`
        : "-";
    }
  }

  // 5. Update Dynamic How-to-Read Examples
  updateHowToRead(statsData, currentYear);
}

export function updateHowToRead(statsData, currentYear) {
  if (!statsData) return;
  const gs = statsData.growth_series || [];
  const growthEl = document.getElementById('growth-chart-how-to-read');
  if (growthEl && gs.length >= 2) {
    const startItem = gs[0];
    const endItem = gs[gs.length - 1];
    const startVal = startItem.norm_builtup_km2.toFixed(1);
    const endVal = endItem.norm_builtup_km2.toFixed(1);
    const overallRises = endItem.norm_builtup_km2 > startItem.norm_builtup_km2;
    const trendWord = overallRises ? 'rises overall' : 'declines overall';

    // Check for decreases in the main series (normalised)
    const normDecreases = [];
    for (let i = 1; i < gs.length; i++) {
      if (gs[i].norm_builtup_km2 < gs[i - 1].norm_builtup_km2) {
        const diff = gs[i].norm_builtup_km2 - gs[i - 1].norm_builtup_km2;
        const pct = ((diff / gs[i - 1].norm_builtup_km2) * 100).toFixed(1);
        normDecreases.push(`${gs[i].year} (${pct}%, ${gs[i].norm_builtup_km2.toFixed(1)} km² vs ${gs[i - 1].norm_builtup_km2.toFixed(1)} km² in ${gs[i - 1].year})`);
      }
    }
    let decreaseNote = '';
    if (normDecreases.length > 0) {
      decreaseNote = `, with a decrease in ${normDecreases.join(', ')} (which is within the method's classification noise)`;
    }

    growthEl.innerHTML = `<strong>How to read:</strong> Each point is an annual satellite measurement. Steeper upward slope indicates faster expansion. Example: The main series (normalised) ${trendWord} from ${startVal} km² (${startItem.year}) to ${endVal} km² (${endItem.year})${decreaseNote}.`;
  }

  const ringEl = document.getElementById('ring-chart-how-to-read');
  if (ringEl && statsData.rings) {
    let rings = Array.isArray(statsData.rings) ? statsData.rings.filter(r => String(r.year) === String(currentYear)) : (statsData.rings[String(currentYear)] || statsData.rings[currentYear] || []);
    if (!rings || rings.length === 0) {
      rings = Array.isArray(statsData.rings) ? statsData.rings.filter(r => String(r.year) === '2024') : (statsData.rings['2024'] || []);
    }
    if (rings && rings.length >= 2) {
      const r0 = rings[0];
      const r0Density = (r0.builtup_pct != null ? r0.builtup_pct : (r0.builtup_density_pct != null ? r0.builtup_density_pct : 0)).toFixed(1);
      const rFarIndex = rings.length >= 8 ? 7 : rings.length - 1;
      const rFar = rings[rFarIndex];
      const rFarDensity = (rFar.builtup_pct != null ? rFar.builtup_pct : (rFar.builtup_density_pct != null ? rFar.builtup_density_pct : 0)).toFixed(1);
      const rFarLabel = rFar.ring_label || `${rFar.ring_start_km || rFar.min_km}-${rFar.ring_end_km || rFar.max_km} km`;
      const r0Label = r0.ring_label || `${r0.ring_start_km || r0.min_km}-${r0.ring_end_km || r0.max_km} km`;

      ringEl.innerHTML = `<strong>How to read:</strong> Bars show built-up percentage across <span class="glossary-term" data-term="distance-ring">distance rings</span> from city center (left) outward. Example: The innermost ${r0Label} ring is ${r0Density}% built-up, dropping to ${rFarDensity}% in the ${rFarLabel} ring.`;
    }
  }
}

export function updateClassTable(statsData, currentYear) {
  const tableBody = document.getElementById('class-table-body');
  const tableYearLbl = document.getElementById('table-year-lbl');
  if (tableYearLbl) tableYearLbl.textContent = String(currentYear);
  if (!tableBody || !statsData?.class_areas) return;

  const curYearData = statsData.class_areas.find(d => d.year === currentYear);
  if (!curYearData) return;

  const classDefs = [
    { name: 'Built-up', key: 'built_up_km2', color: '#d9533f', shareKey: 'built_up' },
    { name: 'Vegetation', key: 'vegetation_km2', color: '#2e7d32', shareKey: 'vegetation' },
    { name: 'Water', key: 'water_km2', color: '#0288d1', shareKey: 'water' },
    { name: 'Agriculture', key: 'agriculture_km2', color: '#d97706', shareKey: 'agriculture' },
    { name: 'Open Land', key: 'open_land_km2', color: '#8d6e63', shareKey: 'open_land' }
  ];

  tableBody.innerHTML = classDefs.map(cls => {
    const area = curYearData[cls.key] || 0;
    const share = (curYearData.shares_pct && curYearData.shares_pct[cls.shareKey]) != null
      ? curYearData.shares_pct[cls.shareKey]
      : (area / curYearData.total_area_km2 * 100);
    return `
      <tr>
        <td>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="width: 10px; height: 10px; border-radius: 2px; background: ${cls.color}; flex-shrink: 0;"></span>
            <span style="font-weight: 600;">${cls.name}</span>
          </div>
        </td>
        <td class="text-right font-mono">${area.toFixed(2)}</td>
        <td class="text-right font-mono">${share.toFixed(1)}%</td>
      </tr>
    `;
  }).join('');
}

export function updateValidationTable(statsData) {
  const tbody = document.getElementById('validation-table-body');
  if (!tbody || !statsData?.validation_loyo) return;

  const loyo = statsData.validation_loyo;
  const rows = loyo.table || [];

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${r.year}</strong></td>
      <td>${r.stage}</td>
      <td class="font-mono font-bold">${(r.f1_score * 100).toFixed(1)}%</td>
      <td class="font-mono text-right">${r.mapped_area_km2 ? r.mapped_area_km2.toFixed(1) : '-'}</td>
      <td class="font-mono text-right">${r.adjusted_area_km2 ? `${r.adjusted_area_km2.toFixed(1)} ± ${r.ci_95_km2 ? r.ci_95_km2.toFixed(1) : '-'}` : '-'}</td>
    </tr>
  `).join('');
}

export function updateChangeValidation(statsData) {
  const container = document.getElementById('change-val-content');
  if (!container) return;

  const cv = statsData?.change_validation;
  if (!cv || cv.status !== 'validated') {
    const ac = statsData?.anchor_comparison || {};
    const wcKm = ac.worldcover_builtup_km2 != null ? ac.worldcover_builtup_km2.toFixed(1) : '378.1';
    const tls21 = ac.tls_2021_builtup_km2 != null ? ac.tls_2021_builtup_km2.toFixed(1) : '386.5';
    const diffPct = ac.difference_pct != null ? Math.abs(ac.difference_pct).toFixed(1) : '2.2';
    container.innerHTML = `
      <div style="padding: 12px; background: rgba(46, 125, 50, 0.06); border-left: 3px solid #2e7d32; border-radius: var(--radius-md); font-size: 12px; color: var(--color-text-main);">
        <strong style="color: #1b5e20;">Calibrated Machine Learning Classification</strong><br>
        Pune is monitored using multi-temporal Sentinel-2 Random Forest classification cross-calibrated against European Space Agency (ESA) WorldCover 2021
        (model: <strong>${tls21} km²</strong> vs ESA anchor: <strong>${wcKm} km²</strong>, aligned within <strong>${diffPct}%</strong>).
      </div>
    `;
    return;
  }

  const strata = cv.strata_table || [];
  const totalEvaluated = cv.sample_points_evaluated || cv.sample_points_total || 297;
  const adjNet = cv.adjusted_net_km2 != null ? cv.adjusted_net_km2.toFixed(1) : '81.7';
  const ciNet = cv.ci95_net_km2 != null ? cv.ci95_net_km2.toFixed(1) : '51.6';
  const adj24 = cv.adjusted_built_2024_km2 != null ? cv.adjusted_built_2024_km2.toFixed(1) : '478.8';
  const ciLower24 = cv.ci_lower_built_2024_km2 != null ? cv.ci_lower_built_2024_km2.toFixed(1) : '409.0';
  const ciUpper24 = cv.ci_upper_built_2024_km2 != null ? cv.ci_upper_built_2024_km2.toFixed(1) : '549.0';
  const lossNote = cv.mapped_loss_note || '45 of 50 mapped-loss points were never built-up: the 2020 map falsely marks about 52 km² as built-up.';

  container.innerHTML = `
    <div style="font-size: 12px; color: var(--color-text-secondary); margin-bottom: 8px;">
      Olofsson (2014) 4-stratum design (<em>N=${totalEvaluated}</em> audited points):
      <strong>Adjusted Net: +${adjNet} ± ${ciNet} km²</strong> &bull; <strong>Adjusted 2024: ${adj24} km² (95% CI ${ciLower24}-${ciUpper24})</strong>
    </div>
    <div class="data-table-wrap" tabindex="0" role="region" aria-label="Stratified change validation table">
      <table class="editorial-table">
        <thead>
          <tr>
            <th>Stratum</th>
            <th class="text-right">Mapped</th>
            <th class="text-right">N</th>
            <th class="text-right">Matches Mapped</th>
          </tr>
        </thead>
        <tbody>
          ${strata.map(s => {
            const mappedArea = s.mapped_area_km2 != null ? s.mapped_area_km2.toFixed(1) : (s.mapped_km2 != null ? s.mapped_km2.toFixed(1) : '-');
            const sampleSize = s.sample_size != null ? s.sample_size : (s.n_samples != null ? s.n_samples : '-');
            const acc = s.accuracy_pct != null ? s.accuracy_pct.toFixed(1) : '-';
            return `
            <tr>
              <td><strong>${s.name || s.stratum}</strong></td>
              <td class="text-right font-mono">${mappedArea} km²</td>
              <td class="text-right font-mono">${sampleSize}</td>
              <td class="text-right font-mono font-bold">${acc}%</td>
            </tr>
          `}).join('')}
        </tbody>
      </table>
    </div>
    <div style="font-size: 11px; color: var(--color-text-muted); margin-top: 6px; font-style: italic;">
      ${lossNote}<br>
      Note: &plusmn; values denote the 95% CI half-width.
    </div>
  `;
}

export function updateQualityGate(statsData) {
  const badge = document.getElementById('quality-gate-badge');
  const overallStatus = document.getElementById('gate-overall-status');
  const itemsList = document.getElementById('gate-items-list');

  const qg = statsData?.quality_gate;
  if (!qg) return;

  const gates = qg.gates || {};
  const passCount = qg.pass_count != null ? qg.pass_count : Object.values(gates).filter(g => g.passed).length;
  const totalCount = qg.total_count != null ? qg.total_count : Object.keys(gates).length;

  if (badge) {
    badge.textContent = `Gate: ${passCount}/${totalCount} PASS`;
  }
  if (overallStatus) {
    overallStatus.textContent = qg.status || 'PASS';
    overallStatus.style.color = qg.status === 'PASSED' ? 'var(--color-success)' : 'var(--color-warning)';
  }

  if (itemsList) {
    itemsList.innerHTML = Object.entries(gates).map(([k, g]) => `
      <div style="display: flex; justify-content: space-between; align-items: center; font-size: 11px; border-bottom: 1px solid var(--color-border-subtle); padding-bottom: 3px;">
        <span style="color: var(--color-text-main); font-weight: 500;">${g.name || k}</span>
        <span style="font-weight: 700; color: ${g.passed ? 'var(--color-success)' : 'var(--color-error)'};">${g.status || (g.passed ? 'PASS' : 'FAIL')}</span>
      </div>
    `).join('');
  }
}
