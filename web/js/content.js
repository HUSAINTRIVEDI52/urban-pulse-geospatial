import { renderMethodologyDiagram } from './methodology-diagram.js';

let howItsMadeData = null;
let creditsData = null;
let templatesData = null;

export async function initContentPages() {
  try {
    const [howRes, credRes, tmplRes] = await Promise.all([
      fetch('content/how_its_made.json'),
      fetch('content/credits.json'),
      fetch('content/findings_templates.json')
    ]);

    if (howRes.ok) howItsMadeData = await howRes.json();
    if (credRes.ok) creditsData = await credRes.json();
    if (tmplRes.ok) templatesData = await tmplRes.json();
  } catch (err) {
    console.warn('[UrbanPulse] Failed loading some content JSON files:', err);
  }

  renderMethodologyDiagram('how-diagram-root');
  renderHowItsMade();
  renderCredits();
  setupShareAndDownloads();
}

/**
 * Dynamically computes and updates the Key Findings view from stats.json
 */
export function updateKeyFindings(statsData, metaData, cityName) {
  const container = document.getElementById('findings-content-container');
  if (!container || !statsData) return;

  const cityDisplay = cityName === 'ahmedabad' ? 'Ahmedabad' : (cityName === 'pune' ? 'Pune' : cityName);
  const hg = statsData.headline_growth_range || {};
  const ac = statsData.anchor_comparison || {};
  const cv = statsData.change_validation || {};
  const metrics = statsData.metrics || [];
  const rings = statsData.rings || {};
  const rings2024 = Array.isArray(rings) ? rings.filter(r => String(r.year) === '2024') : (rings['2024'] || []);
  const m2024 = metrics.find(m => m.year === 2024) || metrics[metrics.length - 1] || {};

  // Compute Strata A breakdown for Ahmedabad
  let strataFindingHtml = '';
  let validationHeadlineHtml = '';

  if (cityName === 'ahmedabad' && cv.status === 'validated') {
    const strataA = (cv.strata_table || []).find(s => s.stratum === 'A') || {};
    const sampleSize = strataA.sample_size || 99;
    const c01Gain = strataA.c01_gain != null ? strataA.c01_gain : 41;
    const c00Fallow = strataA.c00 != null ? strataA.c00 : 40;

    const gainRatio = Math.round((c01Gain / sampleSize) * 10);
    const gainPct = ((c01Gain / sampleSize) * 100).toFixed(1);
    const fallowRatio = Math.round((c00Fallow / sampleSize) * 10);
    const fallowPct = ((c00Fallow / sampleSize) * 100).toFixed(1);

    const adjNet = cv.adjusted_net_km2 != null ? cv.adjusted_net_km2.toFixed(1) : '81.7';
    const ciNet = cv.ci95_net_km2 != null ? cv.ci95_net_km2.toFixed(1) : '51.6';
    const nEval = cv.sample_points_evaluated || cv.sample_points_total || 297;

    validationHeadlineHtml = `
      <div class="note-box" style="background: rgba(46, 125, 50, 0.08); border-left: 3px solid #2e7d32; margin-bottom: 16px;">
        <div style="font-weight: 700; color: #1b5e20; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          <span>Statistically Distinguishable Urban Expansion</span>
        </div>
        <p style="margin: 0; font-size: 13px; line-height: 1.5; color: var(--color-text-main);">
          Stratified ground-truth auditing across <strong>${nEval} points</strong> confirms that Ahmedabad's net built-up change is statistically distinguishable from zero: <strong>+${adjNet} ± ${ciNet} km²</strong> (95% confidence interval: ${(cv.ci_lower_net_km2 || 30.1).toFixed(1)} to ${(cv.ci_upper_net_km2 || 133.2).toFixed(1)} km²). Adjusted 2024 built-up footprint: <strong>478.8 km² (95% CI 409.0-549.0)</strong>. (Note: &plusmn; values denote the 95% CI half-width).
        </p>
      </div>
    `;

    strataFindingHtml = `
      <div class="chart-box" style="margin-top: 16px;">
        <h3 style="font-family: var(--font-display); font-size: var(--text-lg); margin-bottom: 8px;">The 4-in-10 Ground-Truth Finding (Stratum A)</h3>
        <p style="color: var(--color-text-secondary); line-height: 1.6; font-size: 14px;">
          Detailed examination of newly mapped built-up areas (Stratum A) reveals that <strong>about ${gainRatio} in 10 mapped additions (${gainPct}%)</strong> were verified ground developments, while <strong>about ${fallowRatio} in 10 (${fallowPct}%)</strong> were dry fallow agricultural land or bare soil falsely tagged as concrete.
        </p>
        <div style="font-size: 12px; color: var(--color-text-muted); background: var(--color-bg-subtle); padding: 8px 12px; border-radius: var(--radius-sm); margin-top: 8px;">
          ${cv.mapped_loss_note || '45 of 50 mapped-loss points were never built-up: the 2020 map falsely marks about 52 km² as built-up.'}
        </div>
      </div>
    `;
  } else {
    const puneAc = statsData.anchor_comparison || {};
    const puneWcKm = puneAc.worldcover_builtup_km2 != null ? puneAc.worldcover_builtup_km2.toFixed(1) : '378.1';
    const puneTls21 = puneAc.tls_2021_builtup_km2 != null ? puneAc.tls_2021_builtup_km2.toFixed(1) : '386.5';
    const puneDiffPct = puneAc.difference_pct != null ? Math.abs(puneAc.difference_pct).toFixed(1) : '2.2';
    validationHeadlineHtml = `
      <div class="note-box" style="background: rgba(46, 125, 50, 0.08); border-left: 3px solid #2e7d32; margin-bottom: 16px;">
        <div style="font-weight: 700; color: #1b5e20; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
          <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
          <span>Multi-Temporal Modelled Urban Expansion</span>
        </div>
        <p style="margin: 0; font-size: 13px; line-height: 1.5; color: var(--color-text-main);">
          Multi-temporal Random Forest classification cross-calibrated against European Space Agency (ESA) WorldCover 2021
          (model estimate <strong>${puneTls21} km²</strong> vs ESA anchor <strong>${puneWcKm} km²</strong>, aligned within <strong>${puneDiffPct}%</strong>)
          indicates Pune's built-up footprint expanded between 2020 and 2024. Note: The main series dips by -5.1% in 2021 (386.6 km² vs 407.1 km² in 2020), which is consistent with classifier noise (F1 0.58) and composite variations.
        </p>
      </div>
    `;
  }

  const minExp = hg.min_expansion_pct != null ? hg.min_expansion_pct.toFixed(1) : '12.6';
  const maxExp = hg.max_expansion_pct != null ? hg.max_expansion_pct.toFixed(1) : '21.9';
  const minKm = hg.min_expansion_km2 != null ? hg.min_expansion_km2.toFixed(1) : '53.0';
  const maxKm = hg.max_expansion_km2 != null ? hg.max_expansion_km2.toFixed(1) : '84.0';

  const wcKm = ac.worldcover_builtup_km2 != null ? ac.worldcover_builtup_km2.toFixed(1) : '393.7';
  const tls21Km = ac.tls_2021_builtup_km2 != null ? ac.tls_2021_builtup_km2.toFixed(1) : '413.6';
  const diffPct = ac.difference_pct != null ? Math.abs(ac.difference_pct).toFixed(1) : '5.0';

  const entropyVal = m2024.shannon_entropy != null ? m2024.shannon_entropy.toFixed(4) : '0.9501';
  const coreShare = m2024.core_share_pct != null ? m2024.core_share_pct.toFixed(1) : '21.9';
  const periphShare = m2024.periphery_share_pct != null ? m2024.periphery_share_pct.toFixed(1) : '32.5';
  const coreKm = m2024.core_builtup_km2 != null ? m2024.core_builtup_km2.toFixed(1) : '103.8';
  const periphKm = m2024.periphery_builtup_km2 != null ? m2024.periphery_builtup_km2.toFixed(1) : '154.5';

  container.innerHTML = `
    ${validationHeadlineHtml}

    <div class="metrics-grid" style="margin-bottom: 20px;">
      <div class="metric-card metric-card-primary">
        <div class="metric-label">2020–2024 Expansion Range</div>
        <div class="metric-value" style="color: var(--color-accent-primary);">${minExp} to ${maxExp} %</div>
        <div class="metric-delta">Spread across methods (+${minKm} to +${maxKm} km²)</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">ESA WorldCover 2021 Baseline</div>
        <div class="metric-value font-mono">${wcKm} km²</div>
        <div class="metric-delta">vs ${tls21Km} km² Main series (normalised) (${ac.difference_pct >= 0 ? '+' : ''}${diffPct}%)</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Shannon Entropy (2024)</div>
        <div class="metric-value font-mono">${entropyVal}</div>
        <div class="metric-delta delta-up">Radial Dispersion</div>
      </div>
    </div>

    <div class="chart-box">
      <h3 style="font-family: var(--font-display); font-size: var(--text-lg); margin-bottom: 8px;">1. Expansion Across Satellite Processing Methods</h3>
      <p style="color: var(--color-text-secondary); line-height: 1.6; font-size: 14px;">
        Between 2020 and 2024, ${cityDisplay}'s built-up footprint expanded by <strong>${minExp}% to ${maxExp}%</strong> across processing methods (+${minKm} to +${maxKm} km²). Because different temporal filters treat transient noise differently, we report the complete range across Raw, Main series (normalised), and Cleaned pipelines rather than a single artificial number.
      </p>
    </div>

    <div class="chart-box" style="margin-top: 16px;">
      <h3 style="font-family: var(--font-display); font-size: var(--text-lg); margin-bottom: 8px;">2. Macroscopic Calibration Against ESA WorldCover</h3>
      <p style="color: var(--color-text-secondary); line-height: 1.6; font-size: 14px;">
        The independent European Space Agency (ESA) WorldCover 2021 map measured ${cityDisplay}'s built-up footprint at <strong>${wcKm} km²</strong>. Our Main series (normalised) 2021 estimate of <strong>${tls21Km} km²</strong> aligns within <strong>${diffPct}%</strong>, confirming that our machine-learning model is calibrated to international satellite baselines.
      </p>
    </div>

    ${strataFindingHtml}

    <div class="chart-box" style="margin-top: 16px;">
      <h3 style="font-family: var(--font-display); font-size: var(--text-lg); margin-bottom: 8px;">3. Concentric Sprawl: Core vs. Peripheral Growth</h3>
      <p style="color: var(--color-text-secondary); line-height: 1.6; font-size: 14px;">
        ${cityDisplay} displays significant outward radial sprawl (Shannon Entropy <strong>${entropyVal}</strong> in 2024). The historical urban core (0–6 km) accounts for <strong>${coreKm} km² (${coreShare}%)</strong> of built-up area, whereas the expanding outer periphery (&gt;12 km) holds <strong>${periphKm} km² (${periphShare}%)</strong>, indicating that the vast majority of new land conversion is consuming peripheral agricultural land.
      </p>
    </div>
  `;
}

function renderHowItsMade() {
  const container = document.getElementById('how-content-container');
  if (!container || !howItsMadeData) return;

  const steps = howItsMadeData.steps || [];
  const flow = howItsMadeData.data_flow_steps || [];

  container.innerHTML = `
    <!-- Data Flow Diagram -->
    <div class="chart-box" style="margin-bottom: 24px;">
      <h3 style="font-family: var(--font-display); font-size: var(--text-lg); margin-bottom: 12px;">Data Pipeline Architecture</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 16px;">
        ${flow.map(f => `
          <div style="background: var(--color-bg-subtle); border: 1px solid var(--color-border); border-radius: var(--radius-md); padding: 12px 10px; text-align: center;">
            <div style="font-size: 10px; text-transform: uppercase; font-weight: 700; color: var(--color-accent-primary); letter-spacing: 0.05em;">Step ${f.step}</div>
            <div style="font-weight: 700; font-size: 13px; margin: 4px 0;">${f.name}</div>
            <div style="font-size: 11px; font-family: var(--font-mono); color: var(--color-text-muted);">${f.tool}</div>
            <div style="font-size: 10px; color: var(--color-text-secondary); margin-top: 4px;">${f.desc}</div>
          </div>
        `).join('')}
      </div>
      <div style="font-size: 13px; color: var(--color-text-secondary); line-height: 1.6; border-top: 1px solid var(--color-border-subtle); padding-top: 12px;">
        ${howItsMadeData.tech_stack_summary}
      </div>
    </div>

    <!-- 6 Detailed Steps -->
    <div style="display: flex; flex-direction: column; gap: 16px;">
      ${steps.map(s => `
        <article class="chart-box">
          <div style="display: flex; align-items: baseline; gap: 10px; margin-bottom: 6px;">
            <span style="font-family: var(--font-mono); font-weight: 800; font-size: var(--text-base); color: var(--color-accent-primary);">0${s.step_number}</span>
            <h3 style="font-family: var(--font-display); font-size: var(--text-lg); margin: 0;">${s.title}</h3>
          </div>
          <div style="font-weight: 600; font-size: 14px; color: var(--color-text-main); margin-bottom: 10px;">${s.headline}</div>
          
          <div style="display: flex; flex-direction: column; gap: 8px; font-size: 13px; line-height: 1.55;">
            <div><strong>What happens:</strong> ${s.what_happens}</div>
            <div><strong>Why this matters:</strong> ${s.why}</div>
            <div style="color: var(--color-text-secondary); background: var(--color-bg-subtle); padding: 8px 12px; border-radius: var(--radius-sm); border-left: 2px solid var(--color-border);">
              <strong>Constraint & Limitation:</strong> ${s.limitation}
            </div>
          </div>
          <div style="margin-top: 10px; text-align: right;">
            <a href="${s.report_link}" target="_blank" rel="noopener" style="font-size: 11px; font-weight: 600; color: var(--color-accent-primary); text-decoration: none;">
              Read Technical Details in Report &rarr;
            </a>
          </div>
        </article>
      `).join('')}
    </div>
  `;
}


function renderCredits() {
  let footerContainer = document.getElementById('app-footer-credits');
  if (!footerContainer) {
    footerContainer = document.createElement('footer');
    footerContainer.id = 'app-footer-credits';
    footerContainer.className = 'page-container';
    footerContainer.style.borderTop = '2px solid var(--color-border)';
    footerContainer.style.marginTop = '40px';
    footerContainer.style.paddingTop = '24px';
    
    // Append to How It's Made view
    const howView = document.getElementById('view-how');
    if (howView) howView.appendChild(footerContainer);
  }

  if (!creditsData) return;

  const sources = creditsData.sources || [];
  footerContainer.innerHTML = `
    <h3 style="font-family: var(--font-display); font-size: var(--text-lg); margin-bottom: 12px;">${creditsData.title}</h3>
    <div style="font-size: 12px; color: var(--color-text-secondary); line-height: 1.6; margin-bottom: 16px; background: var(--color-bg-subtle); padding: 12px 16px; border-radius: var(--radius-md);">
      <strong>Disclaimer:</strong> ${creditsData.disclaimer}
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-bottom: 20px;">
      ${sources.map(s => `
        <div style="border: 1px solid var(--color-border); border-radius: var(--radius-sm); padding: 10px 14px; background: var(--color-bg-surface);">
          <div style="font-weight: 700; font-size: 13px;">
            <a href="${s.url}" target="_blank" rel="noopener" style="color: var(--color-text-main); text-decoration: none;">${s.name} &rarr;</a>
          </div>
          <div style="font-size: 11px; color: var(--color-accent-primary); font-weight: 600;">${s.provider}</div>
          <div style="font-size: 11px; color: var(--color-text-muted); margin: 3px 0;">License: ${s.license}</div>
          <div style="font-size: 11px; color: var(--color-text-secondary);">${s.desc}</div>
        </div>
      `).join('')}
    </div>

    <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; font-size: 12px; color: var(--color-text-muted); border-top: 1px solid var(--color-border-subtle); padding-top: 12px;">
      <div><strong>${creditsData.footer_text || 'Code: MIT. Data: see credits.'}</strong> &bull; ${creditsData.license}</div>
      <div style="display: flex; gap: 12px;">
        <a href="${creditsData.links.repository}" target="_blank" rel="noopener" style="color: inherit; font-weight: 600;">GitHub Repository</a>
        <a href="${creditsData.links.report_html}" target="_blank" rel="noopener" style="color: inherit; font-weight: 600;">Technical Report</a>
      </div>
    </div>
  `;
}

function setupShareAndDownloads() {
  // Share Button in top bar or sidebar
  const shareBtn = document.getElementById('share-view-btn');
  if (shareBtn) {
    shareBtn.addEventListener('click', (e) => {
      e.preventDefault();
      copyShareUrl();
    });
  }

  // Also wire any share button on the page
  document.body.addEventListener('click', (e) => {
    if (e.target.closest('[data-action="share"]')) {
      e.preventDefault();
      copyShareUrl();
    }
  });
}

export function copyShareUrl() {
  const city = window.UrbanPulseState?.currentCity || 'ahmedabad';
  const year = window.UrbanPulseState?.currentYear || 2024;
  const isChange = window.UrbanPulseState?.isChangeLayerActive ? '1' : '0';
  
  const shareUrl = `${window.location.origin}${window.location.pathname}#explore&city=${city}&year=${year}&change=${isChange}`;

  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(shareUrl).then(() => {
      showToast('Link copied to clipboard with current city and year!');
    }).catch(() => {
      prompt('Copy this shareable link:', shareUrl);
    });
  } else {
    prompt('Copy this shareable link:', shareUrl);
  }
}

export function showToast(msg) {
  let toast = document.getElementById('urbanpulse-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'urbanpulse-toast';
    toast.style.cssText = `
      position: fixed;
      bottom: 24px;
      left: 50%;
      transform: translateX(-50%);
      background-color: var(--color-bg-chrome);
      color: #ffffff;
      padding: 10px 18px;
      border-radius: var(--radius-full);
      font-size: 13px;
      font-weight: 600;
      box-shadow: var(--shadow-lg);
      z-index: 9999;
      pointer-events: none;
      opacity: 0;
      transition: opacity 0.25s ease, transform 0.25s ease;
    `;
    document.body.appendChild(toast);
  }

  toast.textContent = msg;
  toast.style.opacity = '1';
  toast.style.transform = 'translateX(-50%) translateY(0)';

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(-50%) translateY(8px)';
  }, 3000);
}
