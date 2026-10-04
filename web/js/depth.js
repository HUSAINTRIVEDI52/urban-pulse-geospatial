// web/js/depth.js
// Global Simple / Detailed View Depth Toggle & Calculation Info Modals

const DEPTH_STORAGE_KEY = 'urbanpulse_view_depth';
let currentDepth = 'simple';

const CALC_EXPLANATIONS = {
  'expansion-range': {
    title: 'How 2020–2024 Expansion is Calculated',
    meaning: 'Three ways of measuring: raw, cleaned and normalised.',
    calculation: 'Calculated by comparing the initial built-up footprint (2020) against the final footprint (2024) across three measurement methods: raw Random Forest, cleaned sensitivity, and main normalised series. Because each method filters noise differently, we present the full spread of results rather than a single artificial number.'
  },
  'worldcover-anchor': {
    title: 'How the WorldCover Anchor is Calculated',
    meaning: 'A separate global land-cover map we compare against.',
    calculation: 'The European Space Agency (ESA) produced WorldCover 2021 at 10m resolution using global training data. We re-sample this reference to our 60m grid within the exact same administrative AOI boundary to verify model calibration.'
  },
  'builtup-footprint': {
    title: 'How Annual Built-up Footprint is Calculated',
    meaning: 'Buildings, roads and other paved surfaces, estimated from 60 m pixels.',
    calculation: 'Each 60m x 60m pixel (0.0036 km²) classified as built-up is tallied across the metropolitan Area of Interest (AOI).'
  },
  'shannon-entropy': {
    title: 'How Shannon Entropy is Calculated',
    meaning: 'How evenly built-up land is spread across distance rings from the centre. Values near 1 mean it is not concentrated near the centre; this does not measure leapfrog development.',
    calculation: 'We divide the metropolitan area into concentric 2 km distance rings and calculate Shannon Entropy: H_n = -sum(p_i * log(p_i)) / log(k), where p_i is the proportion of total built-up land in ring i, and k is the number of rings. Note: outer rings are larger, so values are high for most cities.'
  },
  'core-periphery': {
    title: 'How Core vs. Periphery is Calculated',
    meaning: 'Tracks whether growth is densifying the historical city center or expanding into outer perimeter boundaries.',
    calculation: 'Core is defined as the innermost 0–6 km rings (historic city center), where existing built-up density is highest. Periphery is defined as rings beyond 12 km from the center.'
  },
  'growth-chart': {
    title: 'How the Multi-Series Growth Chart is Built',
    meaning: 'Compares different satellite processing pipelines over time to test whether detected trends are robust or algorithm-dependent.',
    howToRead: 'Each line represents a measurement approach. Yellow is raw classification, green is cleaned sensitivity, red is the main series (normalised), and purple marks the ESA WorldCover anchor.'
  },
  'ring-chart': {
    title: 'How the Concentric Rings Chart is Built',
    meaning: 'Displays the proportion of land that is paved or built up as you travel outward from the municipal center to the outskirts.',
    howToRead: 'The horizontal axis shows 2-kilometer distance intervals from the city center outward. The vertical axis shows built-up density percentage (0–100%).'
  }
};

export function initDepthToggle() {
  // Read initial preference
  try {
    const saved = localStorage.getItem(DEPTH_STORAGE_KEY);
    if (saved === 'detailed' || saved === 'simple') {
      currentDepth = saved;
    }
  } catch (e) {
    // LocalStorage blocked
  }

  applyDepth(currentDepth);

  const toggleHandler = () => {
    const newDepth = currentDepth === 'simple' ? 'detailed' : 'simple';
    setDepth(newDepth);
  };

  const toggleBtn = document.getElementById('toggle-depth-btn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', toggleHandler);
  }

  document.querySelectorAll('.mobile-depth-toggle').forEach(btn => {
    btn.addEventListener('click', toggleHandler);
  });

  setupCalcInfoButtons();
  window.UrbanPulseDepth = { setDepth, getDepth, applyDepth };
}

export function setDepth(depth) {
  currentDepth = depth;
  try {
    localStorage.setItem(DEPTH_STORAGE_KEY, depth);
  } catch (e) {
    // Ignore
  }
  applyDepth(currentDepth);
}

export function getDepth() {
  return currentDepth;
}

function applyDepth(depth) {
  const isDetailed = depth === 'detailed';
  document.body.classList.toggle('view-detailed', isDetailed);
  document.body.classList.toggle('view-simple', !isDetailed);

  const btn = document.getElementById('toggle-depth-btn');
  const label = document.getElementById('depth-label');
  if (btn) {
    btn.setAttribute('aria-pressed', isDetailed ? 'true' : 'false');
    btn.classList.toggle('active', isDetailed);
  }
  if (label) {
    label.textContent = isDetailed ? 'Detailed' : 'Simple';
  }

  document.querySelectorAll('.mobile-depth-label').forEach(lbl => {
    lbl.textContent = isDetailed ? 'Detailed' : 'Simple';
  });

  // Update card titles and explanations
  document.querySelectorAll('[data-simple-title]').forEach(el => {
    const simple = el.getAttribute('data-simple-title');
    const detailed = el.getAttribute('data-detailed-title') || simple;
    el.textContent = isDetailed ? detailed : simple;
  });

  document.querySelectorAll('[data-simple-meaning]').forEach(el => {
    const meaning = el.getAttribute('data-simple-meaning');
    if (meaning) el.textContent = meaning;
  });
}

function setupCalcInfoButtons() {
  document.body.addEventListener('click', (e) => {
    const infoBtn = e.target.closest('[data-calc-info]');
    if (infoBtn) {
      e.preventDefault();
      const calcKey = infoBtn.getAttribute('data-calc-info');
      showCalcModal(calcKey);
    }
  });
}

function showCalcModal(key) {
  const info = CALC_EXPLANATIONS[key];
  if (!info) return;

  let modal = document.getElementById('calc-info-modal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'calc-info-modal';
    modal.className = 'calc-modal-backdrop';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-modal', 'true');
    modal.innerHTML = `
      <div class="calc-modal-card">
        <div class="calc-modal-header">
          <h3 id="calc-modal-title" class="calc-modal-title"></h3>
          <button id="calc-modal-close" class="tour-close-btn" aria-label="Close dialog">&times;</button>
        </div>
        <div class="calc-modal-body">
          <div class="calc-modal-section">
            <h4 class="calc-section-label">What This Means (Plain Language)</h4>
            <p id="calc-modal-meaning" class="calc-section-text"></p>
          </div>
          <div class="calc-modal-section">
            <h4 class="calc-section-label">How It Is Calculated</h4>
            <p id="calc-modal-calc" class="calc-section-text"></p>
          </div>
        </div>
        <div class="calc-modal-footer">
          <button id="calc-modal-dismiss" class="hero-btn hero-btn-primary" style="font-size: 13px; padding: 6px 16px;">Got It</button>
        </div>
      </div>
    `;
    document.body.appendChild(modal);

    modal.querySelector('#calc-modal-close').addEventListener('click', () => modal.style.display = 'none');
    modal.querySelector('#calc-modal-dismiss').addEventListener('click', () => modal.style.display = 'none');
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.style.display = 'none';
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && modal.style.display !== 'none') modal.style.display = 'none';
    });
  }

  modal.querySelector('#calc-modal-title').textContent = info.title;
  modal.querySelector('#calc-modal-meaning').textContent = info.meaning;
  modal.querySelector('#calc-modal-calc').textContent = info.calculation || info.howToRead || '';

  modal.style.display = 'flex';
}
