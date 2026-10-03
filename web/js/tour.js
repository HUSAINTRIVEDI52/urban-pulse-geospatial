// web/js/tour.js
// First-visit 4-step guided tour for UrbanPulse

const TOUR_STORAGE_KEY = 'urbanpulse_tour_completed';

const TOUR_STEPS = [
  {
    target: '#map',
    title: '1. What the Map Shows',
    content: 'The satellite map classifies metropolitan land into five distinct categories. Bright red marks built-up urban structures and roads, green indicates vegetation, blue is water, and yellow represents active seasonal farmland.',
    position: 'center'
  },
  {
    target: '#timeline-slider-card',
    title: '2. The Year Slider',
    content: 'Drag the timeline slider or press the Play button to step through annual satellite snapshots from 2020 to 2024. Watch how peripheral farmland transforms into urban built-up land year by year.',
    position: 'top'
  },
  {
    target: '#toggle-compare-btn',
    title: '3. Compare Mode',
    content: 'Click Compare Mode to activate side-by-side swipe comparison. You can slide the divider across the city to contrast 2020 ground reality against 2024, or toggle the Change Layer to see gains and losses highlighted.',
    position: 'left'
  },
  {
    target: '#dashboard',
    title: '4. The Analytics Dashboard',
    content: 'This panel shows error-adjusted growth calculations, independent ESA WorldCover benchmarks, spatial entropy (sprawl dispersion), and concentric density rings. Switch between Simple and Detailed view anytime!',
    position: 'left'
  }
];

let currentStepIndex = 0;
let tourOverlay = null;

export function initTour() {
  setupHelpButton();

  // Check first visit
  try {
    const completed = localStorage.getItem(TOUR_STORAGE_KEY);
    if (!completed) {
      // Delay slightly for initial map render
      setTimeout(() => startTour(0), 1200);
    }
  } catch (e) {
    // LocalStorage blocked, allow user to trigger manually
  }
}

export function startTour(startStep = 0) {
  currentStepIndex = startStep;
  createTourDOM();
  renderStep(currentStepIndex);
}

function setupHelpButton() {
  const helpBtn = document.getElementById('tour-help-btn');
  if (helpBtn) {
    helpBtn.addEventListener('click', (e) => {
      e.preventDefault();
      startTour(0);
    });
  }
}

function createTourDOM() {
  if (tourOverlay) return;

  tourOverlay = document.createElement('div');
  tourOverlay.id = 'urbanpulse-tour-overlay';
  tourOverlay.className = 'tour-backdrop';
  tourOverlay.setAttribute('role', 'dialog');
  tourOverlay.setAttribute('aria-label', 'Interactive guided tour');
  tourOverlay.innerHTML = `
    <div class="tour-spotlight" id="tour-spotlight"></div>
    <div class="tour-card" id="tour-card">
      <div class="tour-header">
        <span class="tour-step-count" id="tour-step-badge">Step 1 of 4</span>
        <button class="tour-close-btn" id="tour-skip-btn" aria-label="Skip tour">&times;</button>
      </div>
      <h3 class="tour-title" id="tour-step-title"></h3>
      <p class="tour-text" id="tour-step-text"></p>
      <div class="tour-footer">
        <button class="tour-btn tour-btn-secondary" id="tour-prev-btn">Previous</button>
        <div style="flex: 1;"></div>
        <button class="tour-btn tour-btn-primary" id="tour-next-btn">Next</button>
      </div>
    </div>
  `;
  document.body.appendChild(tourOverlay);

  // Wire buttons
  tourOverlay.querySelector('#tour-skip-btn').addEventListener('click', endTour);
  tourOverlay.querySelector('#tour-prev-btn').addEventListener('click', () => {
    if (currentStepIndex > 0) {
      currentStepIndex--;
      renderStep(currentStepIndex);
    }
  });
  tourOverlay.querySelector('#tour-next-btn').addEventListener('click', () => {
    if (currentStepIndex < TOUR_STEPS.length - 1) {
      currentStepIndex++;
      renderStep(currentStepIndex);
    } else {
      endTour();
    }
  });

  // Keyboard navigation
  document.addEventListener('keydown', handleKeydown);
}

function handleKeydown(e) {
  if (!tourOverlay || tourOverlay.style.display === 'none') return;
  if (e.key === 'Escape') endTour();
  if (e.key === 'ArrowRight') {
    if (currentStepIndex < TOUR_STEPS.length - 1) {
      currentStepIndex++;
      renderStep(currentStepIndex);
    }
  }
  if (e.key === 'ArrowLeft') {
    if (currentStepIndex > 0) {
      currentStepIndex--;
      renderStep(currentStepIndex);
    }
  }
}

function renderStep(index) {
  const step = TOUR_STEPS[index];
  if (!step) return;

  const badge = document.getElementById('tour-step-badge');
  const title = document.getElementById('tour-step-title');
  const text = document.getElementById('tour-step-text');
  const prevBtn = document.getElementById('tour-prev-btn');
  const nextBtn = document.getElementById('tour-next-btn');
  const card = document.getElementById('tour-card');
  const spotlight = document.getElementById('tour-spotlight');

  badge.textContent = `Step ${index + 1} of ${TOUR_STEPS.length}`;
  title.textContent = step.title;
  text.textContent = step.content;

  prevBtn.style.visibility = index === 0 ? 'hidden' : 'visible';
  nextBtn.textContent = index === TOUR_STEPS.length - 1 ? 'Finish' : 'Next';

  // Position card and spotlight around target element
  const targetEl = document.querySelector(step.target);
  if (targetEl && targetEl.offsetParent !== null) {
    const rect = targetEl.getBoundingClientRect();
    spotlight.style.display = 'block';
    spotlight.style.top = `${rect.top - 6}px`;
    spotlight.style.left = `${rect.left - 6}px`;
    spotlight.style.width = `${rect.width + 12}px`;
    spotlight.style.height = `${rect.height + 12}px`;

    // Position tooltip card comfortably
    positionCard(card, rect, step.position);
  } else {
    spotlight.style.display = 'none';
    card.style.top = '50%';
    card.style.left = '50%';
    card.style.transform = 'translate(-50%, -50%)';
  }

  tourOverlay.style.display = 'block';
}

function positionCard(card, targetRect, preferredPosition) {
  card.style.transform = 'none';
  const cardWidth = Math.min(360, window.innerWidth - 32);
  card.style.width = `${cardWidth}px`;

  // On mobile screens, stick card to bottom or center
  if (window.innerWidth <= 768) {
    card.style.top = 'auto';
    card.style.bottom = '80px';
    card.style.left = '16px';
    card.style.right = '16px';
    card.style.width = 'auto';
    return;
  }

  let top = targetRect.top;
  let left = targetRect.left;

  if (preferredPosition === 'center') {
    top = Math.max(100, (window.innerHeight - 280) / 2);
    left = Math.max(20, (window.innerWidth - cardWidth) / 2);
  } else if (preferredPosition === 'left') {
    left = Math.max(16, targetRect.left - cardWidth - 20);
    top = Math.max(80, Math.min(window.innerHeight - 300, targetRect.top + 20));
  } else if (preferredPosition === 'top') {
    top = Math.max(80, targetRect.top - 240);
    left = Math.max(16, Math.min(window.innerWidth - cardWidth - 16, targetRect.left + (targetRect.width / 2) - (cardWidth / 2)));
  } else {
    top = Math.max(80, Math.min(window.innerHeight - 300, targetRect.bottom + 20));
    left = Math.max(16, Math.min(window.innerWidth - cardWidth - 16, targetRect.left));
  }

  card.style.top = `${Math.round(top)}px`;
  card.style.left = `${Math.round(left)}px`;
}

function endTour() {
  if (tourOverlay) {
    tourOverlay.style.display = 'none';
  }
  document.removeEventListener('keydown', handleKeydown);
  try {
    localStorage.setItem(TOUR_STORAGE_KEY, 'true');
  } catch (e) {
    // Ignore
  }
}
