// web/js/glossary.js
// UrbanPulse Glossary & Inline Term Definitions Module

let glossaryTerms = [];
let glossaryMap = new Map();
let activePopover = null;

export async function initGlossary() {
  try {
    const res = await fetch('data/glossary.json');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    glossaryTerms = data.terms || [];
    glossaryMap.clear();
    for (const item of glossaryTerms) {
      glossaryMap.set(item.id.toLowerCase(), item);
      glossaryMap.set(item.term.toLowerCase(), item);
      if (item.aliases) {
        for (const a of item.aliases) {
          glossaryMap.set(a.toLowerCase(), item);
        }
      }
    }
  } catch (err) {
    console.warn('[UrbanPulse] Could not load glossary.json:', err);
    return;
  }

  renderGlossaryPage();
  setupInlineTermListeners();
  setupPopover();
}

export function getTerm(termOrId) {
  if (!termOrId) return null;
  return glossaryMap.get(String(termOrId).toLowerCase().trim()) || null;
}

export function openGlossaryDrawer(termId) {
  // If termId provided, navigate to glossary and highlight/filter
  const term = getTerm(termId);
  const glossaryNavLink = document.querySelector('[data-view="glossary"]');
  if (glossaryNavLink) {
    glossaryNavLink.click();
  } else {
    window.location.hash = '#glossary';
  }

  if (term) {
    setTimeout(() => {
      const card = document.getElementById(`glossary-card-${term.id}`);
      if (card) {
        card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        card.classList.add('highlight-pulse');
        setTimeout(() => card.classList.remove('highlight-pulse'), 2500);
      }
      const searchInput = document.getElementById('glossary-search-input');
      if (searchInput) searchInput.value = term.term;
      filterGlossary(term.term);
    }, 150);
  }
}

function setupInlineTermListeners() {
  document.body.addEventListener('click', (e) => {
    const termEl = e.target.closest('[data-term]');
    if (termEl) {
      e.preventDefault();
      const termId = termEl.getAttribute('data-term');
      showPopover(termEl, termId);
    } else if (activePopover && !e.target.closest('#glossary-popover')) {
      hidePopover();
    }
  });

  document.body.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && activePopover) {
      hidePopover();
    }
  });
}

function setupPopover() {
  let popover = document.getElementById('glossary-popover');
  if (!popover) {
    popover = document.createElement('div');
    popover.id = 'glossary-popover';
    popover.className = 'glossary-popover';
    popover.setAttribute('role', 'dialog');
    popover.setAttribute('aria-modal', 'false');
    popover.setAttribute('aria-label', 'Term definition');
    popover.innerHTML = `
      <div class="popover-arrow"></div>
      <div class="popover-content">
        <div class="popover-header">
          <h4 id="popover-term-title" class="popover-title"></h4>
          <button id="popover-close-btn" class="popover-close" aria-label="Close definition">&times;</button>
        </div>
        <p id="popover-term-def" class="popover-def"></p>
        <div class="popover-actions">
          <button id="popover-more-btn" class="popover-more-link">Full Definition in Glossary &rarr;</button>
        </div>
      </div>
    `;
    document.body.appendChild(popover);

    popover.querySelector('#popover-close-btn').addEventListener('click', hidePopover);
    popover.querySelector('#popover-more-btn').addEventListener('click', () => {
      const currentTermId = popover.getAttribute('data-current-term-id');
      hidePopover();
      if (currentTermId) openGlossaryDrawer(currentTermId);
    });
  }
}

function showPopover(anchorEl, termId) {
  const term = getTerm(termId);
  if (!term) return;

  const popover = document.getElementById('glossary-popover');
  if (!popover) return;

  popover.setAttribute('data-current-term-id', term.id);
  popover.querySelector('#popover-term-title').textContent = term.term;
  popover.querySelector('#popover-term-def').textContent = term.definition;

  // Calculate position
  const rect = anchorEl.getBoundingClientRect();
  popover.style.display = 'block';
  popover.style.visibility = 'hidden';

  const popoverRect = popover.getBoundingClientRect();
  let top = rect.bottom + 8;
  let left = rect.left + (rect.width / 2) - (popoverRect.width / 2);

  // Bounds checking
  if (left < 16) left = 16;
  if (left + popoverRect.width > window.innerWidth - 16) {
    left = window.innerWidth - popoverRect.width - 16;
  }
  if (top + popoverRect.height > window.innerHeight - 16) {
    top = rect.top - popoverRect.height - 8;
  }

  popover.style.top = `${Math.round(top)}px`;
  popover.style.left = `${Math.round(left)}px`;
  popover.style.visibility = 'visible';
  popover.classList.add('visible');
  activePopover = popover;
}

function hidePopover() {
  const popover = document.getElementById('glossary-popover');
  if (popover) {
    popover.classList.remove('visible');
    popover.style.display = 'none';
  }
  activePopover = null;
}

function renderGlossaryPage() {
  const container = document.getElementById('glossary-terms-container');
  if (!container) return;

  const searchInput = document.getElementById('glossary-search-input');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => filterGlossary(e.target.value));
  }

  // Sort alphabetically
  const sorted = [...glossaryTerms].sort((a, b) => a.term.localeCompare(b.term));

  container.innerHTML = sorted.map(t => `
    <article id="glossary-card-${t.id}" class="glossary-card" data-term-id="${t.id}">
      <div class="glossary-card-header">
        <h3 class="glossary-card-title">${t.term}</h3>
        <span class="glossary-tag font-mono">#${t.id}</span>
      </div>
      <p class="glossary-card-def font-medium">${t.definition}</p>
      <div class="glossary-card-detail">
        <p>${t.detail}</p>
      </div>
      ${t.report_link ? `
        <div class="glossary-card-footer">
          <a href="${t.report_link}" class="glossary-report-link" target="_blank" rel="noopener">
            View methodology in Report &rarr;
          </a>
        </div>
      ` : ''}
    </article>
  `).join('');
}

function filterGlossary(query) {
  const q = (query || '').toLowerCase().trim();
  const cards = document.querySelectorAll('.glossary-card');
  let matchedCount = 0;

  cards.forEach(card => {
    const text = card.textContent.toLowerCase();
    const matches = !q || text.includes(q);
    card.style.display = matches ? 'block' : 'none';
    if (matches) matchedCount++;
  });

  const emptyMsg = document.getElementById('glossary-empty-msg');
  if (emptyMsg) {
    emptyMsg.style.display = matchedCount === 0 ? 'block' : 'none';
  }
}
