/**
 * UrbanPulse Navigation Controller (web/js/navigation.js)
 * Manages view switching between Explore, Key Findings, How It's Made, and Glossary.
 * Syncs with URL hash and supports keyboard & mobile bottom tabs.
 */

export function initNavigation() {
  const views = {
    explore: document.getElementById('view-explore'),
    findings: document.getElementById('view-findings'),
    how: document.getElementById('view-how'),
    glossary: document.getElementById('view-glossary')
  };

  const navButtons = document.querySelectorAll('[data-nav-target]');

  function switchView(targetViewId, updateHash = true) {
    const cleanId = targetViewId.replace('#', '');
    const activeView = views[cleanId] || views.explore;

    // Toggle view elements
    Object.entries(views).forEach(([id, el]) => {
      if (el) {
        if (el === activeView) {
          el.classList.add('active');
          el.style.display = (id === 'explore') ? 'flex' : 'block';
        } else {
          el.classList.remove('active');
          el.style.display = 'none';
        }
      }
    });

    // Toggle active state on buttons (both desktop and mobile)
    navButtons.forEach(btn => {
      const target = btn.getAttribute('data-nav-target');
      if (target === cleanId) {
        btn.classList.add('active');
        btn.setAttribute('aria-selected', 'true');
      } else {
        btn.classList.remove('active');
        btn.setAttribute('aria-selected', 'false');
      }
    });

    // Sync hash
    if (updateHash) {
      window.location.hash = cleanId;
    }

    // Trigger map resize if switching back to explore
    if (cleanId === 'explore') {
      window.dispatchEvent(new CustomEvent('urbanpulse:mapresize'));
    }

    // Scroll to top of content page
    if (activeView && cleanId !== 'explore') {
      activeView.scrollTop = 0;
    }
  }

  // Attach button click listeners
  navButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      const target = btn.getAttribute('data-nav-target');
      switchView(target);
    });
  });

  // Hero strip quick buttons
  const heroExploreBtn = document.getElementById('hero-explore-btn');
  if (heroExploreBtn) {
    heroExploreBtn.addEventListener('click', () => switchView('explore'));
  }

  const heroHowBtn = document.getElementById('hero-how-btn');
  if (heroHowBtn) {
    heroHowBtn.addEventListener('click', () => switchView('how'));
  }

  // Handle URL hash changes & initial load
  function handleHash() {
    const hash = window.location.hash.slice(1);
    if (views[hash]) {
      switchView(hash, false);
    } else {
      switchView('explore', false);
    }
  }

  window.addEventListener('hashchange', handleHash);
  handleHash();

  return { switchView };
}
