/**
 * UrbanPulse Theme Controller (web/js/theme.js)
 * Supports System Theme and Manual Light/Dark Toggle with localStorage persistence.
 */

export function initTheme() {
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  const themeIcon = document.getElementById('theme-toggle-icon');

  function getSavedTheme() {
    try {
      return localStorage.getItem('urbanpulse_theme');
    } catch {
      return null;
    }
  }

  function saveTheme(theme) {
    try {
      localStorage.setItem('urbanpulse_theme', theme);
    } catch {
      // Ignore private browsing storage restriction
    }
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (themeIcon) {
      themeIcon.innerHTML = theme === 'dark' 
        ? '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'
        : '<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>';
    }
    window.dispatchEvent(new CustomEvent('urbanpulse:themechange', { detail: { theme } }));
  }

  // Initial theme determination (Default: light mode)
  const saved = getSavedTheme();
  if (saved === 'dark' || saved === 'light') {
    applyTheme(saved);
  } else {
    applyTheme('light');
  }

  // Toggle button listener
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      saveTheme(next);
    });
  }
}
