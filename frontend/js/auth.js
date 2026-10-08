/**
 * Pragya-Edu | auth.js
 * Session guard and user profile utilities
 */

'use strict';

const PragyaAuth = (() => {

  const REQUIRED_KEYS = ['pe_name', 'pe_topic'];

  function isLoggedIn() {
    return REQUIRED_KEYS.every(k => !!localStorage.getItem(k));
  }

  function guardSession(redirectTo = 'login.html') {
    if (!isLoggedIn()) {
      window.location.href = redirectTo;
      return false;
    }
    return true;
  }

  function getProfile() {
    return {
      name:    localStorage.getItem('pe_name')     || 'Learner',
      topic:   localStorage.getItem('pe_topic')    || 'General',
      lang:    localStorage.getItem('pe_lang')     || 'en',
      role:    localStorage.getItem('pe_role')     || 'student',
      profile: JSON.parse(localStorage.getItem('pe_profile') || '{}'),
      score:   parseInt(localStorage.getItem('pe_engagement') || '0'),
    };
  }

  function logout() {
    const keys = Object.keys(localStorage).filter(k => k.startsWith('pe_'));
    keys.forEach(k => localStorage.removeItem(k));
    window.location.href = 'login.html';
  }

  function showToast(msg, type = 'info', duration = 3000) {
    const el = document.createElement('div');
    el.className = 'toast-edu';
    const icons = { info: 'fa-circle-info', success: 'fa-circle-check', warning: 'fa-triangle-exclamation', error: 'fa-circle-xmark' };
    const colors = { info: 'var(--edu-violet)', success: 'var(--edu-emerald)', warning: 'var(--edu-amber)', error: 'var(--edu-rose)' };
    el.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}" style="color:${colors[type] || colors.info};font-size:1.1rem;flex-shrink:0;"></i><span>${msg}</span>`;
    document.body.appendChild(el);
    setTimeout(() => el.classList.add('hide'), duration - 400);
    setTimeout(() => el.remove(), duration);
  }

  function populateSidebarUser() {
    const p = getProfile();
    document.querySelectorAll('[data-user-name]').forEach(el => el.textContent = p.name);
    document.querySelectorAll('[data-user-topic]').forEach(el => el.textContent = p.topic);
    document.querySelectorAll('[data-user-initial]').forEach(el => el.textContent = p.name[0].toUpperCase());
  }

  return { isLoggedIn, guardSession, getProfile, logout, showToast, populateSidebarUser };
})();

// Auto-run on pages that need auth (all except login.html)
document.addEventListener('DOMContentLoaded', () => {
  if (!window.location.pathname.includes('login.html')) {
    // Allow access without login for prototype demo purposes
    // PragyaAuth.guardSession(); // Uncomment to enforce login
  }
  PragyaAuth.populateSidebarUser();
});
