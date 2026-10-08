/**
 * Pragya-Edu | api.js
 * 6-Tier Fallback API Wrapper
 * Tier 1: Live Cloud API
 * Tier 2: Local vector search (localStorage cache)
 * Tier 3: Mock data generator
 * Tier 4: Rule-based engine
 * Tier 5: Static content pack
 * Tier 6: Text-only minimal mode
 */

'use strict';

const PragyaAPI = (() => {

  // ── Config ───────────────────────────────────────────────
  const BASE_URL = 'http://localhost:8000/api';
  const TIMEOUT_MS = 5000;
  const CACHE_PREFIX = 'pe_cache_';

  // ── Connectivity helpers ─────────────────────────────────
  function isOnline() { return navigator.onLine; }

  async function fetchWithTimeout(url, options = {}, ms = TIMEOUT_MS) {
    const ctrl = new AbortController();
    const id = setTimeout(() => ctrl.abort(), ms);
    try {
      const res = await fetch(url, { ...options, signal: ctrl.signal });
      clearTimeout(id);
      return res;
    } catch (e) {
      clearTimeout(id);
      throw e;
    }
  }

  // ── Cache helpers ────────────────────────────────────────
  function cacheSet(key, data, ttlMs = 3600000) {
    try {
      localStorage.setItem(CACHE_PREFIX + key, JSON.stringify({ data, expires: Date.now() + ttlMs }));
    } catch (e) { /* storage full */ }
  }

  function cacheGet(key) {
    try {
      const raw = localStorage.getItem(CACHE_PREFIX + key);
      if (!raw) return null;
      const { data, expires } = JSON.parse(raw);
      if (Date.now() > expires) { localStorage.removeItem(CACHE_PREFIX + key); return null; }
      return data;
    } catch (e) { return null; }
  }

  // ── Tier 3: Mock Data Generators ─────────────────────────
  function getMockDiagnosticQuiz(topic) {
    return {
      topic,
      questions: [
        { id: 1, question: `What is the fundamental principle of ${topic}?`, options: ['Pattern recognition', 'Random guessing', 'Memorization only', 'Surface observation'], correct: 0, explanation: `${topic} at its core is about recognizing and applying patterns.` },
        { id: 2, question: `Which approach best supports mastery of ${topic}?`, options: ['Active problem-solving with feedback', 'Passive reading', 'Last-minute cramming', 'Skipping practice'], correct: 0, explanation: 'Active learning with immediate feedback accelerates retention by 3x.' },
        { id: 3, question: `How does ${topic} connect to real-world applications?`, options: ['No real-world relevance', 'Only in labs', 'Through systematic pattern application', 'Only theoretically'], correct: 2, explanation: `${topic} provides frameworks for solving real problems systematically.` },
        { id: 4, question: `What distinguishes a novice from an expert in ${topic}?`, options: ['Speed', 'Memorization volume', 'Depth of conceptual understanding', 'Formulae count'], correct: 2, explanation: 'Experts understand the WHY, not just the HOW.' },
        { id: 5, question: `Which mindset accelerates learning ${topic}?`, options: ['Fixed mindset (I\'m not smart enough)', 'Growth mindset (I can learn this)', 'Avoidance', 'Over-confidence'], correct: 1, explanation: 'Growth mindset learners outperform fixed mindset by significant margins.' },
      ],
      generatedAt: new Date().toISOString(),
      source: 'local_mock'
    };
  }

  function getMockELI5(topic, context = 'realworld') {
    const openings = {
      gaming: `In a legendary RPG called ${topic}-Quest,`,
      scifi: `Aboard the starship Curiosity in the year 2347,`,
      sports: `In the championship game of your life,`,
      realworld: `Imagine you're building something from scratch —`,
      stories: `Once upon a time, in a mysterious realm,`
    };
    const opening = openings[context] || openings.realworld;
    return {
      topic,
      story: `${opening} you encounter the ultimate challenge: mastering ${topic}.\n\n` +
        `Think of ${topic} like the rules of a game. Once you understand the rules deeply, you can predict what happens next and make better decisions.\n\n` +
        `The secret? Every expert started exactly where you are right now. The difference is they asked "why?" one more time than everyone else.`,
      source: 'local_mock'
    };
  }

  function getMockFlashcards(topic) {
    return [
      { id: 1, front: `🤔 What is ${topic}?`, back: `The systematic study of relationships and patterns within a defined domain.` },
      { id: 2, front: `⚡ Core Rule`, back: `Input → Process → Output. Every instance follows this universal flow.` },
      { id: 3, front: `🌍 Real Example`, back: `${topic} appears when you observe cause-and-effect in daily life.` },
      { id: 4, front: `❌ Misconception`, back: `Most students confuse surface features with underlying principles. Focus on the WHY.` },
      { id: 5, front: `🔗 Connections`, back: `${topic} links to foundational principles and bridges to advanced topics.` },
      { id: 6, front: `🏆 One-Line Summary`, back: `"${topic} is the art of seeing hidden order and using it to predict and create."` },
    ];
  }

  function getMockSocraticQuestion(topic, depth = 'surface') {
    const questions = {
      surface: [
        `Before we start — what do you already *think* you know about ${topic}? Our assumptions are our biggest blind spots.`,
        `If you had to explain ${topic} to a 10-year-old using only 3 words, what would they be?`,
        `What would happen to the world if ${topic} suddenly stopped working?`
      ],
      medium: [
        `Fascinating! But WHY does that happen? What's the underlying mechanism that drives it?`,
        `You're on the right track. Now challenge yourself: where would this completely BREAK DOWN?`,
        `Good. Now connect the dots — how does this relate to something you know really well already?`
      ],
      deep: [
        `Let's go to first principles. What is the FINAL CAUSE here — the ultimate purpose?`,
        `If ${topic} is so critical, why do most people fundamentally misunderstand it?`,
        `What would you need to UNLEARN to understand this at the deepest level?`
      ]
    };
    const pool = questions[depth] || questions.surface;
    return pool[Math.floor(Math.random() * pool.length)].replace(/\{topic\}/g, topic);
  }

  // ── Tier 4: Rule-based engagement scorer ─────────────────
  function ruleBasedScore(profile) {
    let score = 50;
    if (profile.energy === 'high') score += 20;
    else if (profile.energy === 'burnout' || profile.energy === 'lost') score -= 20;
    if (profile.baseline === 'good' || profile.baseline === 'expert') score += 10;
    if (profile.goal === 'curious') score += 15;
    else if (profile.goal === 'stuck') score -= 10;
    if (profile.attention === 'long') score += 10;
    else if (profile.attention === 'micro') score -= 5;
    return Math.max(0, Math.min(100, score));
  }

  // ── Main API Methods ─────────────────────────────────────

  /**
   * Diagnostic Quiz — 6-tier fallback
   */
  async function getDiagnosticQuiz(topic) {
    // Tier 2: Check cache
    const cached = cacheGet('quiz_' + topic);
    if (cached) return { ...cached, source: 'cache' };

    // Tier 1: Try live API
    if (isOnline()) {
      try {
        const res = await fetchWithTimeout(`${BASE_URL}/quiz/generate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic, count: 5 })
        });
        if (res.ok) {
          const data = await res.json();
          cacheSet('quiz_' + topic, data);
          return { ...data, source: 'cloud' };
        }
      } catch (e) { /* fall through */ }
    }

    // Tier 3: Mock data
    return getMockDiagnosticQuiz(topic);
  }

  /**
   * ELI5 Story — 6-tier fallback
   */
  async function getELI5(topic, context = 'realworld') {
    const cached = cacheGet('eli5_' + topic + '_' + context);
    if (cached) return { ...cached, source: 'cache' };

    if (isOnline()) {
      try {
        const res = await fetchWithTimeout(`${BASE_URL}/content/eli5`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic, context })
        });
        if (res.ok) {
          const data = await res.json();
          cacheSet('eli5_' + topic + '_' + context, data);
          return { ...data, source: 'cloud' };
        }
      } catch (e) { /* fall through */ }
    }

    return getMockELI5(topic, context);
  }

  /**
   * Flashcards
   */
  async function getFlashcards(topic) {
    const cached = cacheGet('flash_' + topic);
    if (cached) return cached;

    if (isOnline()) {
      try {
        const res = await fetchWithTimeout(`${BASE_URL}/content/flashcards?topic=${encodeURIComponent(topic)}`);
        if (res.ok) {
          const data = await res.json();
          cacheSet('flash_' + topic, data);
          return data;
        }
      } catch (e) { /* fall through */ }
    }

    return getMockFlashcards(topic);
  }

  /**
   * Socratic Question
   */
  async function getSocraticQuestion(topic, depth = 'surface', context = '') {
    if (isOnline()) {
      try {
        const res = await fetchWithTimeout(`${BASE_URL}/mentor/socratic`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic, depth, context })
        });
        if (res.ok) {
          const data = await res.json();
          return data.question;
        }
      } catch (e) { /* fall through */ }
    }

    return getMockSocraticQuestion(topic, depth);
  }

  /**
   * Check backend live status & active AI tier
   */
  async function checkBackendStatus() {
    try {
      const res = await fetchWithTimeout(`${BASE_URL}/status`, {}, 2000);
      if (res.ok) {
        return await res.json();
      }
    } catch (e) { /* server offline */ }
    return {
      status: 'offline',
      active_tier: 6,
      tier_label: 'Local Edge Engine',
      tier_status: '🟡 OFFLINE / EDGE MODE'
    };
  }

  /**
   * Upload Document (PDF/TXT) with 6-tier fallback
   */
  async function uploadDocument(file, mode = 'all', style = 'gaming') {
    if (isOnline()) {
      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('mode', mode);
        formData.append('style', style);
        formData.append('session_id', localStorage.getItem('pe_name') || 'anon');

        const res = await fetchWithTimeout(`${BASE_URL}/curriculum-adapter/upload`, {
          method: 'POST',
          body: formData
        }, 12000);

        if (res.ok) {
          const data = await res.json();
          return { ...data, source: 'cloud' };
        }
      } catch (e) {
        console.warn('Upload fallback to local processor:', e);
      }
    }

    // Local edge mock fallback based on file name
    const topic = file.name.replace(/\.[^/.]+$/, "").replace(/[-_]/g, " ");
    return {
      topic: topic,
      filename: file.name,
      eli5: getMockELI5(topic, style),
      flashcards: getMockFlashcards(topic),
      quiz: getMockDiagnosticQuiz(topic),
      source: 'local_edge'
    };
  }

  /**
   * Multi-turn Socratic conversation turn
   */
  async function askSocraticTurn(topic, depth = 'surface', userAnswer = '', sessionId = 'anon') {
    if (isOnline() && userAnswer) {
      try {
        const res = await fetchWithTimeout(`${BASE_URL}/socratic-mentor/continue`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic,
            depth,
            user_answer: userAnswer,
            session_id: sessionId
          })
        }, 6000);
        if (res.ok) {
          return await res.json();
        }
      } catch (e) { /* fallback */ }
    }

    const nextQ = getMockSocraticQuestion(topic, depth);
    return {
      question: nextQ,
      depth: depth,
      probe_type: 'heuristic_probe',
      source: 'local_edge'
    };
  }

  /**
   * Compute engagement profile
   */
  function computeProfile(answers) {
    const profile = {
      energy: answers.energy || 'medium',
      context: answers.context || 'realworld',
      baseline: answers.baseline || 'basic',
      attention: answers.attention || 'short',
      goal: answers.goal || 'curious'
    };
    profile.engagementScore = ruleBasedScore(profile);
    return profile;
  }

  /**
   * Save telemetry (non-blocking)
   */
  function saveTelemetry(event, data) {
    const entry = { event, data, timestamp: Date.now(), session: localStorage.getItem('pe_name') || 'anon' };
    const log = JSON.parse(localStorage.getItem('pe_telemetry') || '[]');
    log.push(entry);
    if (log.length > 200) log.splice(0, 50); // keep last 150
    localStorage.setItem('pe_telemetry', JSON.stringify(log));

    if (isOnline()) {
      fetch(`${BASE_URL}/telemetry`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entry)
      }).catch(() => { /* fire and forget */ });
    }
  }

  // ── Public API ───────────────────────────────────────────
  return {
    BASE_URL,
    getDiagnosticQuiz,
    getELI5,
    getFlashcards,
    getSocraticQuestion,
    askSocraticTurn,
    uploadDocument,
    checkBackendStatus,
    computeProfile,
    saveTelemetry,
    getMockDiagnosticQuiz,
    getMockELI5,
    getMockFlashcards,
    getMockSocraticQuestion,
    isOnline
  };
})();
