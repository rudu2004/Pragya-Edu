# Pragya-Edu: Personalized Engagement & Curiosity Engine

> Built on top of KarmaYogi Pragya architecture — refactored for student-centered curiosity learning.

## Quick Start (Frontend Only — No Server Needed)
1. Open `index.html` in your browser
2. Click **Launch App** → fills in your name and topic → Enter
3. Navigate the 5 modules via the sidebar

## Quick Start (With Backend AI)
```bash
cd backend
pip install -r requirements.txt
python main.py
```
Then open `index.html` — the app auto-detects the backend and upgrades from mock data.

## 5 Modules
| # | Module | Page | Key Feature |
|---|--------|------|-------------|
| 1 | Curiosity & Vibe Check | `vibe-check.html` | MCQ diagnostic → Radar Chart |
| 2 | Curriculum Adapter | `curriculum-adapter.html` | PDF → ELI5 / Micro-Bites / Quiz |
| 3 | Socratic Mentor | `socratic-mentor.html` | AI chat + TTS voice |
| 4 | Classroom Analytics | `vibe-analytics.html` | Heatmap + at-risk detection |
| 5 | Dashboard / Low-BW | `dashboard.html` | Progress + offline mode |

## 6-Tier Fallback Architecture
```
Tier 1: Cloud API (FastAPI backend)
Tier 2: LocalStorage cache
Tier 3: Mock data generators (js/api.js)
Tier 4: Rule-based scoring engine
Tier 5: Static content pack (built-in)
Tier 6: Text-only minimal mode
```

## Tech Stack
- **Frontend**: HTML5, Vanilla CSS (glassmorphism dark theme), Vanilla JS
- **Charts**: Chart.js (radar, doughnut)
- **Icons**: Font Awesome 6
- **Fonts**: Outfit + Inter (Google Fonts)
- **Backend**: FastAPI + SQLite
- **Persistence**: localStorage + sessionStorage

## Architecture Notes
- Refactored from KarmaYogi Pragya (Hackathon 2026)
- Edge-first: All modules degrade gracefully offline
- No login required for demo (auth guard commented out)
