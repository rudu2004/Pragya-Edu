"""
Pragya-Edu Backend — FastAPI Server
Personalized Engagement & Curiosity Engine
6-Tier Fallback Architecture
Routes for all 4 Modules + Dashboard + Analytics
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import time
import datetime
import random
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database import (
    init_db, get_db, upsert_profile, log_telemetry,
    save_session_output, get_classroom_analytics
)
from ai_engine import (
    generate_adaptive_quiz, generate_eli5, generate_flashcards,
    generate_socratic_question, analyze_vibe, generate_response,
    GEMINI_API_KEY, GROQ_API_KEY, HF_TOKEN
)
from document_processor import process_document, extract_text, extract_topic
from socratic_engine import (
    run_vibe_check, get_vibe_analysis,
    start_socratic_session, continue_socratic_session, get_session_summary
)

# ── Init DB on startup ─────────────────────────────────────────────────
init_db()

# ── App Setup ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Pragya-Edu API",
    description="Personalized Engagement & Curiosity Engine — 6-Tier AI Fallback Backend",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend static files ────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

if os.path.isdir(FRONTEND_DIR):
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# ══════════════════════════════════════════════════════════════════════
# Pydantic Request Models
# ══════════════════════════════════════════════════════════════════════

class VibeCheckRequest(BaseModel):
    topic: str
    energy: Optional[str] = "medium"          # high / medium / low / burnout
    curiosity_type: Optional[str] = "explorer" # explorer / builder / connector / analyzer
    mood_score: Optional[float] = 60.0
    session_id: Optional[str] = "anon"

class VibeAnalysisRequest(BaseModel):
    topic: str
    energy: Optional[str] = "medium"
    curiosity_type: Optional[str] = "explorer"
    mood_score: Optional[float] = 60.0
    quiz_scores: Optional[List[float]] = []
    session_id: Optional[str] = "anon"

class QuizRequest(BaseModel):
    topic: str
    count: Optional[int] = 10
    difficulty: Optional[str] = "adaptive"

class ELI5Request(BaseModel):
    topic: str
    context: Optional[str] = ""
    style: Optional[str] = "gaming"    # gaming / scifi / sports / detective / realworld

class FlashcardRequest(BaseModel):
    topic: str
    context: Optional[str] = ""
    count: Optional[int] = 8

class SocraticStartRequest(BaseModel):
    topic: str
    depth: Optional[str] = "surface"
    session_id: Optional[str] = "anon"

class SocraticContinueRequest(BaseModel):
    topic: str
    depth: Optional[str] = "surface"
    user_answer: str
    session_id: Optional[str] = "anon"

class TelemetryEntry(BaseModel):
    event: str
    session_id: Optional[str] = "anon"
    data: Optional[dict] = {}
    timestamp: Optional[float] = None

class ProfileSave(BaseModel):
    session_id: str
    name: Optional[str] = "Student"
    topic: Optional[str] = "General"
    energy_type: Optional[str] = "medium"
    curiosity: Optional[str] = "explorer"
    mood_score: Optional[float] = 60.0
    learning_style: Optional[str] = "visual"
    vibe_json: Optional[str] = "{}"


# ══════════════════════════════════════════════════════════════════════
# Helper: detect active AI tier
# ══════════════════════════════════════════════════════════════════════
def _detect_active_tier() -> dict:
    """Quick-check which AI tier is alive right now."""
    if GEMINI_API_KEY:
        return {"tier": 1, "label": "Gemini Flash", "status": "🟢 ONLINE"}
    if GROQ_API_KEY:
        return {"tier": 3, "label": "Groq LLaMA 70B", "status": "🟡 FALLBACK"}
    if HF_TOKEN:
        return {"tier": 5, "label": "HuggingFace Mistral", "status": "🟡 FALLBACK"}
    return {"tier": 6, "label": "Local Heuristic", "status": "🔴 OFFLINE MODE"}


# ══════════════════════════════════════════════════════════════════════
# SYSTEM ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """Serve the landing page."""
    index_path = os.path.join(ROOT_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"name": "Pragya-Edu API", "version": "2.0.0", "status": "running"}


@app.get("/api/status")
def api_status():
    """
    Returns system health + active AI tier.
    Used by frontend retro arcade status bars.
    """
    tier_info = _detect_active_tier()
    return {
        "status":       "healthy",
        "timestamp":    datetime.datetime.utcnow().isoformat() + "Z",
        "active_tier":  tier_info["tier"],
        "tier_label":   tier_info["label"],
        "tier_status":  tier_info["status"],
        "api_keys": {
            "gemini": bool(GEMINI_API_KEY),
            "groq":   bool(GROQ_API_KEY),
            "hf":     bool(HF_TOKEN),
        },
        "modules": {
            "vibe_check":        "/api/vibe-check",
            "curriculum_adapter": "/api/curriculum-adapter",
            "socratic_mentor":   "/api/socratic-mentor",
            "vibe_analytics":    "/api/vibe-analytics",
        }
    }


@app.get("/api/health")
def health():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}


# ══════════════════════════════════════════════════════════════════════
# MODULE 1: VIBE CHECK — /api/vibe-check
# ══════════════════════════════════════════════════════════════════════

@app.get("/vibe-check")
def serve_vibe_check():
    f = os.path.join(FRONTEND_DIR, "vibe-check.html")
    return FileResponse(f) if os.path.isfile(f) else {"error": "not found"}


@app.post("/api/vibe-check/start")
def vibe_check_start(req: VibeCheckRequest):
    """
    Run a full vibe check: detect energy/curiosity state, return adaptive quiz.
    """
    result = run_vibe_check(
        topic         = req.topic,
        energy        = req.energy,
        curiosity_type = req.curiosity_type,
        mood_score    = req.mood_score,
    )
    log_telemetry("vibe_check_start", req.session_id,
                  {"topic": req.topic, "energy": req.energy}, time.time())
    return result


@app.post("/api/vibe-check/analyze")
def vibe_check_analyze(req: VibeAnalysisRequest):
    """
    Analyze completed vibe check — return readiness score + personalized insight.
    """
    result = get_vibe_analysis({
        "topic":         req.topic,
        "energy":        req.energy,
        "curiosity_type": req.curiosity_type,
        "mood_score":    req.mood_score,
        "quiz_scores":   req.quiz_scores,
    })

    # Persist profile
    upsert_profile(req.session_id, {
        "name":          "Student",
        "topic":         req.topic,
        "energy_type":   req.energy,
        "curiosity":     req.curiosity_type,
        "mood_score":    req.mood_score,
        "learning_style": "visual",
        "vibe_json":     json.dumps(result),
    })
    log_telemetry("vibe_check_complete", req.session_id,
                  {"readiness": result.get("readiness_score")}, time.time())
    return result


@app.post("/api/quiz/generate")
def create_quiz(req: QuizRequest):
    """Generate an adaptive MCQ quiz on any topic."""
    return generate_adaptive_quiz(req.topic, req.count)


# ══════════════════════════════════════════════════════════════════════
# MODULE 2: CURRICULUM ADAPTER — /api/curriculum-adapter
# ══════════════════════════════════════════════════════════════════════

@app.get("/curriculum-adapter")
def serve_curriculum():
    f = os.path.join(FRONTEND_DIR, "curriculum-adapter.html")
    return FileResponse(f) if os.path.isfile(f) else {"error": "not found"}


@app.post("/api/curriculum-adapter/upload")
async def upload_document(
    file: UploadFile = File(...),
    mode:  str = Form("all"),      # eli5 / flashcards / quiz / all
    style: str = Form("gaming"),   # gaming / scifi / sports / detective / realworld
    session_id: str = Form("anon"),
):
    """
    Upload PDF/TXT/DOCX → extract text → run AI pipeline.
    Returns ELI5 story, flashcards, and/or quiz depending on mode.
    """
    ALLOWED = {".pdf", ".txt", ".docx", ".doc"}
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in ALLOWED:
        raise HTTPException(400, f"Unsupported format '{ext}'. Upload PDF, TXT, or DOCX.")

    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(413, "File too large. Maximum 10 MB.")

    result = process_document(file_bytes, file.filename, mode=mode, style=style)

    # Persist session output
    save_session_output(
        session_id  = session_id,
        topic       = result.get("topic", file.filename),
        filename    = file.filename,
        story       = json.dumps(result.get("eli5")) if result.get("eli5") else None,
        flashcards  = json.dumps(result.get("flashcards")) if result.get("flashcards") else None,
        quiz        = json.dumps(result.get("quiz")) if result.get("quiz") else None,
        mode        = mode,
    )
    log_telemetry("document_upload", session_id,
                  {"filename": file.filename, "mode": mode, "topic": result.get("topic")},
                  time.time())
    return result


@app.post("/api/curriculum-adapter/eli5")
@app.post("/api/content/eli5")
def get_eli5(req: ELI5Request):
    """Generate a gamified ELI5 story for any topic (no file upload required)."""
    return generate_eli5(req.topic, req.context, req.style)


@app.get("/api/curriculum-adapter/flashcards")
def get_flashcards(topic: str, count: int = 8):
    """Generate flashcards for a topic (GET convenience endpoint)."""
    return generate_flashcards(topic, count=count)


@app.post("/api/curriculum-adapter/flashcards")
def post_flashcards(req: FlashcardRequest):
    """Generate flashcards with optional document context."""
    return generate_flashcards(req.topic, req.context, req.count)


# ══════════════════════════════════════════════════════════════════════
# MODULE 3: SOCRATIC MENTOR — /api/socratic-mentor
# ══════════════════════════════════════════════════════════════════════

@app.get("/socratic-mentor")
def serve_socratic():
    f = os.path.join(FRONTEND_DIR, "socratic-mentor.html")
    return FileResponse(f) if os.path.isfile(f) else {"error": "not found"}


@app.post("/api/socratic-mentor/start")
def socratic_start(req: SocraticStartRequest):
    """Start a new Socratic dialogue session (Turn 1)."""
    result = start_socratic_session(req.session_id, req.topic, req.depth)
    log_telemetry("socratic_start", req.session_id,
                  {"topic": req.topic, "depth": req.depth}, time.time())
    return result


@app.post("/api/socratic-mentor/continue")
def socratic_continue(req: SocraticContinueRequest):
    """Submit an answer and receive the next Socratic probe."""
    result = continue_socratic_session(
        req.session_id, req.topic, req.depth, req.user_answer
    )
    log_telemetry("socratic_turn", req.session_id,
                  {"topic": req.topic, "depth": req.depth,
                   "answer_length": len(req.user_answer)}, time.time())
    return result


@app.get("/api/socratic-mentor/summary")
def socratic_summary(session_id: str, topic: str):
    """Get a reflection summary for a completed Socratic session."""
    return get_session_summary(session_id, topic)


@app.post("/api/mentor/socratic")
def legacy_socratic(req: SocraticStartRequest):
    """Legacy endpoint — backwards compatible."""
    return start_socratic_session(req.session_id, req.topic, req.depth)


# ══════════════════════════════════════════════════════════════════════
# MODULE 4: VIBE ANALYTICS — /api/vibe-analytics
# ══════════════════════════════════════════════════════════════════════

@app.get("/vibe-analytics")
def serve_analytics():
    f = os.path.join(FRONTEND_DIR, "vibe-analytics.html")
    return FileResponse(f) if os.path.isfile(f) else {"error": "not found"}


@app.get("/api/vibe-analytics/classroom")
def classroom_analytics(session_id: str = "demo_class_001"):
    """
    Returns classroom analytics:
    - Per-student energy, quiz scores, burnout risk
    - Class vibe score aggregate
    - Peer-matching recommendations
    """
    students = get_classroom_analytics(session_id)

    # Augment with peer-match recommendations
    for s in students:
        match_map = {
            "explorer":  "analyzer",
            "analyzer":  "connector",
            "connector": "builder",
            "builder":   "explorer",
        }
        s["peer_recommendation"] = match_map.get(s.get("peer_match_tag", "explorer"), "explorer")

    # Class aggregate KPIs
    if students:
        avg_score     = round(sum(s["quiz_score"] for s in students) / len(students), 1)
        burnout_count = sum(1 for s in students if s["burnout_risk"] == "Critical")
        high_energy   = sum(1 for s in students if s["energy"] == "high")
        class_vibe    = max(10, min(100, int(avg_score * 0.7 + high_energy * 3 - burnout_count * 5)))
    else:
        avg_score = burnout_count = high_energy = 0
        class_vibe = 50

    return {
        "session_id":       session_id,
        "students":         students,
        "class_kpis": {
            "class_vibe_score":   class_vibe,
            "avg_quiz_score":     avg_score,
            "total_students":     len(students),
            "burnout_alerts":     burnout_count,
            "high_energy_count":  high_energy,
        },
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/vibe-analytics/trends")
def analytics_trends():
    """Return mock weekly engagement trends for sparkline charts."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "labels":          days,
        "engagement":      [random.randint(55, 95) for _ in days],
        "quiz_completion": [random.randint(40, 90) for _ in days],
        "burnout_risk":    [random.randint(10, 45) for _ in days],
    }


@app.get("/api/vibe-analytics/isolation-heatmap")
def isolation_heatmap():
    """Return peer-isolation risk heatmap data."""
    students = get_classroom_analytics()
    archetype_counts: dict = {}
    for s in students:
        tag = s.get("peer_match_tag", "explorer")
        archetype_counts[tag] = archetype_counts.get(tag, 0) + 1

    # Identify isolated archetypes (count == 1 → no peer match)
    isolated = [k for k, v in archetype_counts.items() if v == 1]
    return {
        "archetype_distribution": archetype_counts,
        "isolated_archetypes":    isolated,
        "recommendation":         (
            f"Students with archetype(s) [{', '.join(isolated)}] have no peer matches. "
            "Consider facilitating cross-archetype study groups."
            if isolated else
            "All curiosity archetypes have at least one peer match. Excellent diversity!"
        )
    }


# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITY ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.post("/api/telemetry")
def save_telemetry(entry: TelemetryEntry):
    log_telemetry(entry.event, entry.session_id, entry.data or {},
                  entry.timestamp or time.time())
    return {"status": "saved"}


@app.post("/api/profile/save")
def save_profile(p: ProfileSave):
    upsert_profile(p.session_id, p.dict())
    return {"status": "saved", "session_id": p.session_id}


@app.get("/api/profile/{session_id}")
def get_profile(session_id: str):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM user_profiles WHERE session_id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, f"Profile '{session_id}' not found")
    return dict(row)


@app.get("/api/content/flashcards")
def legacy_flashcards(topic: str):
    """Legacy GET endpoint — backwards compatible."""
    return generate_flashcards(topic)


# ── Frontend page routes ───────────────────────────────────────────────
@app.get("/login")
def serve_login():
    f = os.path.join(FRONTEND_DIR, "login.html")
    return FileResponse(f) if os.path.isfile(f) else {"error": "not found"}


@app.get("/dashboard")
def serve_dashboard():
    f = os.path.join(FRONTEND_DIR, "dashboard.html")
    return FileResponse(f) if os.path.isfile(f) else {"error": "not found"}


# ══════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
