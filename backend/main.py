"""
Pragya-Edu Backend — FastAPI
Personalized Engagement & Curiosity Engine
6-Tier Fallback Architecture
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import json, random, datetime, os, sqlite3

app = FastAPI(
    title="Pragya-Edu API",
    description="Personalized Engagement & Curiosity Engine Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database ────────────────────────────────────────────────
DB_PATH = "pragya_edu.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event TEXT, session TEXT, data TEXT,
            timestamp REAL, created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            session TEXT PRIMARY KEY,
            name TEXT, topic TEXT, profile_json TEXT,
            engagement_score INTEGER, updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Pydantic Models ─────────────────────────────────────────
class QuizRequest(BaseModel):
    topic: str
    count: Optional[int] = 5
    difficulty: Optional[str] = "auto"

class ELI5Request(BaseModel):
    topic: str
    context: Optional[str] = "realworld"

class SocraticRequest(BaseModel):
    topic: str
    depth: Optional[str] = "surface"
    context: Optional[str] = ""

class TelemetryEntry(BaseModel):
    event: str
    session: Optional[str] = "anon"
    data: Optional[dict] = {}
    timestamp: Optional[float] = None

class ProfileSave(BaseModel):
    name: str
    topic: str
    profile: dict
    engagement_score: int

# ── Mock Generators ─────────────────────────────────────────
def generate_quiz(topic: str, count: int = 5) -> dict:
    templates = [
        {"q": f"What is the fundamental principle of {topic}?",
         "opts": ["Pattern recognition and systematic thinking", "Random trial and error", "Memorization without understanding", "Surface-level observation"],
         "ans": 0, "exp": f"{topic} fundamentally involves recognizing and applying patterns."},
        {"q": f"Which approach best supports mastery of {topic}?",
         "opts": ["Active problem-solving with feedback", "Passive reading only", "Last-minute cramming", "Skipping practice entirely"],
         "ans": 0, "exp": "Active learning with immediate feedback accelerates retention by 3x."},
        {"q": f"How does {topic} connect to real-world applications?",
         "opts": ["No real-world relevance", "Only in laboratory settings", "Through systematic pattern application", "Only theoretically relevant"],
         "ans": 2, "exp": f"{topic} provides frameworks for solving real problems systematically."},
        {"q": f"What distinguishes a novice from an expert in {topic}?",
         "opts": ["Speed of calculation", "Quantity memorized", "Depth of conceptual understanding", "Number of formulas known"],
         "ans": 2, "exp": "Experts understand the WHY, not just the HOW."},
        {"q": f"Which mindset accelerates learning {topic}?",
         "opts": ["Fixed mindset", "Growth mindset", "Avoidance strategy", "Over-confidence"],
         "ans": 1, "exp": "Growth mindset learners consistently outperform fixed mindset learners."},
    ]
    selected = random.sample(templates, min(count, len(templates)))
    return {
        "topic": topic,
        "questions": [{"id": i+1, **q} for i, q in enumerate(selected)],
        "generatedAt": datetime.datetime.utcnow().isoformat(),
        "source": "backend_mock"
    }

def generate_eli5(topic: str, context: str = "realworld") -> dict:
    openings = {
        "gaming": f"In a legendary RPG called {topic.split()[0]}-Quest, you face the ultimate challenge...",
        "scifi": f"Aboard the starship Curiosity in 2347, you must master {topic} to save the crew...",
        "sports": f"In the championship game of your life, understanding {topic} is the decisive skill...",
        "realworld": f"Imagine you are building your first startup and you desperately need to understand {topic}...",
        "stories": f"Once upon a time, a curious student stumbled upon the mysteries of {topic}..."
    }
    opening = openings.get(context, openings["realworld"])
    return {
        "topic": topic,
        "story": f"{opening}\n\nHere is the key insight: {topic} is all about understanding relationships and applying patterns. Every expert was once a beginner who asked 'why?' one more time than everyone else.\n\nRemember: Input → Process → Output. That is the universal flow underlying {topic}.",
        "context": context,
        "source": "backend_mock"
    }

SOCRATIC_QUESTIONS = {
    "surface": [
        "What do you already *think* you know about {topic}? Our assumptions are often our biggest obstacles.",
        "If you had to explain {topic} to a 10-year-old in 3 words, what would they be?",
        "What would happen to the world if {topic} suddenly ceased to exist?"
    ],
    "medium": [
        "Fascinating! But WHY does that happen? What is the underlying mechanism?",
        "You are on the right track. Now: where would this principle completely BREAK DOWN?",
        "How does this connect to something you already know really well?"
    ],
    "deep": [
        "What is the FINAL CAUSE here — the ultimate purpose or teleological end?",
        "If {topic} is so critical, why do most people fundamentally misunderstand it?",
        "What would you need to UNLEARN to grasp this at the deepest level?"
    ]
}

# ── API Routes ──────────────────────────────────────────────

@app.get("/")
def root():
    return {"name": "Pragya-Edu API", "version": "1.0.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}

@app.post("/api/quiz/generate")
def create_quiz(req: QuizRequest):
    return generate_quiz(req.topic, req.count)

@app.post("/api/content/eli5")
def get_eli5(req: ELI5Request):
    return generate_eli5(req.topic, req.context)

@app.get("/api/content/flashcards")
def get_flashcards(topic: str):
    cards = [
        {"id": 1, "front": f"What is {topic}?", "back": f"The systematic study of relationships and patterns in this domain."},
        {"id": 2, "front": "Core Rule", "back": "Input → Process → Output. Every instance follows this universal flow."},
        {"id": 3, "front": "Real-World Example", "back": f"{topic} appears when you observe cause-and-effect in daily life."},
        {"id": 4, "front": "Common Misconception", "back": "Most learners confuse surface features with underlying principles. Focus on the WHY."},
        {"id": 5, "front": "Connections", "back": f"{topic} links to foundational principles and bridges to advanced topics."},
        {"id": 6, "front": "One-Line Summary", "back": f'"{topic} is the art of seeing hidden order and using it to predict and create."'},
    ]
    return {"topic": topic, "cards": cards, "source": "backend"}

@app.post("/api/mentor/socratic")
def get_socratic(req: SocraticRequest):
    pool = SOCRATIC_QUESTIONS.get(req.depth, SOCRATIC_QUESTIONS["surface"])
    q = random.choice(pool).replace("{topic}", req.topic)
    return {"question": q, "depth": req.depth, "topic": req.topic}

@app.post("/api/telemetry")
def save_telemetry(entry: TelemetryEntry):
    conn = get_db()
    conn.execute(
        "INSERT INTO telemetry (event, session, data, timestamp, created_at) VALUES (?,?,?,?,?)",
        (entry.event, entry.session, json.dumps(entry.data),
         entry.timestamp or datetime.datetime.utcnow().timestamp(),
         datetime.datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return {"status": "saved"}

@app.post("/api/profile/save")
def save_profile(p: ProfileSave):
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO profiles (session, name, topic, profile_json, engagement_score, updated_at)
        VALUES (?,?,?,?,?,?)
    """, (p.name, p.name, p.topic, json.dumps(p.profile), p.engagement_score,
          datetime.datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return {"status": "saved"}

@app.get("/api/analytics/class")
def class_analytics():
    """Returns mock classroom analytics data"""
    students = []
    names = ["Aarav", "Priya", "Rohan", "Ananya", "Kabir", "Sneha", "Dev", "Meera"]
    energies = ["high", "burnout", "medium", "low"]
    for i, name in enumerate(names):
        energy = random.choice(energies)
        quiz = random.randint(10, 95)
        students.append({
            "name": name,
            "energy": energy,
            "quizScore": quiz,
            "sessions": random.randint(1, 16),
            "risk": "Critical" if quiz < 30 else "Moderate" if quiz < 50 else "Low"
        })
    return {"students": students, "classVibeScore": random.randint(40, 80)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
