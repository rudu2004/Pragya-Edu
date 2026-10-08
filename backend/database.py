"""
Pragya-Edu: database.py
SQLite schema + seeding for pragya_edu.db
Covers: user_profiles, sessions, classroom_analytics, telemetry
"""
import sqlite3
import hashlib
import random
import os
from datetime import datetime, timedelta

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pragya_edu.db")


# ─────────────────────────────────────────────────────────────────────
# Connection helper
# ─────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────────────
# Schema Initialization
# ─────────────────────────────────────────────────────────────────────
def init_db():
    conn = get_db()
    cur = conn.cursor()

    # ── User Profiles (Module 1: Vibe Check results stored here) ──────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    UNIQUE NOT NULL,
            name        TEXT    NOT NULL DEFAULT 'Student',
            topic       TEXT    NOT NULL DEFAULT 'General',
            energy_type TEXT    NOT NULL DEFAULT 'medium',  -- high / medium / low / burnout
            curiosity   TEXT    NOT NULL DEFAULT 'explorer', -- explorer / builder / connector / analyzer
            mood_score  REAL    NOT NULL DEFAULT 50.0,       -- 0-100
            learning_style TEXT DEFAULT 'visual',            -- visual / auditory / kinesthetic / reading
            vibe_json   TEXT    DEFAULT '{}',
            created_at  TEXT    DEFAULT (datetime('now')),
            updated_at  TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Sessions (Module 2: Curriculum Adapter uploads/generations) ───
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id          TEXT    NOT NULL,
            filename            TEXT,
            file_content_hash   TEXT,
            topic               TEXT,
            generated_story     TEXT,
            generated_flashcards TEXT,
            generated_quiz      TEXT,
            mode                TEXT    DEFAULT 'eli5',  -- eli5 / flashcards / quiz / all
            created_at          TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Classroom Analytics (Module 4: Vibe Analytics) ────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS classroom_analytics (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            student_name    TEXT NOT NULL,
            energy          TEXT NOT NULL DEFAULT 'medium',
            quiz_score      REAL NOT NULL DEFAULT 0.0,
            sessions_count  INTEGER DEFAULT 1,
            burnout_risk    TEXT DEFAULT 'Low',   -- Low / Moderate / Critical
            peer_match_tag  TEXT,                  -- curiosity archetype for peer matching
            updated_at      TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Telemetry (raw event log for all modules) ─────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event       TEXT NOT NULL,
            session_id  TEXT DEFAULT 'anon',
            data        TEXT DEFAULT '{}',
            timestamp   REAL,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    # ── Socratic Sessions (Module 3 multi-turn dialogue log) ──────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS socratic_sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            topic       TEXT NOT NULL,
            depth       TEXT NOT NULL DEFAULT 'surface',   -- surface / medium / deep
            turn        INTEGER DEFAULT 1,
            question    TEXT NOT NULL,
            user_answer TEXT,
            followup    TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()

    # Seed demo classroom data if empty
    cur.execute("SELECT COUNT(*) FROM classroom_analytics")
    if cur.fetchone()[0] == 0:
        _seed_classroom_data(cur, conn)

    conn.close()


# ─────────────────────────────────────────────────────────────────────
# Seed helpers
# ─────────────────────────────────────────────────────────────────────
def _seed_classroom_data(cur, conn):
    """Seed 12 demo students for the Vibe Analytics classroom heatmap."""
    random.seed(42)
    students = [
        ("Aarav Mehta",   "high",   88, 12, "builder"),
        ("Priya Sharma",  "burnout", 32,  3, "connector"),
        ("Rohan Das",     "medium",  65,  8, "analyzer"),
        ("Ananya Singh",  "high",   91, 14, "explorer"),
        ("Kabir Nair",    "low",    45,  5, "builder"),
        ("Sneha Patel",   "burnout", 28,  2, "connector"),
        ("Dev Joshi",     "medium",  72, 10, "analyzer"),
        ("Meera Pillai",  "high",   83, 11, "explorer"),
        ("Arjun Verma",   "low",    38,  4, "builder"),
        ("Kavya Iyer",    "medium",  60,  9, "connector"),
        ("Rishi Gupta",   "burnout", 22,  1, "analyzer"),
        ("Tara Malhotra", "high",   79,  7, "explorer"),
    ]

    for (name, energy, score, scount, tag) in students:
        risk = "Critical" if score < 30 else ("Moderate" if score < 50 else "Low")
        cur.execute("""
            INSERT INTO classroom_analytics
                (session_id, student_name, energy, quiz_score, sessions_count, burnout_risk, peer_match_tag)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("demo_class_001", name, energy, float(score), scount, risk, tag))

    conn.commit()


def _hash_password(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────
# CRUD helpers used by main.py
# ─────────────────────────────────────────────────────────────────────
def upsert_profile(session_id: str, data: dict):
    conn = get_db()
    conn.execute("""
        INSERT INTO user_profiles (session_id, name, topic, energy_type, curiosity,
                                   mood_score, learning_style, vibe_json, updated_at)
        VALUES (:sid, :name, :topic, :energy, :curiosity,
                :mood, :style, :vibe, datetime('now'))
        ON CONFLICT(session_id) DO UPDATE SET
            name=excluded.name, topic=excluded.topic,
            energy_type=excluded.energy_type, curiosity=excluded.curiosity,
            mood_score=excluded.mood_score, learning_style=excluded.learning_style,
            vibe_json=excluded.vibe_json, updated_at=excluded.updated_at
    """, {
        "sid":     session_id,
        "name":    data.get("name", "Student"),
        "topic":   data.get("topic", "General"),
        "energy":  data.get("energy_type", "medium"),
        "curiosity": data.get("curiosity", "explorer"),
        "mood":    data.get("mood_score", 50.0),
        "style":   data.get("learning_style", "visual"),
        "vibe":    data.get("vibe_json", "{}"),
    })
    conn.commit()
    conn.close()


def log_telemetry(event: str, session_id: str, data: dict, ts: float = None):
    import json, time
    conn = get_db()
    conn.execute(
        "INSERT INTO telemetry (event, session_id, data, timestamp) VALUES (?, ?, ?, ?)",
        (event, session_id, json.dumps(data), ts or time.time())
    )
    conn.commit()
    conn.close()


def save_session_output(session_id: str, topic: str, filename: str,
                        story: str = None, flashcards: str = None,
                        quiz: str = None, mode: str = "all"):
    conn = get_db()
    conn.execute("""
        INSERT INTO sessions (session_id, filename, topic, generated_story,
                              generated_flashcards, generated_quiz, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (session_id, filename, topic, story, flashcards, quiz, mode))
    conn.commit()
    conn.close()


def get_classroom_analytics(session_id: str = "demo_class_001") -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM classroom_analytics WHERE session_id = ?", (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
