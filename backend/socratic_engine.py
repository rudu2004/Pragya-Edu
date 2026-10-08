"""
Pragya-Edu: socratic_engine.py
Adaptive Quiz (Module 1 - Vibe Check) and Socratic Dialogue (Module 3 - Mentor)
"""

import json
import random
from typing import Optional
from ai_engine import generate_adaptive_quiz, generate_socratic_question, analyze_vibe


# ══════════════════════════════════════════════════════════════════════
# MODULE 1: VIBE CHECK — Adaptive Diagnostic
# ══════════════════════════════════════════════════════════════════════

# Energy/mood to difficulty mapping
ENERGY_DIFFICULTY_MAP = {
    "high":    {"count": 10, "mix": [2, 5, 3]},   # Easy/Med/Hard
    "medium":  {"count": 8,  "mix": [3, 4, 1]},
    "low":     {"count": 6,  "mix": [4, 2, 0]},
    "burnout": {"count": 5,  "mix": [5, 0, 0]},
}

CURIOSITY_ARCHETYPES = {
    "explorer":  "You ask WHY before HOW. You love big-picture narratives.",
    "builder":   "You ask HOW. You learn best by doing and creating.",
    "connector": "You ask WHO and WHEN. You excel at relating concepts to people and contexts.",
    "analyzer":  "You ask WHAT precisely. You love data, structure, and deep mechanics.",
}

LEARNING_STYLES = ["visual", "auditory", "kinesthetic", "reading"]


def run_vibe_check(topic: str, energy: str = "medium",
                   curiosity_type: str = "explorer",
                   mood_score: float = 60.0) -> dict:
    """
    Generate an adaptive diagnostic quiz calibrated to the student's energy
    and curiosity archetype. Returns the quiz + a vibe profile summary.
    """
    energy = energy.lower() if energy in ENERGY_DIFFICULTY_MAP else "medium"
    config = ENERGY_DIFFICULTY_MAP[energy]

    # Get calibrated quiz
    quiz_result = generate_adaptive_quiz(topic, count=config["count"])

    # Infer curiosity archetype description
    archetype_desc = CURIOSITY_ARCHETYPES.get(curiosity_type,
                                               CURIOSITY_ARCHETYPES["explorer"])

    # Build vibe profile
    vibe_profile = {
        "energy":           energy,
        "curiosity_type":   curiosity_type,
        "curiosity_desc":   archetype_desc,
        "mood_score":       mood_score,
        "recommended_mode": _recommend_mode(energy, curiosity_type),
        "session_tip":      _session_tip(energy),
    }

    return {
        "topic":        topic,
        "vibe_profile": vibe_profile,
        "quiz":         quiz_result,
    }


def _recommend_mode(energy: str, curiosity: str) -> str:
    """Suggest the best learning mode based on energy + curiosity."""
    if energy == "burnout":
        return "flashcards"  # Lightest cognitive load
    if energy == "low":
        return "eli5"        # Story-driven, low pressure
    if curiosity == "builder":
        return "quiz"        # Active challenge
    if curiosity in ("explorer", "connector"):
        return "eli5"        # Narrative-first
    return "all"             # Analyzer gets the full suite


def _session_tip(energy: str) -> str:
    tips = {
        "high":    "🔥 Your energy is peak. Tackle the hardest material first — ride the wave.",
        "medium":  "⚡ Steady state. Use the Pomodoro: 25min focused study, 5min break.",
        "low":     "🌙 Low power detected. Stick to 1 concept. Victory lap over previously learned material.",
        "burnout": "🆘 Burnout mode. Do NOT push through. 1 flashcard. Then step away for 20 min.",
    }
    return tips.get(energy, "⚡ Stay consistent and trust the process.")


def get_vibe_analysis(session_data: dict) -> dict:
    """Analyze a completed vibe check and return a personalized readiness report."""
    scores = session_data.get("quiz_scores", [])
    return analyze_vibe(
        mood_score     = session_data.get("mood_score", 60.0),
        energy         = session_data.get("energy", "medium"),
        quiz_scores    = scores,
        curiosity_type = session_data.get("curiosity_type", "explorer"),
        topic          = session_data.get("topic", "General"),
    )


# ══════════════════════════════════════════════════════════════════════
# MODULE 3: SOCRATIC MENTOR — Multi-Turn Dialogue Engine
# ══════════════════════════════════════════════════════════════════════

# In-memory session store (replace with DB in production scale-up)
_SOCRATIC_SESSIONS: dict[str, list] = {}


def start_socratic_session(session_id: str, topic: str, depth: str = "surface") -> dict:
    """Initialize a new Socratic dialogue session (Turn 1)."""
    _SOCRATIC_SESSIONS[session_id] = []
    return _next_turn(session_id, topic, depth, user_answer="", turn=1)


def continue_socratic_session(session_id: str, topic: str,
                               depth: str, user_answer: str) -> dict:
    """Process user's answer and generate the next Socratic probe."""
    history = _SOCRATIC_SESSIONS.get(session_id, [])
    turn = len(history) + 1

    # Auto-escalate depth after turn 3
    if turn > 6:
        depth = "deep"
    elif turn > 3:
        depth = "medium" if depth == "surface" else depth

    result = _next_turn(session_id, topic, depth, user_answer, turn)

    # Log the user's answer in session history
    _SOCRATIC_SESSIONS[session_id].append({
        "turn":        turn,
        "user_answer": user_answer,
        "question":    result.get("question", ""),
        "depth":       depth,
    })
    return result


def _next_turn(session_id: str, topic: str, depth: str,
               user_answer: str, turn: int) -> dict:
    """Internal helper: fetch the next Socratic question from AI."""
    result = generate_socratic_question(
        topic=topic, depth=depth, user_answer=user_answer, turn=turn
    )
    result["turn"] = turn
    result["session_id"] = session_id
    result["depth_label"] = {
        "surface": "🌱 Surface — Foundation",
        "medium":  "🌿 Medium — Application",
        "deep":    "🌳 Deep — Philosophical",
    }.get(depth, "🌱 Surface")

    # Determine next depth recommendation
    if turn < 3:
        result["next_depth"] = "surface"
    elif turn < 6:
        result["next_depth"] = "medium"
    else:
        result["next_depth"] = "deep"

    return result


def get_session_summary(session_id: str, topic: str) -> dict:
    """Generate a reflection summary at the end of a Socratic session."""
    history = _SOCRATIC_SESSIONS.get(session_id, [])
    turn_count = len(history)

    if turn_count == 0:
        return {"summary": "No dialogue recorded yet.", "turns": 0}

    # Tally how many answers were substantive (>15 chars)
    substantive = sum(1 for h in history if len(h.get("user_answer", "")) > 15)
    mastery_pct = int((substantive / max(turn_count, 1)) * 100)

    depth_reached = history[-1].get("depth", "surface") if history else "surface"
    depth_badge = {
        "surface": "🌱 Surface Explorer",
        "medium":  "🌿 Applied Thinker",
        "deep":    "🌳 Deep Philosopher",
    }.get(depth_reached, "🌱 Surface Explorer")

    return {
        "session_id":   session_id,
        "topic":        topic,
        "turns":        turn_count,
        "mastery_pct":  mastery_pct,
        "depth_badge":  depth_badge,
        "depth_reached": depth_reached,
        "insight": (
            f"You engaged with {topic} across {turn_count} Socratic turns, "
            f"reaching {depth_badge} level. "
            f"Your response substantiveness: {mastery_pct}%."
        ),
        "next_challenge": (
            "Try the Curriculum Adapter to test this understanding against a document!"
            if mastery_pct >= 60 else
            "Revisit the Vibe Check module to reinforce the fundamentals before going deeper."
        ),
    }
