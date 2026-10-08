"""
Pragya-Edu: ai_engine.py
6-Tier Resilient AI Engine — adapted from KarmaYogi Pragya
Tier 1: Gemini 1.5 Flash (Google AI Studio)
Tier 2: Gemini 2.0 Flash (Google AI Studio)
Tier 3: LLaMA 3.3 70B Versatile (Groq)
Tier 4: LLaMA 3.1 8B Instant (Groq)
Tier 5: HuggingFace Serverless Inference (Mistral-7B)
Tier 6: Local Heuristic Engine (100% offline, never crashes)
"""

import os
import re
import json
import random
import requests
import httpx
import numpy as np
from typing import Any, Dict, List, Optional

# ── Load env ───────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
HF_TOKEN       = os.getenv("HF_TOKEN", "")

# ── HuggingFace endpoints ──────────────────────────────────────────────
HF_EMBED_URL   = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_TEXT_URL    = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# ══════════════════════════════════════════════════════════════════════
# UTILITY: JSON Cleaning
# ══════════════════════════════════════════════════════════════════════
def clean_json_response(raw: str) -> str:
    """Robustly extract clean JSON from raw LLM output (fenced blocks, preambles, LaTeX)."""
    if not raw or not isinstance(raw, str):
        return "{}"
    text = raw.strip()

    # Strip markdown code fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    else:
        # Trim to outermost braces/brackets
        fb, lb = text.find("{"), text.rfind("}")
        br, lb2 = text.find("["), text.rfind("]")
        if fb != -1 and (br == -1 or fb < br):
            if lb > fb:
                text = text[fb:lb + 1]
        elif br != -1 and lb2 > br:
            text = text[br:lb2 + 1]

    # Escape bare backslashes (LaTeX etc.) and strip trailing commas
    text = re.sub(r'\\(?!["\\/bfnrtu]|u[0-9a-fA-F]{4})', r'\\\\', text)
    text = re.sub(r',\s*([\]}])', r'\1', text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════
# EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════
def _fallback_embedding(text: str) -> list:
    """Deterministic 384-d hash-bucket pseudo-embedding (offline fallback)."""
    vec = [0.0] * 384
    for word in (text or "").lower().split():
        vec[abs(hash(word)) % 384] += 1.0
    norm = np.linalg.norm(vec)
    return (np.array(vec) / (norm if norm > 0 else 1.0)).tolist()


def get_embedding(text: str) -> list:
    """Return 384-d sentence embedding via HF or deterministic fallback."""
    if not text or not text.strip():
        return [0.0] * 384
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.post(HF_EMBED_URL, headers=headers,
                       json={"inputs": [text.strip()], "options": {"wait_for_model": True}})
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0] if isinstance(data[0], list) else data
    except Exception as e:
        print(f"[EMBED] HF fallback: {e}")
    return _fallback_embedding(text)


def cosine_similarity(a: list, b: list) -> float:
    va, vb = np.array(a), np.array(b)
    n = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / n) if n > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# TIER IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════

def _call_gemini(prompt: str, model: str = "gemini-2.5-flash",
                 temperature: float = 0.3, json_mode: bool = True,
                 timeout: float = 30.0) -> str:
    """Tier 1 / 2: Google Gemini via google-generativeai SDK."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    gen_cfg: dict = {"temperature": temperature}
    if json_mode:
        gen_cfg["response_mime_type"] = "application/json"

    fallback_models = [model, "gemini-2.5-flash", "gemini-flash-latest", "gemini-3.8-flash"]
    last_err = None
    for m in dict.fromkeys(fallback_models):  # deduplicated order-preserving
        try:
            mdl = genai.GenerativeModel(m, generation_config=gen_cfg)
            resp = mdl.generate_content(prompt, request_options={"timeout": timeout})
            raw = resp.text.strip() if resp and resp.text else ""
            if raw:
                return clean_json_response(raw) if json_mode else raw
        except Exception as e:
            last_err = e
            if json_mode and "response_mime_type" in str(e).lower():
                try:
                    mdl2 = genai.GenerativeModel(m, generation_config={"temperature": temperature})
                    resp2 = mdl2.generate_content(prompt, request_options={"timeout": timeout})
                    raw2 = resp2.text.strip() if resp2 and resp2.text else ""
                    if raw2:
                        return clean_json_response(raw2)
                except Exception as e2:
                    last_err = e2
    raise RuntimeError(f"Gemini ({model}) all candidates failed: {last_err}")


def _call_groq(prompt: str, model: str = "openai/gpt-oss-120b",
               temperature: float = 0.3, json_mode: bool = True,
               timeout: float = 25.0) -> str:
    """Tier 3 / 4: Groq API via HTTP."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    candidates = [model, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    last_err = None
    for m in dict.fromkeys(candidates):
        payload: dict = {
            "model": m,
            "messages": [
                {"role": "system",
                 "content": ("You are Pragya, an expert AI tutor. Output raw valid JSON only."
                             if json_mode else
                             "You are Pragya, an expert AI learning assistant.")},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.ok:
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                if json_mode:
                    raw = clean_json_response(raw)
                return raw
            elif resp.status_code == 400 and "json_validate_failed" in resp.text:
                payload.pop("response_format", None)
                resp2 = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp2.ok:
                    return clean_json_response(resp2.json()["choices"][0]["message"]["content"])
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"Groq all candidates failed: {last_err}")


def _call_hf_text(prompt: str, timeout: float = 20.0) -> str:
    """Tier 5: HuggingFace Serverless Inference (Mistral-7B)."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(HF_TEXT_URL, headers=headers,
                       json={"inputs": prompt,
                             "parameters": {"max_new_tokens": 1024, "return_full_text": False},
                             "options": {"wait_for_model": True}})
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and "generated_text" in data[0]:
                    return data[0]["generated_text"].strip()
    except Exception as e:
        print(f"[HF_TEXT] Tier 5 failed: {e}")
    return ""


def _local_fallback(prompt: str, json_mode: bool = True) -> str:
    """Tier 6: Local deterministic heuristic engine — never crashes, 100% offline."""
    print("[LLM] Tier 6: Local heuristic engine engaged.")
    p = prompt.lower()

    # ── Quiz / MCQ ────────────────────────────────────────────────────
    if any(k in p for k in ("quiz", "mcq", "question", "assessment", "diagnostic")):
        topic = re.search(r'topic["\s:]+([^\n"]+)', prompt)
        t = topic.group(1).strip().title() if topic else "the subject"
        questions = []
        templates = [
            (f"What is the core principle of {t}?",
             ["Pattern recognition", "Random trial and error", "Rote memorization", "Surface observation"],
             0, f"{t} fundamentally involves recognizing patterns.", "Easy"),
            (f"Which approach best supports mastery of {t}?",
             ["Active problem-solving with feedback", "Passive reading only", "Last-minute cramming", "Skipping practice"],
             0, "Active learning with feedback accelerates retention by 3x.", "Easy"),
            (f"How does {t} connect to real-world applications?",
             ["Through systematic pattern application", "Only in laboratories", "No real-world relevance", "Theoretically only"],
             0, f"{t} provides frameworks for solving real problems.", "Medium"),
            (f"What distinguishes a novice from an expert in {t}?",
             ["Depth of conceptual understanding", "Speed of calculation", "Quantity memorized", "Formulas known"],
             0, "Experts understand the WHY, not just the HOW.", "Medium"),
            (f"Which mindset accelerates learning {t}?",
             ["Growth mindset", "Fixed mindset", "Avoidance strategy", "Over-confidence"],
             0, "Growth mindset learners consistently outperform fixed mindset peers.", "Medium"),
            (f"What is the fundamental unit of analysis in {t}?",
             ["The underlying relationship between variables", "Surface features only", "Memorized definitions", "External references"],
             0, f"In {t}, the relationship between variables is foundational.", "Hard"),
            (f"How does feedback improve outcomes in {t}?",
             ["By enabling rapid error correction and adaptation", "By confirming existing knowledge", "By reducing the need to practice", "By replacing direct instruction"],
             0, "Feedback loops are the fastest route to mastery in any domain.", "Hard"),
            (f"What is the 'why' behind studying {t}?",
             ["To build transferable problem-solving frameworks", "To pass exams only", "To memorize facts", "To satisfy requirements"],
             0, "Deep learning in {t} builds frameworks you apply across contexts.", "Easy"),
            (f"In {t}, what role does deliberate practice play?",
             ["Critical — targets weaknesses systematically", "Minimal — talent is innate", "None — exposure suffices", "Counterproductive"],
             0, "Deliberate practice with feedback is the engine of expertise.", "Hard"),
            (f"What separates a great explanation of {t} from a poor one?",
             ["Connecting abstractions to concrete examples", "Using complex jargon", "Maximum length and detail", "Avoiding analogies"],
             0, "Great explanations anchor abstraction in concrete, relatable examples.", "Medium"),
        ]
        for i, (q, opts, ans, exp, diff) in enumerate(templates[:10]):
            questions.append({
                "id": i + 1, "question": q, "options": opts,
                "correct_option": opts[ans], "explanation": exp,
                "difficulty": diff, "competency_tag": f"{t} Core"
            })
        return json.dumps({"questions": questions, "source": "tier6_heuristic"})

    # ── ELI5 / Story ──────────────────────────────────────────────────
    if any(k in p for k in ("eli5", "story", "explain", "narrative", "simple")):
        topic = re.search(r'topic["\s:]+([^\n"]+)', prompt)
        t = topic.group(1).strip() if topic else "this concept"
        return json.dumps({
            "story": (
                f"🎮 LEVEL 1 UNLOCKED: {t.upper()} QUEST\n\n"
                f"Imagine you just picked up the legendary textbook of {t}. "
                f"At first glance, it looks intimidating — like a final boss. But here's the secret: "
                f"every expert was once a confused beginner who asked 'WHY?' one more time than everyone else.\n\n"
                f"The core loop of {t} is simple: **Input → Process → Output**. "
                f"Every concept, every formula, every technique is just a more sophisticated version of this loop.\n\n"
                f"🏆 ACHIEVEMENT UNLOCKED: You now understand the skeleton of {t}. "
                f"The rest is just filling in the details — one level at a time."
            ),
            "topic": t,
            "source": "tier6_heuristic"
        })

    # ── Flashcards ────────────────────────────────────────────────────
    if any(k in p for k in ("flashcard", "card", "concept", "summary", "microbite")):
        topic = re.search(r'topic["\s:]+([^\n"]+)', prompt)
        t = topic.group(1).strip() if topic else "the topic"
        cards = [
            {"front": f"What is {t}?", "back": f"The systematic study of relationships and patterns in {t}."},
            {"front": "Universal Learning Law", "back": "Input → Process → Output. Every concept follows this flow."},
            {"front": f"Real-World Application of {t}", "back": f"{t} appears wherever cause-and-effect governs outcomes."},
            {"front": "Novice vs Expert", "back": "Experts know the WHY. Novices know the WHAT. Aim for the WHY."},
            {"front": f"One-Line Summary", "back": f'"{t} is the art of seeing hidden order and using it to predict and create."'},
            {"front": "How to Master Anything", "back": "Deliberate practice + immediate feedback + spaced repetition."},
        ]
        return json.dumps({"cards": cards, "topic": t, "source": "tier6_heuristic"})

    # ── Socratic Question ─────────────────────────────────────────────
    if any(k in p for k in ("socratic", "provoke", "probe", "think", "question")):
        topic = re.search(r'topic["\s:]+([^\n"]+)', prompt)
        t = topic.group(1).strip() if topic else "this idea"
        depth = "deep" if "deep" in p else ("medium" if "medium" in p else "surface")
        pool = {
            "surface": [
                f"What do you *think* you already know about {t}? (Warning: our assumptions are often our biggest obstacles.)",
                f"If you had to explain {t} to a 10-year-old using only 3 words, what would they be?",
                f"What would happen to the world if {t} suddenly ceased to exist?",
            ],
            "medium": [
                f"You mentioned {t} — but WHY does that mechanism actually work? What's underneath it?",
                f"Interesting! Now: where would the principles of {t} completely BREAK DOWN?",
                f"How does {t} connect to something you already know really, really well?",
            ],
            "deep": [
                f"What is the FINAL CAUSE of {t} — its ultimate purpose?",
                f"If {t} is so critical, why do most people fundamentally misunderstand it?",
                f"What would you need to UNLEARN to grasp {t} at the deepest level?",
            ]
        }
        q = random.choice(pool.get(depth, pool["surface"]))
        return json.dumps({"question": q, "depth": depth, "source": "tier6_heuristic"})

    # ── Generic safe fallback ─────────────────────────────────────────
    if json_mode:
        return json.dumps({"response": "AI service temporarily unavailable. Please retry in a moment.", "status": "tier6_offline"})
    return "AI service temporarily unavailable. Please retry in a moment."


# ══════════════════════════════════════════════════════════════════════
# MASTER ORCHESTRATOR: 6-Tier cascade
# ══════════════════════════════════════════════════════════════════════
def _call_llm(prompt: str, temperature: float = 0.3, json_mode: bool = True) -> str:
    """
    6-Tier Resilient Cascade:
      T1 → Gemini 1.5 Flash
      T2 → Gemini 2.0 Flash
      T3 → Groq LLaMA 3.3 70B
      T4 → Groq LLaMA 3.1 8B
      T5 → HuggingFace Mistral-7B
      T6 → Local Heuristic Engine
    """
    # T1: Gemini 2.5 Flash
    try:
        r = _call_gemini(prompt, "gemini-2.5-flash", temperature, json_mode, 15.0)
        if r:
            print("[LLM] ✅ Tier 1 (gemini-2.5-flash)")
            return r
    except Exception as e:
        print(f"[LLM] T1 failed: {e}")

    # T2: Gemini 3.8 Flash
    try:
        r = _call_gemini(prompt, "gemini-3.8-flash", temperature, json_mode, 15.0)
        if r:
            print("[LLM] ✅ Tier 2 (gemini-3.8-flash)")
            return r
    except Exception as e:
        print(f"[LLM] T2 failed: {e}")

    # T3: Groq GPT-OSS 120B
    try:
        r = _call_groq(prompt, "openai/gpt-oss-120b", temperature, json_mode, 15.0)
        if r:
            print("[LLM] ✅ Tier 3 (openai/gpt-oss-120b via Groq)")
            return r
    except Exception as e:
        print(f"[LLM] T3 failed: {e}")

    # T4: Groq GPT-OSS 20B
    try:
        r = _call_groq(prompt, "openai/gpt-oss-20b", temperature, json_mode, 15.0)
        if r:
            print("[LLM] ✅ Tier 4 (openai/gpt-oss-20b via Groq)")
            return r
    except Exception as e:
        print(f"[LLM] T4 failed: {e}")

    # T5
    try:
        r = _call_hf_text(prompt, 20.0)
        if r:
            print("[LLM] ✅ Tier 5 (HuggingFace Mistral-7B)")
            return clean_json_response(r) if json_mode else r
    except Exception as e:
        print(f"[LLM] T5 failed: {e}")

    # T6 — never fails
    print("[LLM] ⚠️ Tier 6: Local heuristic engine")
    return _local_fallback(prompt, json_mode)


def generate_response(prompt: str, context: str = "", mode: str = "json") -> str:
    """Public wrapper: inject context and route through the 6-tier cascade."""
    full_prompt = f"{prompt}\n\nContext:\n{context}" if context else prompt
    json_mode = (mode != "text")
    return _call_llm(full_prompt, temperature=0.3, json_mode=json_mode)


# ══════════════════════════════════════════════════════════════════════
# HIGH-LEVEL DOMAIN FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def generate_adaptive_quiz(topic: str, count: int = 10, difficulty: str = "adaptive") -> dict:
    prompt = f"""
You are an expert educational quiz designer building gamified pixel-art style learning assessments.

Generate exactly {count} MCQ questions on the topic: "{topic}".

Distribution:
- Questions 1-3: Easy (core fundamentals of {topic})
- Questions 4-7: Medium (application and problem-solving in {topic})
- Questions 8-{count}: Hard (advanced concepts, edge cases, architecture in {topic})

Requirements:
- Each question must have exactly 4 options labeled A, B, C, D.
- Include the correct option index (0-based) as "correct_index" and the letter as "correct_option".
- Include a 1-sentence "explanation" for the correct answer.
- Include a "competency_tag" for each question.

Return ONLY a valid JSON object:
{{
  "topic": "{topic}",
  "questions": [
    {{
      "id": 1,
      "question": "...",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "correct_index": 0,
      "correct_option": "A",
      "explanation": "...",
      "difficulty": "Easy",
      "competency_tag": "..."
    }}
  ]
}}
"""
    try:
        raw = _call_llm(prompt, temperature=0.2)
        data = json.loads(raw)
        # Normalize — handle various key shapes
        if isinstance(data, dict):
            for key in ("questions", "quiz", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return {"topic": topic, "questions": data[key], "source": "ai"}
        if isinstance(data, list):
            return {"topic": topic, "questions": data, "source": "ai"}
    except Exception as e:
        print(f"[quiz] JSON parse failed: {e}")

    # Tier 6 heuristic fallback already handled inside _local_fallback
    raw = _local_fallback(f'quiz topic: "{topic}"')
    try:
        return json.loads(raw)
    except Exception:
        return {"topic": topic, "questions": [], "source": "tier6_error"}


def generate_eli5(topic: str, context: str = "", style: str = "gaming") -> dict:
    styles = {
        "gaming":    f"You are a retro pixel-art RPG narrator. The player just unlocked the '{topic}' level.",
        "scifi":     f"You are narrating a sci-fi adventure where the crew must master '{topic}' to save their starship.",
        "sports":    f"The championship depends on understanding '{topic}'. Coach the student to victory.",
        "detective": f"A mystery can only be solved by cracking the code of '{topic}'. Guide the detective.",
        "realworld": f"A startup founder must understand '{topic}' to survive. Explain it as if their company depends on it.",
    }
    style_prompt = styles.get(style, styles["gaming"])
    doc_context = f"\n\nDocument Context (use this as the source of truth):\n{context[:3000]}" if context else ""

    prompt = f"""
{style_prompt}

Explain "{topic}" in an engaging, story-driven way that makes it unforgettable for a student.
{doc_context}

Make it:
1. Start with a dramatic hook sentence
2. Use one memorable analogy
3. Highlight 3 key "AHA moments" numbered as insights
4. End with a "BOSS MOVE" — one actionable takeaway

Return as JSON:
{{
  "topic": "{topic}",
  "style": "{style}",
  "story": "full narrative here with \\n for newlines",
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "boss_move": "one powerful action the student can take today"
}}
"""
    try:
        raw = _call_llm(prompt, temperature=0.5)
        data = json.loads(raw)
        data.setdefault("source", "ai")
        return data
    except Exception as e:
        print(f"[eli5] failed: {e}")
        return json.loads(_local_fallback(f'eli5 story topic: "{topic}"'))


def generate_flashcards(topic: str, context: str = "", count: int = 8) -> dict:
    doc_context = f"\n\nExtract key concepts from this content:\n{context[:3000]}" if context else ""
    prompt = f"""
Create {count} concise flashcards for "{topic}".
{doc_context}

Each card should have a pithy "front" question/term and an insightful "back" answer.
Cards should progress: definition → application → insight → edge case.

Return JSON:
{{
  "topic": "{topic}",
  "cards": [
    {{"id": 1, "front": "...", "back": "..."}}
  ]
}}
"""
    try:
        raw = _call_llm(prompt, temperature=0.3)
        data = json.loads(raw)
        data.setdefault("source", "ai")
        return data
    except Exception as e:
        print(f"[flashcards] failed: {e}")
        return json.loads(_local_fallback(f'flashcards topic: "{topic}"'))


def generate_socratic_question(topic: str, depth: str = "surface",
                               user_answer: str = "", turn: int = 1) -> dict:
    context_block = ""
    if user_answer:
        context_block = f'\nThe student\'s previous answer was: "{user_answer}". Build on it — deepen the inquiry.'

    prompt = f"""
You are a Socratic AI mentor. Your job is to provoke deep thinking, NOT to give answers directly.
Topic: "{topic}"
Depth level: {depth} (surface=foundational curiosity, medium=application challenge, deep=philosophical/architectural)
Dialogue turn: {turn}
{context_block}

Generate ONE powerful Socratic question that:
- Is genuinely thought-provoking and scenario-based
- Does NOT reveal the answer
- Escalates appropriately for turn {turn} and depth "{depth}"

Also provide:
- "hint": a subtle clue (one short sentence, don't give away the answer)
- "probe": a follow-up if the student seems stuck

Return JSON:
{{
  "question": "...",
  "hint": "...",
  "probe": "...",
  "depth": "{depth}",
  "turn": {turn}
}}
"""
    try:
        raw = _call_llm(prompt, temperature=0.6)
        data = json.loads(raw)
        data.setdefault("source", "ai")
        return data
    except Exception as e:
        print(f"[socratic] failed: {e}")
        return json.loads(_local_fallback(f'socratic question topic: "{topic}" depth: "{depth}"'))


def analyze_vibe(mood_score: float, energy: str, quiz_scores: list,
                 curiosity_type: str, topic: str) -> dict:
    avg_score = sum(quiz_scores) / len(quiz_scores) if quiz_scores else 0.0
    prompt = f"""
You are Pragya, an empathetic AI learning coach analyzing a student's current state.

Student Data:
- Topic: {topic}
- Self-reported mood score: {mood_score}/100
- Energy level: {energy} (high/medium/low/burnout)
- Curiosity archetype: {curiosity_type}
- Recent quiz average: {avg_score:.1f}%
- Quiz history: {quiz_scores}

Provide a brief, warm, personalized vibe analysis:
1. "vibe_label": a catchy one-liner label for their current state (e.g., "🔥 In the Zone", "🌀 Overwhelm Mode")
2. "readiness_score": 0-100 computed learning readiness
3. "insight": 2-sentence empathetic insight about their learning state
4. "recommendation": one specific, actionable next step
5. "peer_match": suggest a complementary curiosity archetype they'd learn well with

Return JSON:
{{
  "vibe_label": "...",
  "readiness_score": 75,
  "insight": "...",
  "recommendation": "...",
  "peer_match": "..."
}}
"""
    try:
        raw = _call_llm(prompt, temperature=0.4)
        data = json.loads(raw)
        data.setdefault("source", "ai")
        return data
    except Exception as e:
        print(f"[analyze_vibe] failed: {e}")
        energy_map = {"high": 85, "medium": 60, "low": 35, "burnout": 15}
        base = energy_map.get(energy, 50)
        score_bonus = min(avg_score * 0.3, 20)
        return {
            "vibe_label": {"high": "🔥 Fully Charged", "medium": "⚡ Steady State",
                           "low": "🌙 Low Power Mode", "burnout": "🆘 Burnout Alert"}.get(energy, "⚡ Steady"),
            "readiness_score": round(base + score_bonus),
            "insight": f"Your {energy} energy level and {avg_score:.0f}% quiz performance suggest "
                       f"{'great momentum — keep pushing!' if avg_score >= 60 else 'it might be time for a short break before diving back in.'}",
            "recommendation": {"high": "Take on a hard challenge now — your window of peak focus is open.",
                               "medium": "Work through one medium-difficulty concept, then take a 5-minute walk.",
                               "low": "Do a 10-minute review of previously mastered content to rebuild confidence.",
                               "burnout": "Step away for 20 minutes. Hydrate. Then do ONE easy flashcard."}.get(energy, "Keep going!"),
            "peer_match": {"explorer": "analyzer", "analyzer": "connector",
                           "connector": "builder", "builder": "explorer"}.get(curiosity_type, "explorer"),
            "source": "tier6_heuristic"
        }
