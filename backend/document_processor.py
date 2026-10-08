"""
Pragya-Edu: document_processor.py
Multi-format document intelligence pipeline.
Supports: PDF, TXT, DOCX
Output: topic extraction, ELI5 story, 60s micro-summaries, MCQ quiz
"""

import os
import re
import json
import hashlib
from typing import Optional, Tuple

# ── Optional heavy imports (graceful degradation) ─────────────────────
try:
    from pypdf import PdfReader
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

try:
    from docx import Document as DocxDocument
    _DOCX_OK = True
except ImportError:
    _DOCX_OK = False


# ══════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from PDF bytes using pypdf."""
    if not _PDF_OK:
        return "[PDF extraction unavailable — install pypdf]"
    try:
        import io
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        return f"[PDF parse error: {e}]"


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract raw text from DOCX bytes using python-docx."""
    if not _DOCX_OK:
        return "[DOCX extraction unavailable — install python-docx]"
    try:
        import io
        doc = DocxDocument(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        return f"[DOCX parse error: {e}]"


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode plaintext bytes (UTF-8 with fallback to latin-1)."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1", errors="replace")


def extract_text(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Route to the correct extractor based on file extension.
    Returns (raw_text, detected_format).
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes), "pdf"
    elif ext in (".docx", ".doc"):
        return extract_text_from_docx(file_bytes), "docx"
    else:  # .txt or unknown
        return extract_text_from_txt(file_bytes), "txt"


# ══════════════════════════════════════════════════════════════════════
# CHUNKING & TOPIC EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split long text into overlapping chunks for LLM processing."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def extract_topic(text: str, filename: str) -> str:
    """
    Heuristic topic extraction from document:
    1. First heading (# Heading or ALL CAPS line)
    2. First meaningful sentence
    3. Filename stem as fallback
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Check for markdown heading
    for line in lines[:10]:
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()

    # All-caps short line (title-like)
    for line in lines[:5]:
        if line.isupper() and 3 < len(line.split()) < 10:
            return line.title()

    # First non-trivial sentence
    sentences = re.split(r"[.!?]", " ".join(lines[:3]))
    for s in sentences:
        s = s.strip()
        if len(s.split()) >= 4:
            return s[:80]

    # Filename fallback
    stem = os.path.splitext(filename)[0]
    return stem.replace("_", " ").replace("-", " ").title()


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


# ══════════════════════════════════════════════════════════════════════
# INTELLIGENCE PIPELINE — thin wrappers that call ai_engine
# ══════════════════════════════════════════════════════════════════════

def process_document(file_bytes: bytes, filename: str,
                     mode: str = "all", style: str = "gaming") -> dict:
    """
    Full document intelligence pipeline.
    mode: 'eli5' | 'flashcards' | 'quiz' | 'all'
    style: 'gaming' | 'scifi' | 'sports' | 'detective' | 'realworld'

    Returns dict with keys: topic, text_preview, eli5, flashcards, quiz
    """
    # Import here to avoid circular imports
    from ai_engine import generate_eli5, generate_flashcards, generate_adaptive_quiz

    raw_text, fmt = extract_text(file_bytes, filename)
    topic = extract_topic(raw_text, filename)

    # Use first 4000 chars as LLM context window
    context = raw_text[:4000].strip()
    preview = raw_text[:500].strip()
    fhash   = file_hash(file_bytes)

    result: dict = {
        "topic":        topic,
        "format":       fmt,
        "file_hash":    fhash,
        "char_count":   len(raw_text),
        "text_preview": preview,
        "eli5":         None,
        "flashcards":   None,
        "quiz":         None,
    }

    if mode in ("eli5", "all"):
        try:
            result["eli5"] = generate_eli5(topic, context, style)
        except Exception as e:
            result["eli5"] = {"error": str(e), "story": f"Could not generate story for {topic}."}

    if mode in ("flashcards", "all"):
        try:
            result["flashcards"] = generate_flashcards(topic, context, count=8)
        except Exception as e:
            result["flashcards"] = {"error": str(e), "cards": []}

    if mode in ("quiz", "all"):
        try:
            result["quiz"] = generate_adaptive_quiz(topic, count=10)
        except Exception as e:
            result["quiz"] = {"error": str(e), "questions": []}

    return result


def extract_key_concepts(text: str, max_concepts: int = 10) -> list[str]:
    """
    Lightweight local extraction of key concepts from text.
    Returns a list of noun-phrase candidates.
    """
    # Remove markdown, numbers, and short words
    clean = re.sub(r"[#*_`\[\]()>]", "", text)
    clean = re.sub(r"\d+", "", clean)

    # Candidate: capitalized phrases (likely proper nouns / key terms)
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", clean)
    freq: dict[str, int] = {}
    for c in candidates:
        if len(c.split()) >= 2 or len(c) > 5:
            freq[c] = freq.get(c, 0) + 1

    # Sort by frequency, return top N
    sorted_concepts = sorted(freq.items(), key=lambda x: -x[1])
    return [c for c, _ in sorted_concepts[:max_concepts]]
