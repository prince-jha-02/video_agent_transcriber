import os
import time
import hashlib
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Fallback candidate models if quota or temporary demand issues occur
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-3.6-flash",
]
# Deduplicate while preserving priority order
CANDIDATE_MODELS = list(dict.fromkeys(FALLBACK_MODELS))

MAX_RETRIES_PER_MODEL = 2

# In-memory transcript cache to avoid duplicate API calls
_ANALYSIS_CACHE: Dict[str, Dict[str, str]] = {}


# ============================================================
# LLM INSTANTIATION
# ============================================================

def get_llm(model_name: str = None) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in environment or .env")

    model = model_name or DEFAULT_MODEL
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        timeout=30,
    )


def _hash_transcript(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _format_field_value(val: Any, default_empty: str = "") -> str:
    """Format extracted field as clean markdown string."""
    if not val:
        return default_empty
    if isinstance(val, list):
        items = []
        for i, item in enumerate(val):
            item_str = str(item).strip()
            if item_str.startswith(("-", "*", "•")):
                items.append(item_str)
            else:
                items.append(f"{i + 1}. {item_str}")
        return "\n".join(items)
    return str(val).strip()


# ============================================================
# UNIFIED VIDEO ANALYSIS (1 API CALL FOR EVERYTHING)
# ============================================================

def analyze_video_content(transcript: str) -> Dict[str, str]:
    """
    Extracts title, summary, action items, key decisions, and open questions
    in a SINGLE unified Gemini API call with structured JSON parsing.
    Reduces Gemini API calls by 80% and saves input/output tokens.
    """
    if not transcript or not transcript.strip():
        return {
            "title": "Video Analysis",
            "summary": "No transcript available.",
            "action_items": "No action items found.",
            "decisions": "No key decisions found.",
            "questions": "No open questions found.",
        }

    cache_key = _hash_transcript(transcript)
    if cache_key in _ANALYSIS_CACHE:
        print("[CACHE] Returning video analysis from cache (0 Gemini API calls)")
        return _ANALYSIS_CACHE[cache_key]

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in environment or .env")

    system_prompt = """You are an expert video and meeting intelligence analyst.
Analyze the provided transcript and extract all the following sections in valid JSON format:
{{
  "title": "A short, professional title (maximum 8 words, no quotation marks)",
  "summary": "A clear, professional summary using bullet points covering main topics, important information, major conclusions, and context.",
  "action_items": "Numbered list of action items with Task description, Owner, and Deadline (if mentioned, else 'Not specified'). If none found, write 'No action items found.'",
  "decisions": "Numbered list of key decisions made. If none found, write 'No key decisions found.'",
  "questions": "Numbered list of unresolved questions or topics needing follow-up. If none found, write 'No open questions found.'"
}}
Return strictly a valid JSON object without markdown fences or extra explanations.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}")
    ])

    last_error = None
    parsed_result = None

    for model_name in CANDIDATE_MODELS:
        for attempt in range(MAX_RETRIES_PER_MODEL):
            try:
                print(f"[GEMINI] Calling model ({model_name}) for video analysis...")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    timeout=30,
                )
                chain = prompt | llm | JsonOutputParser()
                parsed_result = chain.invoke({"text": transcript})
                if parsed_result and isinstance(parsed_result, dict):
                    break
            except Exception as e:
                err_msg = str(e)
                last_error = e
                print(f"[WARNING] Model {model_name} (attempt {attempt + 1}): {err_msg[:120]}")

                # If resource exhausted (quota limit) or 404, immediately try next fallback model
                if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "404" in err_msg:
                    break

                if attempt < MAX_RETRIES_PER_MODEL - 1:
                    time.sleep(1)

        if parsed_result:
            break


    if not parsed_result:
        raise RuntimeError(f"All Gemini models failed to analyze transcript: {last_error}")

    result = {
        "title": _format_field_value(parsed_result.get("title"), "Video Analysis"),
        "summary": _format_field_value(parsed_result.get("summary"), "No summary generated."),
        "action_items": _format_field_value(parsed_result.get("action_items"), "No action items found."),
        "decisions": _format_field_value(parsed_result.get("decisions"), "No key decisions found."),
        "questions": _format_field_value(parsed_result.get("questions"), "No open questions found."),
    }

    _ANALYSIS_CACHE[cache_key] = result
    return result


# ============================================================
# BACKWARD-COMPATIBLE STANDALONE FUNCTIONS
# ============================================================

def summarize(transcript: str) -> str:
    """Generates summary using the unified cache/analyzer."""
    res = analyze_video_content(transcript)
    return res["summary"]


def genrate_title(transcript: str) -> str:
    """Generates title using the unified cache/analyzer."""
    res = analyze_video_content(transcript)
    return res["title"]