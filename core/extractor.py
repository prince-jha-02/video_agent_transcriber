# Actionable items, key decisions, and open questions extraction
# Leverages the unified analyze_video_content engine to eliminate redundant Gemini API calls.

from core.sammarize import analyze_video_content


def extract_action_items(transcript: str) -> str:
    """Extract action items from transcript using cached single-call intelligence."""
    res = analyze_video_content(transcript)
    return res.get("action_items", "No action items found.")


def extract_key_decisions(transcript: str) -> str:
    """Extract key decisions from transcript using cached single-call intelligence."""
    res = analyze_video_content(transcript)
    return res.get("decisions", "No key decisions found.")


def extract_questions(transcript: str) -> str:
    """Extract open questions from transcript using cached single-call intelligence."""
    res = analyze_video_content(transcript)
    return res.get("questions", "No open questions found.")