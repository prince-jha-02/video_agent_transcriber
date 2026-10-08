import os
from dotenv import load_dotenv
from pydub import AudioSegment
import requests

from assemblyai import TranscriptStatus
from assemblyai.prerecorded.v2 import (
    Transcriber,
    TranscriptionConfig
)

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

SARVAM_PIECE_SECONDS = 25


# ------------------------------------------------------------
# ASSEMBLYAI
# ------------------------------------------------------------

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")


# ------------------------------------------------------------
# SARVAM
# ------------------------------------------------------------

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_STT_TRANSLATE_URL = (
    "https://api.sarvam.ai/speech-to-text-translate"
)

SARVAM_MODEL = os.getenv(
    "SARVAM_STT_MODEL",
    "saaras:v2.5"
)


# ============================================================
# ASSEMBLYAI STT
# ============================================================

def transcribe_chunk_assemblyai(chunk_path: str) -> str:
    """
    Transcribe one audio chunk using AssemblyAI.

    Works for English and can also handle
    Hindi-English code-switching.
    """

    if not ASSEMBLYAI_API_KEY:
        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is not set in environment / .env"
        )

    print(
        f"  → Sending "
        f"{os.path.basename(chunk_path)} "
        f"to AssemblyAI STT..."
    )

    config = TranscriptionConfig(
        speech_models=["universal-3-5-pro"]
    )

    transcriber = Transcriber(
        api_key=ASSEMBLYAI_API_KEY,
        config=config
    )

    transcript = transcriber.transcribe(
        chunk_path
    )

    if transcript.status == TranscriptStatus.error:
        raise RuntimeError(
            f"AssemblyAI transcription failed: "
            f"{transcript.error}"
        )

    return (
        transcript.text.strip()
        if transcript.text
        else ""
    )


# ============================================================
# SARVAM
# ============================================================

def _send_to_sarvam(piece_path: str) -> str:
    """
    Send one <=25 second WAV file to Sarvam
    and return the transcript.
    """

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in environment / .env"
        )

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(piece_path, "rb") as f:

        files = {
            "file": (
                os.path.basename(piece_path),
                f,
                "audio/wav"
            )
        }

        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false"
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:

        print(
            f"\n❌ Sarvam returned "
            f"{response.status_code}"
        )

        print(
            f"Response body:\n"
            f"{response.text}\n"
        )

        response.raise_for_status()

    result = response.json()

    return result.get(
        "transcript",
        ""
    ).strip()


# ============================================================
# SARVAM CHUNKING
# ============================================================

def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API has a 30-second limit.

    Therefore we split each chunk into
    25-second pieces.
    """

    if not SARVAM_API_KEY:
        raise RuntimeError(
            "SARVAM_API_KEY is not set in environment / .env"
        )

    audio = AudioSegment.from_wav(
        chunk_path
    )

    piece_ms = (
        SARVAM_PIECE_SECONDS * 1000
    )

    full_text_parts = []

    total_pieces = (
        (len(audio) + piece_ms - 1)
        // piece_ms
    )

    for i, start in enumerate(
        range(0, len(audio), piece_ms)
    ):

        piece = audio[
            start:start + piece_ms
        ]

        piece_path = (
            f"{chunk_path}_sv_{i}.wav"
        )

        piece.export(
            piece_path,
            format="wav"
        )

        try:

            print(
                f"  → Sarvam piece "
                f"{i + 1}/{total_pieces}..."
            )

            text = _send_to_sarvam(
                piece_path
            )

            if text:
                full_text_parts.append(
                    text
                )

        finally:

            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(
        full_text_parts
    ).strip()


# ============================================================
# ROUTER
# ============================================================

def transcribe_chunk(
    chunk_path: str,
    language: str = "english"
) -> str:

    """
    Select transcription engine.

    English:
        → AssemblyAI

    Hinglish:
        → Sarvam
    """

    if language.lower() == "hinglish":

        return transcribe_chunk_sarvam(
            chunk_path
        )

    return transcribe_chunk_assemblyai(
        chunk_path
    )


# ============================================================
# TRANSCRIBE ALL
# ============================================================

def transcribe_all(
    chunks: list,
    language: str = "english"
) -> str:

    full_transcript_parts = []

    if language.lower() == "hinglish":
        engine = "Sarvam AI"
    else:
        engine = "AssemblyAI"

    print(
        f"Using {engine} for transcription."
    )

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(chunks)}..."
        )

        text = transcribe_chunk(
            chunk,
            language=language
        )

        if text:
            full_transcript_parts.append(
                text
            )

    print(
        "Transcription complete."
    )

    return " ".join(
        full_transcript_parts
    ).strip()