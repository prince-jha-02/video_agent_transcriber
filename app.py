import os
import tempfile
from pathlib import Path

import streamlit as st
import yt_dlp

from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.sammarize import analyze_video_content


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

MAX_VIDEO_SECONDS = 10 * 60


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Video Assistance",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SIMPLE MODERN STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0b0d0f;
    }

    /* Main width */
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Headings */
    h1 {
        letter-spacing: -0.04em;
    }

    h2, h3 {
        letter-spacing: -0.025em;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }

    /* Cards */
    .card {
        background-color: #111417;
        border: 1px solid #292d32;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .card-label {
        color: #777d85;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    .card-title {
        color: #f0f1f2;
        font-size: 18px;
        font-weight: 600;
    }

    .small-text {
        color: #858b93;
        font-size: 13px;
        line-height: 1.6;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #555b63;
        font-size: 12px;
        padding-top: 30px;
        margin-top: 50px;
        border-top: 1px solid #202428;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns(
    [5, 1],
    vertical_alignment="center",
)

with header_col1:

    st.markdown(
        "## 🎬 Video Assistance"
    )

    st.caption(
        "Video intelligence workspace"
    )

with header_col2:

    st.markdown(
        """
        <div style="
            text-align:center;
            border:1px solid #30343a;
            background:#15181b;
            padding:8px 5px;
            border-radius:20px;
            color:#aeb3ba;
            font-size:12px;
        ">
        MAX 10 MIN
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# ============================================================
# INTRO
# ============================================================

st.title(
    "Understand your video."
)

st.write(
    "Upload a short video or paste a YouTube link. "
    "The application will extract the important information "
    "and let you ask questions about the content."
)


# ============================================================
# LIMIT NOTICE
# ============================================================

st.info(
    "⏱️ **10-minute limit** — Videos longer than 10 minutes "
    "are not supported. This demo uses free-tier transcription "
    "and AI services."
)


# ============================================================
# INPUT CARD
# ============================================================

st.subheader(
    "Video source"
)


source_type = st.radio(
    "Choose how you want to provide the video",
    [
        "YouTube URL",
        "Upload video",
    ],
    horizontal=True,
)


source = None
uploaded_file = None


# ============================================================
# YOUTUBE
# ============================================================

if source_type == "YouTube URL":

    source = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )


# ============================================================
# UPLOAD
# ============================================================

else:

    uploaded_file = st.file_uploader(
        "Upload your video",
        type=[
            "mp4",
            "mov",
            "mkv",
            "webm",
            "avi",
        ],
    )


# ============================================================
# LANGUAGE
# ============================================================

language = st.selectbox(
    "Language",
    [
        "english",
        "hinglish",
    ],
)


st.write("")


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "▶  Analyze video",
    type="primary",
    use_container_width=True,
)


# ============================================================
# FILE HELPER
# ============================================================

def save_uploaded_file(file):

    extension = Path(
        file.name
    ).suffix

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=extension,
    )

    temp_file.write(
        file.getbuffer()
    )

    temp_file.close()

    return temp_file.name


# ============================================================
# YOUTUBE DURATION
# ============================================================

def get_youtube_duration(url):

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }

    try:

        with yt_dlp.YoutubeDL(options) as ydl:

            info = ydl.extract_info(
                url,
                download=False,
            )

        return info.get("duration")

    except Exception as e:

        st.error(
            f"Unable to read YouTube video information: {e}"
        )

        return None


# ============================================================
# FORMAT DURATION
# ============================================================

def format_duration(seconds):

    if seconds is None:
        return "Unknown"

    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes}:{seconds:02d}"


# ============================================================
# PROCESS VIDEO
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if source_type == "YouTube URL":

        if not source:

            st.warning(
                "Please enter a YouTube URL."
            )

            st.stop()

    else:

        if uploaded_file is None:

            st.warning(
                "Please upload a video."
            )

            st.stop()

        source = save_uploaded_file(
            uploaded_file
        )


    # --------------------------------------------------------
    # CHECK YOUTUBE DURATION
    # --------------------------------------------------------

    if source_type == "YouTube URL":

        with st.spinner(
            "Checking video duration..."
        ):

            duration = get_youtube_duration(
                source
            )


        if duration is None:
            st.stop()


        duration_text = format_duration(
            duration
        )


        if duration > MAX_VIDEO_SECONDS:

            st.error(
                f"""
                This video is **{duration_text}** long.

                The maximum supported duration is
                **10:00**.

                Please select a video that is 10 minutes
                or shorter.
                """
            )

            st.stop()


        st.success(
            f"Video duration: {duration_text} ✓"
        )


    # --------------------------------------------------------
    # PROCESSING
    # --------------------------------------------------------

    try:

        with st.status(
            "Analyzing video...",
            expanded=True,
        ):

            # ----------------------------------------------
            # AUDIO
            # ----------------------------------------------

            st.write(
                "Extracting audio..."
            )

            chunks = process_input(
                source
            )


            st.write(
                f"Audio prepared — {len(chunks)} chunk(s)"
            )


            # ----------------------------------------------
            # TRANSCRIPTION
            # ----------------------------------------------

            st.write(
                "Transcribing audio..."
            )

            transcript = transcribe_all(
                chunks,
                language=language,
            )


            if not transcript or not transcript.strip():

                st.error(
                    "No speech was detected in the video."
                )

                st.stop()


            # ----------------------------------------------
            # UNIFIED INTELLIGENCE ANALYSIS (1 API CALL)
            # ----------------------------------------------

            st.write(
                "Analyzing video intelligence with Gemini (single optimized call)..."
            )

            analysis = analyze_video_content(
                transcript
            )

            title = analysis["title"]
            summary = analysis["summary"]
            action_items = analysis["action_items"]
            decisions = analysis["decisions"]
            questions = analysis["questions"]


            # ----------------------------------------------
            # RAG / Q&A PREPARATION
            # ----------------------------------------------

            st.write(
                "Preparing video Q&A..."
            )

            from core.rag_engine import build_rag_chain

            rag_chain = build_rag_chain(
                transcript
            )


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        st.session_state.result = {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "decisions": decisions,
            "questions": questions,
            "rag_chain": rag_chain,
        }
        st.session_state.chat_history = []



        st.success(
            "Video analysis completed successfully."
        )


    except Exception as e:

        st.error(
            "Something went wrong while processing the video."
        )

        st.exception(e)

        st.stop()


# ============================================================
# RESULTS
# ============================================================

result = st.session_state.result


if result is not None:

    st.divider()


    # ========================================================
    # TITLE
    # ========================================================

    st.caption(
        "VIDEO ANALYSIS"
    )

    st.header(
        result["title"]
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    st.subheader(
        "Summary"
    )

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True,
    )

    st.write(
        result["summary"]
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


    # ========================================================
    # INSIGHTS
    # ========================================================

    st.subheader(
        "Insights"
    )


    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )


    with col1:

        st.markdown(
            """
            <div class="card">

                <div class="card-label">
                    Next steps
                </div>

                <div class="card-title">
                    Action Items
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            result["action_items"]
        )


    with col2:

        st.markdown(
            """
            <div class="card">

                <div class="card-label">
                    Outcomes
                </div>

                <div class="card-title">
                    Key Decisions
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            result["decisions"]
        )


    with col3:

        st.markdown(
            """
            <div class="card">

                <div class="card-label">
                    Follow-up
                </div>

                <div class="card-title">
                    Open Questions
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(
            result["questions"]
        )


    # ========================================================
    # TRANSCRIPT
    # ========================================================

    st.divider()

    st.subheader(
        "Transcript"
    )

    with st.expander(
        "View full transcript"
    ):

        st.write(
            result["transcript"]
        )


    # ========================================================
    # VIDEO CHAT
    # ========================================================

    st.divider()

    st.subheader(
        "Ask your video"
    )

    st.caption(
        "Ask questions about anything discussed in the video."
    )


    # Display existing conversation history
    for chat_msg in st.session_state.chat_history:
        with st.chat_message(chat_msg["role"]):
            st.markdown(chat_msg["content"])

    question = st.chat_input(
        "Ask something about this video..."
    )

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the video..."):
                from core.rag_engine import ask_question
                answer = ask_question(
                    result["rag_chain"],
                    question,
                )
            st.markdown(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })



# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Video Assistance · Maximum video duration: 10 minutes
    </div>
    """,
    unsafe_allow_html=True,
)
