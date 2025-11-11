#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini  # ✅ Import Gemini
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import dotenv
import time
import re
from pathlib import Path
import tempfile
import os
import base64


# Set Streamlit page configuration
st.set_page_config(
    page_title="Multimodal AI Agent- Video Summarizer", page_icon="🎥", layout="wide"
)

# Load .env file
dotenv.load_dotenv()

# API keys configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# PHI_API_KEY = os.getenv("PHI_API_KEY")

# Allow selecting a model from the available Gemini models (helps avoid quota-limited models)
MODEL_CHOICES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-live",
    "gemini-2.0-flash-live",
]
# Default can be overridden with env var MODEL_ID
default_model = os.getenv("MODEL_ID", "gemini-2.5-flash")
if default_model not in MODEL_CHOICES:
    MODEL_CHOICES.insert(0, default_model)

# Streamlit selector so you can switch models without editing code
st.sidebar.markdown("## Model selection")
selected_model = st.sidebar.selectbox(
    "Choose Gemini model (pick one with available quota)",
    MODEL_CHOICES,
    index=MODEL_CHOICES.index(default_model),
)

if not GOOGLE_API_KEY:
    st.error("Google API key is missing. Please check your configuration.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        st.success("✅ Google Generative AI API Key Configured Successfully!")
    except Exception as e:
        st.error(f"❌ Failed to configure API: {e}")

st.title("Video AI Summarizer Agent 🎥🎤🖬")
st.header("Powered by Gemini 2.0 Flash Exp")


# Initialize the AI agent
@st.cache_resource
def initialize_agent(model_id: str = "gemini-2.5-flash"):
    """Initialize and cache the Agent for a given Gemini model id."""
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id=model_id),
        tools=[],
        markdown=True,
    )


# Initialize agent with the selected model
multimodal_Agent = initialize_agent(selected_model)


def run_with_retries(agent, prompt, videos=None, max_attempts: int = 3):
    """Run agent.run with simple retry logic on 429 / quota errors.

    - Parses retry seconds from common error messages when possible.
    - Exponential backoff if parsing fails.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return agent.run(prompt, videos=videos)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "Quota exceeded" in msg or "rate limit" in msg.lower():
                # Try to extract retry seconds from message
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg)
                if not m:
                    # Some messages include retry_delay { seconds: N }
                    m = re.search(r"retry_delay.*seconds[:\s]+(\d+(?:\.\d+)?)", msg)
                wait = float(m.group(1)) if m else min(60, 2**attempt)
                st.warning(
                    f"Rate limit detected (attempt {attempt}/{max_attempts}). Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
                continue
            # Not a rate limit — re-raise to be handled by outer try/except
            raise
    raise Exception(
        "Exceeded retries due to rate limits. Consider switching model or upgrading quota."
    )


# File upload
video_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi"],
    help="Upload a video for AI analysis",
)

if video_file:
    # Read file data once
    video_data = video_file.read()

    # Save the video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_data)
        video_path = temp_video.name

    # Encode the video in base64
    video_base64 = base64.b64encode(video_data).decode("utf-8")

    # Display the video
    st.markdown(
        f"""
    <div class="video-container">
        <video controls>
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    """,
        unsafe_allow_html=True,
    )

    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content...",
        help="Provide specific questions or insights you want from the video.",
    )

    progress_bar = st.progress(0)

    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    st.text("Uploading and processing the video...")
                    processed_video = upload_file(video_path)
                    progress_bar.progress(25)

                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)
                        progress_bar.progress(50)

                    st.text("Generating analysis prompt...")
                    analysis_prompt = f"""
                    Analyze the uploaded video for content and context.
                    Respond to the following query using video insights:
                    {user_query}
                    Provide a detailed, user-friendly, and actionable response.
                    """

                    st.text("Running AI analysis on the video...")
                    response = multimodal_Agent.run(
                        analysis_prompt, videos=[processed_video]
                    )
                    progress_bar.progress(75)

                    st.subheader("Analysis Result")
                    st.markdown(response.content)

                    progress_bar.progress(100)

            except Exception as error:
                st.error(f"❌ An error occurred during analysis: {error}")
            finally:
                Path(video_path).unlink(missing_ok=True)

else:
    st.info("Upload a video file to begin analysis.")

# Customize UI
st.markdown(
    """
    <style>
        .stTextArea textarea {
            height: 100px;
        }
    .video-container video {
        width: 300px !important;   /* Keep video small */
        max-width: 100%;           /* Responsive */
        height: auto !important;   /* Maintain aspect ratio */
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        margin: 10px auto;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
