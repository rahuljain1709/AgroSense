import os
import sys

from streamlit_mic_recorder import mic_recorder
from src.audio_utils import speech_to_text_from_bytes, text_to_speech_bytes

import streamlit as st
from dotenv import load_dotenv

# ----------------- CONFIG / SECRETS SETUP -----------------

# Load .env for local development
load_dotenv()


def get_config(key: str, default: str | None = None) -> str | None:
    # On Streamlit Cloud, prefer st.secrets
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    # Locally, fall back to environment variables / .env
    return os.getenv(key, default)


# List only the keys your project actually needs
for k in [
    "OPENAI_API_KEY",
]:
    v = get_config(k)
    if v:
        # Make sure downstream code using os.getenv(...) can see them
        os.environ[k] = v

# ----------------- IMPORT YOUR GRAPH AFTER SECRETS -----------------

# Make sure Python can see the `src` package
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agent.graph import graph  # type: ignore

# ----------------- STREAMLIT UI -----------------

st.set_page_config(
    page_title="AgroSense AI",
    page_icon="🌱",
    layout="wide",
)

with st.sidebar:
    st.title("🌾 AgroSense AI")
    st.markdown(
        """
        Multi-turn AI crop advisory assistant.

        **How it works:**
        1. You ask a general question.
        2. AgroSense asks for missing soil / weather details.
        3. You reply with those details.
        4. It recommends crops + practical tips.
        """
    )
    st.markdown("---")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Persist agent state between turns
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {}

st.title("🌱 AgroSense – The Curious Farming Assistant")

st.markdown(
    """
    Hi, I am your assistant for your farming related queries | 
    नमस्कार, मैं आपकी कृषि संबंधी प्रश्नों के लिए आपका सहायक हूँ।
    """
)

# ----------------- INPUT: VOICE OR TEXT (AND AGENT CALL) -----------------

# Move mic button closer to input field
with st.container():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("**🎙 बोलें:**")
    with col2:
        audio_data = mic_recorder(
            start_prompt="🎙️ बोलना शुरू करें",
            stop_prompt="⏹️ रिकॉर्डिंग बंद करें",
            just_once=True,
            key="voice_recorder",
        )

user_input = None

if audio_data:
    st.success("रिकॉर्डिंग हो गई, अब मैं आपकी आवाज़ को टेक्स्ट में बदल रहा हूँ...")
    transcript = speech_to_text_from_bytes(audio_data["bytes"])
    if transcript:
        st.info(f"🗣️ आपने कहा: **{transcript}**")
        user_input = transcript
    else:
        st.error("माफ़ कीजिए, मैं आपकी आवाज़ समझ नहीं पाया। कृपया दोबारा कोशिश करें।")

# 2) Normal text input fallback
if user_input is None:
    text_input = st.chat_input("अपना प्रश्न पूछें")
    if text_input:
        user_input = text_input

# ---- If we got any user_input this run, call the agent and update history ----
if user_input:
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Prepare input state: merge previous state + new query
    prev_state = st.session_state["agent_state"] or {}
    input_state = {**prev_state, "query": user_input}

    try:
        result = graph.invoke(input_state)
    except Exception as e:
        assistant_reply = (
            "Sorry, something went wrong while processing your request:\n\n"
            f"`{e}`"
        )
        extracted_params = None
        crop_results = []
    else:
        assistant_reply = result.get("answer", "I could not generate an answer.")
        extracted_params = result.get("extracted_params", None)
        crop_results = result.get("crop_results", [])

        # Save new agent state for next turn
        st.session_state["agent_state"] = result

    # Save assistant reply in history
    st.session_state["messages"].append(
        {"role": "assistant", "content": assistant_reply}
    )

    # Store extra reasoning info for THIS last turn in session_state
    # (so we can show it under the last assistant message)
    st.session_state["last_extracted_params"] = extracted_params
    st.session_state["last_crop_results"] = crop_results

# ----------------- RENDER FULL CHAT HISTORY (WITH TTS + REASONING) -----------------

last_extracted = st.session_state.get("last_extracted_params")
last_crops = st.session_state.get("last_crop_results")

for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            # TTS button for each assistant message
            if st.button("🔊 जवाब सुनें", key=f"tts_{i}"):
                audio_bytes = text_to_speech_bytes(msg["content"])
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.error("आवाज़ बनाने में दिक्कत आई, कृपया दोबारा कोशिश करें।")

            # Show reasoning only for the *latest* assistant message
            if i == len(st.session_state["messages"]) - 1:
                with st.expander("🔍 See AgroSense reasoning", expanded=False):
                    if last_extracted:
                        st.markdown("**Known Environment Parameters:**")
                        st.json(last_extracted)
                    if last_crops:
                        st.markdown("**Top Crop Candidates:**")
                        for j, item in enumerate(last_crops, start=1):
                            st.write(
                                f"{j}. **{item['crop']}** (score = `{item['score']:.2f}`)"
                            )
