import os
import sys
from streamlit_mic_recorder import mic_recorder
from audio_utils import speech_to_text_from_bytes, text_to_speech_bytes

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
    # add more if you use them, e.g.:
    # "LANGCHAIN_API_KEY",
    # "LANGCHAIN_PROJECT",
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

# Show previous chat
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Input: voice OR text ----
st.write("### बोलकर या लिखकर अपना प्रश्न पूछें")

# 1) Voice input via mic
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
    user_input = st.chat_input("अपना प्रश्न पूछें")

# ---- Existing logic: send user_input into LangGraph ----
if user_input:
    # Show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

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

    # Show assistant message
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

        # 🔊 Text-to-speech button
        if st.button(
            "🔊 जवाब सुनें",
            key=f"tts_{len(st.session_state['messages'])}",
        ):
            audio_bytes = text_to_speech_bytes(assistant_reply)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            else:
                st.error("आवाज़ बनाने में दिक्कत आई, कृपया दोबारा कोशिश करें।")

        with st.expander("🔍 See AgroSense reasoning", expanded=False):
            if extracted_params:
                st.markdown("**Known Environment Parameters:**")
                st.json(extracted_params)
            if crop_results:
                st.markdown("**Top Crop Candidates:**")
                for i, item in enumerate(crop_results, start=1):
                    st.write(
                        f"{i}. **{item['crop']}** (score = `{item['score']:.2f}`)"
                    )

    # Store assistant message in history (if you weren't already doing this elsewhere)
    st.session_state["messages"].append(
        {"role": "assistant", "content": assistant_reply}
    )


