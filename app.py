import os
import sys

import streamlit as st
from dotenv import load_dotenv

# Make sure Python can see the `src` package
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from agent.graph import graph  # type: ignore

load_dotenv()

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


if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Persist agent state between turns
if "agent_state" not in st.session_state:
    st.session_state["agent_state"] = {}

st.title("🌱 AgroSense – The Curious Farming Assistant")

st.markdown(
    """
    Hi, I am your assistant for your farming related queries | नमस्कार, मैं आपकी कृषि संबंधी प्रश्नों के लिए आपका सहायक हूँ। 
    """
)

# Show chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("अपना प्रश्न पूछें")

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
        assistant_reply = f"Sorry, something went wrong while processing your request:\n\n`{e}`"
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

        with st.expander("🔍 See AgroSense reasoning", expanded=False):
            if extracted_params:
                st.markdown("**Known Environment Parameters:**")
                st.json(extracted_params)
            if crop_results:
                st.markdown("**Top Crop Candidates:**")
                for i, item in enumerate(crop_results, start=1):
                    st.write(f"{i}. **{item['crop']}** (score = `{item['score']:.2f}`)")

    st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})

