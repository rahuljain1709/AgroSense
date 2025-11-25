import os
import sys

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
