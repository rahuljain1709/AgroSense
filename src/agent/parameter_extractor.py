# src/agent/parameter_extractor.py

from langchain_openai import ChatOpenAI
from agent.schema import AgentState
from dotenv import load_dotenv
import json

# These are the minimal required keys to start crop recommendation
REQUIRED_KEYS = ["n", "p", "k", "ph", "temperature", "rainfall"]  # humidity optional

# Phrases meaning "I don't know / can't give details"
# We support both Latin (Hinglish/English) and Devanagari (Hindi) forms.
REFUSAL_PATTERNS = [
    # Latin / Hinglish
    "nahi", "nahin",
    "nahi de sakta", "nahi de sakti",
    "nahi bata sakta", "nahi bata sakti",
    "nahi bata paunga", "nahi bata paungi",
    "mujhe nahi pata", "mujhe nahin pata",
    "mujhe pata nahi", "mujhe pata nahin",
    "pata nahi", "pata nahin",
    "don't know", "dont know",
    "can't say", "cant say",

    # Devanagari Hindi
    "मुझे नहीं पता",
    "मुझे नही पता",
    "मुझे पता नहीं",
    "मुझे पता नही",
    "पता नहीं",
    "पता नही",
    "नहीं पता",
    "नही पता",
]


def parse_environment_parameters(state: AgentState) -> dict:
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Previous parameters we already know from earlier turns
    prev = state.extracted_params or {
        "n": None,
        "p": None,
        "k": None,
        "temperature": None,
        "humidity": None,
        "ph": None,
        "rainfall": None,
    }

    # NOTE: This model's only job is to output JSON. We don't care about
    # Hindi/English style here, only correct numeric extraction.
    prompt = f"""
You are an expert agronomy assistant. Your ONLY job is to read the user's text
and extract farming environmental parameters into a JSON object.

USER MESSAGE (may be Hindi, Hinglish, or English, in Devanagari or Latin script):
{state.query}

INTERPRETATION RULES:
- The farmer may mix Hindi and English (e.g., "mere khet me nitrogen kam hai", "rainfall high hai").
- Understand words like: kam = low, zyada = high, medium = medium, normal = medium.
- Understand phrases like "pani zyada padta hai" → high rainfall, "garam ilaaka" → warm temperature.
- You may see some text in Urdu-like script as well; ignore the script and focus on meaning.

OUTPUT RULES:
- Respond ONLY with a JSON object, no extra text, comments, or explanations.
- If numeric values are explicitly mentioned, use them.
  - "temperature 30", "30 degree", "30°C" → temperature = 30
  - "pH 6.5", "ph 6 ke aas paas" → ph ≈ 6
- If vague words like "low / medium / high" (or Hindi equivalents) appear, convert using THESE FIXED MAPPINGS:

  NITROGEN (N):
    low / kam = 30
    medium / normal = 60
    high / zyada = 90
  
  PHOSPHORUS (P):
    low / kam = 30
    medium / normal = 50
    high / zyada = 70
  
  POTASSIUM (K):
    low / kam = 20
    medium / normal = 40
    high / zyada = 80

- For temperature, humidity, pH, rainfall:
    - Extract numeric values if present (e.g., "temp 30", "40 degree") → temperature=30 or 40.
    - If vague terms appear, use:
        temperature: thanda/cool=20, garam/warm=30, bahut garam/very hot=35
        rainfall: kam barish=80, normal barish=150, zyada barish/bohot barish=220
        humidity: kam=40, medium=60, zyada=80
    - If not mentioned at all, set to null.

RETURN JSON WITH EXACT KEYS:
{{
    "n": ...,
    "p": ...,
    "k": ...,
    "temperature": ...,
    "humidity": ...,
    "ph": ...,
    "rainfall": ...
}}
"""

    raw = llm.invoke(prompt).content

    try:
        current = json.loads(raw)
    except Exception:
        current = {
            "n": None,
            "p": None,
            "k": None,
            "temperature": None,
            "humidity": None,
            "ph": None,
            "rainfall": None,
        }

    # Merge: new non-null values override old ones
    combined = {}
    for key in prev.keys():
        val_new = current.get(key)
        combined[key] = val_new if val_new is not None else prev.get(key)

    # Compute which important fields are still missing
    missing = [k for k in REQUIRED_KEYS if combined.get(k) is None]
    needs_more_info = len(missing) > 0

    # 🔴 If user clearly says they *cannot* give details, stop asking further
    query_lower = state.query.lower()
    # We check both the original text and lowercased version
    text_variants = [state.query, query_lower]
    if any(phrase in variant for variant in text_variants for phrase in REFUSAL_PATTERNS):
        missing = []
        needs_more_info = False

    return {
        "extracted_params": combined,
        "missing_fields": missing,
        "needs_more_info": needs_more_info,
    }
