# src/agent/tools.py

from agent.schema import AgentState
from tools.crop_recommender import load_crop_profiles
from rag.retriever import retrieve_agri_docs


def tool_crop_recommendation(state: AgentState) -> dict:
    """
    Use extracted_params from the state to score crops.
    Missing values are ignored instead of filled with defaults.
    """
    params = state.extracted_params or {}

    user_vec = {}

    mapping = {
        "n": "ideal_n",
        "p": "ideal_p",
        "k": "ideal_k",
        "temperature": "ideal_temp",
        "humidity": "ideal_humidity",
        "ph": "ideal_ph",
        "rainfall": "ideal_rainfall",
    }

    for key, target in mapping.items():
        val = params.get(key)
        if val is not None:
            user_vec[target] = float(val)

    df = load_crop_profiles()

    scored = []
    for _, row in df.iterrows():
        ideal = {
            "ideal_n": row["ideal_n"],
            "ideal_p": row["ideal_p"],
            "ideal_k": row["ideal_k"],
            "ideal_temp": row["ideal_temp"],
            "ideal_humidity": row["ideal_humidity"],
            "ideal_ph": row["ideal_ph"],
            "ideal_rainfall": row["ideal_rainfall"],
        }

        score = 0.0
        for key, user_val in user_vec.items():
            ideal_val = ideal.get(key)
            if ideal_val is not None:
                score += abs(user_val - ideal_val)

        scored.append((row["crop"], score))

    scored_sorted = sorted(scored, key=lambda x: x[1])
    results = [{"crop": c, "score": s} for c, s in scored_sorted[:5]]

    return {"crop_results": results}


def tool_rag_retrieve(state: AgentState) -> dict:
    """
    Query the RAG index using the user's query text.
    """
    docs = retrieve_agri_docs(state.query, k=5)
    return {"rag_results": docs}

