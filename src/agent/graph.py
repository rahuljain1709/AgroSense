# src/agent/graph.py

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from agent.schema import AgentState
from agent.parameter_extractor import parse_environment_parameters
from agent.tools import tool_crop_recommendation, tool_rag_retrieve


# ---------- Nodes ----------

def node_extract_params(state: AgentState) -> dict:
    """
    Extract and merge environment parameters, decide if we need more info.
    """
    return parse_environment_parameters(state)


def node_ask_for_more_info(state: AgentState) -> dict:
    """
    Ask user for only the missing fields in a short, conversational way.
    """
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    missing = state.missing_fields or []

    # Map internal keys to farmer-friendly names
    pretty_names = {
        "n": "nitrogen (N)",
        "p": "phosphorus (P)",
        "k": "potassium (K)",
        "ph": "soil pH",
        "temperature": "temperature (°C)",
        "rainfall": "rainfall",
    }
    missing_readable = [pretty_names.get(m, m) for m in missing]

    prompt = f"""
    You are AgroSense, a friendly agronomy assistant.

    The farmer said:
    {state.query}

    We still need these details to give a good crop recommendation:
    {missing_readable}

    The farmer may be speaking in Hindi, Hinglish, or English.
    - If their message is mostly Hindi/Hinglish, reply in simple Hinglish.
    - Otherwise reply in simple English.

    In 1–2 short sentences, ask the farmer to provide ONLY these missing values.
    Use full names like "nitrogen, phosphorus, potassium" instead of letters.
    Keep it brief and conversational.
    """

    resp = llm.invoke(prompt)
    return {"answer": resp.content}


def node_llm_answer(state: AgentState) -> dict:
    """
    Final answer: combine numeric results + RAG into a short, human-like reply.
    """
    load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    prompt = f"""
    You are AgroSense, a friendly agronomy assistant talking to a farmer.

    Farmer's latest message:
    {state.query}

    Known field conditions (from all messages so far):
    {state.extracted_params}

    Top crop candidates with numeric suitability (lower score = better):
    {state.crop_results}

    Relevant agronomy notes (summaries of documents):
    {state.rag_results}

    Write a SHORT, conversational answer:
    - Start with 1–2 best crops and clearly say why they fit.
    - Mention 2–3 practical tips (soil, water, nutrients) in simple words.
    - If some important info is still missing, briefly say what they should measure next.

    Use at most 2–3 short paragraphs or 1 paragraph + a few bullet points.
    Avoid long essays and heavy technical jargon.
    """

    resp = llm.invoke(prompt)
    return {"answer": resp.content}


# ---------- Graph Definition ----------

workflow = StateGraph(AgentState)

workflow.add_node("extract_params", node_extract_params)
workflow.add_node("ask_for_more_info", node_ask_for_more_info)
workflow.add_node("crop_recommender", tool_crop_recommendation)
workflow.add_node("rag_retrieve", tool_rag_retrieve)
workflow.add_node("llm_answer", node_llm_answer)


def route_after_extract(state: AgentState) -> str:
    """
    If we still need key parameters, ask for them.
    Otherwise, proceed to recommendation.
    """
    if state.needs_more_info:
        return "ask_for_more_info"
    return "crop_recommender"


# Entry point
workflow.set_entry_point("extract_params")

# Conditional branch after extraction
workflow.add_conditional_edges(
    "extract_params",
    route_after_extract,
    {
        "ask_for_more_info": "ask_for_more_info",
        "crop_recommender": "crop_recommender",
    },
)

# Normal linear flow for full reasoning
workflow.add_edge("crop_recommender", "rag_retrieve")
workflow.add_edge("rag_retrieve", "llm_answer")
workflow.add_edge("ask_for_more_info", END)
workflow.add_edge("llm_answer", END)

graph = workflow.compile()

