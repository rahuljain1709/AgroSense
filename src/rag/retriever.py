# src/rag/retriever.py

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# project root = .../agro_sense_ai
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB_DIR = os.path.join(ROOT, "data", "chroma_db")
COLLECTION_NAME = "agro_docs"


def _load_vectordb():
    """
    Internal helper to load the persisted Chroma DB.
    Call this inside your tools / LangGraph nodes.
    """
    load_dotenv()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME,
    )
    return vectordb


def retrieve_agri_docs(query: str, k: int = 5, topic_filter: str | None = None):
    """
    Main retrieval function for AgroSense.

    Args:
        query: natural language question or description.
        k: number of documents to return.
        topic_filter: optional crop/topic filter, e.g. "rice", "maize".
                     If provided, only docs with metadata["topic"] == topic_filter are considered.

    Returns:
        List of dicts: [{ "content": ..., "source": ..., "topic": ... }, ...]
    """
    vectordb = _load_vectordb()

    search_kwargs = {}
    if topic_filter:
        # Filter uses metadata keys; we stored "topic" in build_index.py
        search_kwargs["filter"] = {"topic": topic_filter}

    docs = vectordb.similarity_search(query, k=k, **search_kwargs)

    results = []
    for d in docs:
        results.append(
            {
                "content": d.page_content,
                "source": d.metadata.get("source"),
                "topic": d.metadata.get("topic"),
            }
        )

    return results
