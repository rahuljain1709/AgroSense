# src/rag/build_index.py

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma


# project root = two levels up from this file: ...\agro_sense_ai
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS_DIR = os.path.join(ROOT, "data", "docs")
DB_DIR = os.path.join(ROOT, "data", "chroma_db")



def load_text_docs():
    """Load all .txt files from data/docs into LangChain Document objects."""
    docs = []
    for fname in os.listdir(DOCS_DIR):
        if not fname.lower().endswith(".txt"):
            continue

        path = os.path.join(DOCS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Use filename (without extension) as a simple 'topic' / 'crop' label
        base = os.path.splitext(fname)[0]

        metadata = {
            "source": fname,
            "topic": base,           # e.g., "rice", "soil_basics"
        }

        docs.append(Document(page_content=text, metadata=metadata))

    return docs


def build_chroma_index():
    print("Loading environment variables...")
    load_dotenv()

    print(f"Loading docs from: {DOCS_DIR}")
    docs = load_text_docs()
    print(f"Total documents loaded: {len(docs)}")

    if not docs:
        raise ValueError("No .txt files found in data/docs")

    print("Creating embeddings object...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Remove old DB if you want a fresh build each time
    if os.path.exists(DB_DIR):
        print(f"Removing existing DB directory at: {DB_DIR}")
        import shutil
        shutil.rmtree(DB_DIR)

    print(f"Building Chroma DB at: {DB_DIR}")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name="agro_docs",
    )

    vectordb.persist()
    print("Chroma DB built and persisted successfully.")


if __name__ == "__main__":
    build_chroma_index()
