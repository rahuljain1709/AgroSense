# src/check_env.py
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
import chromadb

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY loaded:", bool(api_key))

    # Simple test client (won't actually call the API yet)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    print("LLM object created:", llm is not None)

    client = chromadb.Client()
    print("Chroma client created:", client is not None)

if __name__ == "__main__":
    main()
