# src/rag/test_retriever.py

from rag.retriever import retrieve_agri_docs


def main():
    query = "Which crop prefers flooded fields and high water availability?"
    print("Query:", query)

    results = retrieve_agri_docs(query, k=3)
    print("\nTop 3 results (no filter):")
    for i, r in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Source:", r["source"])
        print("Topic :", r["topic"])
        print(r["content"][:300], "...")
    
    # Example with topic filter for rice
    print("\n\nNow with topic_filter='rice':")
    results_rice = retrieve_agri_docs(query, k=3, topic_filter="rice")
    for i, r in enumerate(results_rice, start=1):
        print(f"\n--- Rice Result {i} ---")
        print("Source:", r["source"])
        print("Topic :", r["topic"])
        print(r["content"][:300], "...")


if __name__ == "__main__":
    main()
