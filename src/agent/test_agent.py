# src/agent/test_agent.py

from agent.graph import graph


def main():
    # Try changing this text to see how extraction + recommendation reacts
    user_input = (
        "My soil has medium nitrogen and low phosphorus. "
        "Temperature is around 30 degrees, rainfall is high, "
        "and pH is close to 6."
    )

    result = graph.invoke({"query": user_input})

    print("\n=== USER QUERY ===")
    print(user_input)

    print("\n=== EXTRACTED PARAMETERS ===")
    print(result.get("extracted_params"))

    print("\n=== TOP CROP RECOMMENDATIONS (lower score = better) ===")
    for idx, item in enumerate(result.get("crop_results", []), start=1):
        print(f"{idx}. {item['crop']} (score={item['score']:.2f})")

    print("\n=== FINAL ANSWER ===\n")
    print(result.get("answer", ""))


if __name__ == "__main__":
    main()

