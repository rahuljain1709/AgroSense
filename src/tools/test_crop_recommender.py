from tools.crop_recommender import recommend_crops


def main():
    print("Testing crop recommender...")

    recs = recommend_crops(
        n=80,
        p=40,
        k=40,
        temperature=28,
        humidity=75,
        ph=6.5,
        rainfall=200,
        top_k=5
    )

    for crop, score in recs:
        print(f"{crop} (score={score:.2f})")


if __name__ == "__main__":
    main()
