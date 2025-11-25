# src/tools/crop_recommender.py

import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CROP_PROFILES_PATH = os.path.join(ROOT, "data", "crop_profiles.csv")


def load_crop_profiles():
    if not os.path.exists(CROP_PROFILES_PATH):
        raise FileNotFoundError(f"crop_profiles.csv not found at {CROP_PROFILES_PATH}")

    df = pd.read_csv(CROP_PROFILES_PATH)
    return df


def score_crop(user_values, ideal_values):
    """
    Simple similarity scoring between user conditions vs crop ideal conditions.
    Lower difference = better.
    """
    score = 0
    for key in user_values:
        if key in ideal_values:
            diff = abs(user_values[key] - ideal_values[key])
            score += diff
    return score


def recommend_crops(n, p, k, temperature, humidity, ph, rainfall, top_k=5):
    df = load_crop_profiles()

    user = {
        "ideal_n": n,
        "ideal_p": p,
        "ideal_k": k,
        "ideal_temp": temperature,
        "ideal_humidity": humidity,
        "ideal_ph": ph,
        "ideal_rainfall": rainfall,
    }

    scores = []

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

        s = score_crop(user, ideal)
        scores.append((row["crop"], s))

    # Sort: lowest difference is best match
    scores_sorted = sorted(scores, key=lambda x: x[1])

    return scores_sorted[:top_k]
