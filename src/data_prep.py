# src/data_prep.py

import os
from dotenv import load_dotenv
import pandas as pd

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_PATH = os.path.join(ROOT, "data", "crop_recommendation_raw.csv")
OUTPUT_PATH = os.path.join(ROOT, "data", "crop_profiles.csv")

def main():
    print("Loading dataset from:", RAW_DATA_PATH)
    df = pd.read_csv(RAW_DATA_PATH)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Group by crop label and compute averaged ideal conditions
    grouped = df.groupby("label").agg(
        ideal_n=("N", "mean"),
        ideal_p=("P", "mean"),
        ideal_k=("K", "mean"),
        ideal_temp=("temperature", "mean"),
        ideal_humidity=("humidity", "mean"),
        ideal_ph=("ph", "mean"),
        ideal_rainfall=("rainfall", "mean"),
    ).reset_index().rename(columns={"label": "crop"})

    # Add placeholder columns for future extension
    grouped["soil_types"] = "unknown"
    grouped["season"] = "unknown"
    grouped["description"] = "No description added yet."

    # Save the processed crop profiles
    grouped.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved crop profiles to:", OUTPUT_PATH)
    print("\nPreview:")
    print(grouped.head(10))

if __name__ == "__main__":
    load_dotenv()
    main()
