import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/final_clean_dataset.csv"

# FIX: separate directory and file paths
OUTPUT_DIR = "analysis_results/sentiment_analysis"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sentiment_results.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "sentiment_summary.csv")

# create directory ONLY
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_COLUMN = "clean_text"

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading cleaned policy data...")
df = pd.read_csv(DATA_PATH)

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"Column '{TEXT_COLUMN}' not found in dataset")

# -----------------------------
# LOAD SENTIMENT MODEL
# -----------------------------
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment",
    truncation=True,
    max_length=512
)

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# -----------------------------
# RUN SENTIMENT ANALYSIS (BATCH)
# -----------------------------
print("Running sentiment analysis...")

results = sentiment_pipeline(
    df[TEXT_COLUMN].astype(str).tolist(),
    batch_size=16,
    truncation=True
)

df["sentiment_label"] = [label_map[r["label"]] for r in results]
df["sentiment_score"] = [r["score"] for r in results]

def confidence_level(score):
    if score >= 0.85:
        return "High"
    elif score >= 0.65:
        return "Medium"
    else:
        return "Low"

df["sentiment_confidence"] = df["sentiment_score"].apply(confidence_level)

# -----------------------------
# SAVE RESULTS
# -----------------------------
df.to_csv(OUTPUT_PATH, index=False)
print("Sentiment analysis completed!")
print(f"Results saved to: {os.path.abspath(OUTPUT_PATH)}")

# -----------------------------
# SENTIMENT SUMMARY
# -----------------------------
summary = (
    df["sentiment_label"]
    .value_counts()
    .reset_index()
)

summary.columns = ["sentiment_label", "count"]
summary["percentage"] = (summary["count"] / summary["count"].sum()) * 100

summary.to_csv(SUMMARY_PATH, index=False)
print("Sentiment summary saved to:", os.path.abspath(SUMMARY_PATH))