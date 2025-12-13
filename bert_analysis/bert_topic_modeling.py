# -----------------------------
# Topic Modeling using BERTopic
# -----------------------------

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import os

# -----------------------------
# Paths
# -----------------------------
INPUT_FILE = "data/final_clean_dataset.csv"
OUTPUT_TOPICS_FILE = "data/bert_topic_model_results.csv"
OUTPUT_SUMMARY_FILE = "data/bert_topic_summary.csv"
MODEL_SAVE_DIR = "models/bertopic_model"

# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv(INPUT_FILE)

# Use the cleaned text
texts = df["clean_text"].astype(str).tolist()

print(f"Loaded {len(texts)} documents for topic modeling.")

# -----------------------------
# Sentence Embeddings Model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Create BERTopic Model
# -----------------------------
topic_model = BERTopic(
    min_topic_size=3,
    nr_topics="auto",
    embedding_model=embedding_model,
    verbose=True
)

# -----------------------------
# Fit the model
# -----------------------------
topics, probabilities = topic_model.fit_transform(texts)

# Save results to dataframe
df["topic"] = topics
# ----------------------------------------------------
# Safe probability extraction (fix for axis error)
# ----------------------------------------------------
try:
    if probabilities is None:
        df["topic_probability"] = 0.0
    elif isinstance(probabilities, list):
        df["topic_probability"] = probabilities
    elif probabilities.ndim == 1:
        df["topic_probability"] = probabilities
    else:
        df["topic_probability"] = probabilities.max(axis=1)
except Exception as e:
    print("Probability extraction failed:", e)
    df["topic_probability"] = 0.0


df.to_csv(OUTPUT_TOPICS_FILE, index=False, encoding="utf-8")
print(f"Saved topic assignments → {OUTPUT_TOPICS_FILE}")

# -----------------------------
# Topic Summary Table
# -----------------------------
topic_summary = topic_model.get_topic_info()
topic_summary.to_csv(OUTPUT_SUMMARY_FILE, index=False)
print(f"Saved topic summary → {OUTPUT_SUMMARY_FILE}")

# -----------------------------
# Save Model
# -----------------------------
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
# -----------------------------
# Save Model Safely
# -----------------------------
import os

if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

# Save the model inside the folder as a file
topic_model.save(f"{MODEL_SAVE_DIR}/bertopic_model.pkl")

print(f"BERTopic model saved to → {MODEL_SAVE_DIR}")

print("\nTopic Modeling Completed Successfully!")
