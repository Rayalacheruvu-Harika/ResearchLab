import pandas as pd
import ast
import os

# -----------------------------
# Paths
# -----------------------------
BERT_RESULTS = "data/bert_topic_model_results.csv"
TOPIC_LABELS = "data/bert_topic_labels.csv"
OUTPUT = "analysis_results/multi_topic_profiling/university_multi_topic_profiles.csv"

TOP_N = 3  # number of topics per university

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(BERT_RESULTS)
labels = pd.read_csv(TOPIC_LABELS)

# -----------------------------
# Map topic_id → topic_name
# -----------------------------
df = df.merge(
    labels[["topic_id", "topic_name"]],
    left_on="topic",
    right_on="topic_id",
    how="left"
)

# -----------------------------
# Aggregate topics per university
# -----------------------------
topic_counts = (
    df.groupby(["url", "country", "topic_name"])
      .size()
      .reset_index(name="count")
)

# -----------------------------
# Select top N topics per university
# -----------------------------
top_topics = (
    topic_counts.sort_values(["url", "count"], ascending=[True, False])
                .groupby("url")
                .head(TOP_N)
)

# -----------------------------
# Collapse into one row per university
# -----------------------------
final = (
    top_topics.groupby(["url", "country"])["topic_name"]
              .apply(lambda x: "; ".join(sorted(set(x))))
              .reset_index(name="multi_topics")
)

# -----------------------------
# Save
# -----------------------------
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
final.to_csv(OUTPUT, index=False)
print("✅ University-level multi-topic profiling completed")
print("Saved →", OUTPUT)