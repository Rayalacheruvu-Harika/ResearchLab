# ---------------------------------------------
# University-level Multi-Topic Profiling
# ---------------------------------------------

import pandas as pd
import os

# ---------------------------------------------
# Paths
# ---------------------------------------------
TOPIC_RESULTS = "data/bert_topic_model_results.csv"
TOPIC_LABELS = "data/bert_topic_labels.csv"
OUTPUT_FILE = "analysis_results/university_top_topics.csv"

os.makedirs("analysis_results", exist_ok=True)

# ---------------------------------------------
# Load data
# ---------------------------------------------
df_topics = pd.read_csv(TOPIC_RESULTS)
df_labels = pd.read_csv(TOPIC_LABELS)

# ---------------------------------------------
# Validate columns
# ---------------------------------------------
required_topics = {"country", "topic"}
required_labels = {"topic_id", "topic_name"}

if not required_topics.issubset(df_topics.columns):
    raise ValueError(f"Missing columns in topic results: {required_topics}")

if not required_labels.issubset(df_labels.columns):
    raise ValueError(f"Missing columns in label file: {required_labels}")

# ---------------------------------------------
# Merge topic names
# ---------------------------------------------
df = df_topics.merge(
    df_labels,
    left_on="topic",
    right_on="topic_id",
    how="left"
)

df["topic_name"] = df["topic_name"].fillna("Unlabeled")

# ---------------------------------------------
# Count topics per university
# ---------------------------------------------
topic_counts = (
    df.groupby(["country", "topic_name"])
      .size()
      .reset_index(name="count")
)

# ---------------------------------------------
# Rank topics within each university
# ---------------------------------------------
topic_counts["rank"] = (
    topic_counts
    .groupby("country")["count"]
    .rank(method="first", ascending=False)
)

# ---------------------------------------------
# Keep top 3 topics per university
# ---------------------------------------------
top_topics = topic_counts[topic_counts["rank"] <= 3]

# ---------------------------------------------
# Pivot into profile format
# ---------------------------------------------
profile = (
    top_topics
    .sort_values(["country", "rank"])
    .pivot(index="country", columns="rank", values="topic_name")
    .reset_index()
)

profile.columns = [
    "country",
    "primary_topic",
    "secondary_topic",
    "tertiary_topic"
]

# ---------------------------------------------
# Save output
# ---------------------------------------------
profile.to_csv(OUTPUT_FILE, index=False)

print("✓ University-level multi-topic profiling completed")
print("✓ Saved →", OUTPUT_FILE)
