import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency

# -------------------------------------
# File Paths
# -------------------------------------
CLEAN_DATA = "data/final_clean_dataset.csv"
TOPIC_DATA = "data/bert_topic_model_results.csv"
LABELS_DATA = "data/bert_topic_labels.csv"

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------
# Load Data
# -------------------------------------
df_clean = pd.read_csv(CLEAN_DATA)
df_topics = pd.read_csv(TOPIC_DATA)
df_labels = pd.read_csv(LABELS_DATA)

# -------------------------------------
# Validate that topic labels file has correct columns
# -------------------------------------
required_cols = {"topic_id", "topic_name"}
if not required_cols.issubset(df_labels.columns):
    raise ValueError(f"labels file must contain columns: {required_cols}")

# -------------------------------------
# Merge Clean Data + Topic IDs + Human Topic Names
# -------------------------------------
df = (
    df_clean.merge(df_topics[["url", "topic"]], on="url", how="left")
            .merge(df_labels[["topic_id", "topic_name"]],
                   left_on="topic", right_on="topic_id", how="left")
)

df["topic_name"] = df["topic_name"].fillna("Unlabeled Topic")

# -------------------------------------
# 1. Country × Topic (Counts)
# -------------------------------------
pivot_counts = df.pivot_table(
    index="country",
    columns="topic_name",
    values="clean_text",
    aggfunc="count",
    fill_value=0
)

pivot_counts = pivot_counts.reindex(sorted(pivot_counts.columns), axis=1)
pivot_counts.to_csv(f"{OUTPUT_DIR}/country_topic_counts.csv")

# -------------------------------------
# 2. Country × Topic (Proportions)
# -------------------------------------
pivot_props = pivot_counts.div(pivot_counts.sum(axis=1).replace(0, 1), axis=0)
pivot_props.to_csv(f"{OUTPUT_DIR}/country_topic_proportions.csv")

# -------------------------------------
# 3. Heatmap — Proportions
# -------------------------------------
plt.figure(figsize=(20, 10))
sns.heatmap(pivot_props, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
plt.title("Country vs Topic Distribution (Proportions)", fontsize=18)
plt.ylabel("Country")
plt.xlabel("Topic")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_topics_proportions.png", dpi=300)
plt.show()

# -------------------------------------
# 4. Heatmap — Raw Counts
# -------------------------------------
plt.figure(figsize=(20, 10))
sns.heatmap(pivot_counts, annot=True, cmap="Greens", fmt="d", linewidths=0.5)
plt.title("Country vs Topic Distribution (Counts)", fontsize=18)
plt.ylabel("Country")
plt.xlabel("Topic")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_topics_counts.png", dpi=300)
plt.show()

# -------------------------------------
# 5. Bar Chart of Topic Frequency
# -------------------------------------
topic_counts = df["topic_name"].value_counts()

plt.figure(figsize=(14, 8))
sns.barplot(y=topic_counts.index, x=topic_counts.values, palette="viridis")
plt.title("Overall Topic Frequency Across All Universities", fontsize=18)
plt.xlabel("Count")
plt.ylabel("Topic")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/topic_frequency_bar.png", dpi=300)
plt.show()

print("\n✔ Country–Topic Distribution Analysis Completed Successfully!")
