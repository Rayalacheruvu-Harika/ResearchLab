import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2_contingency

# ------------------------------
# File paths (EDIT IF NEEDED)
# ------------------------------
CLEAN_DATA = "../data/final_clean_dataset.csv"
TOPIC_DATA = "../data/topic_model_results.csv"
LABELS_DATA = "../data/topic_labels.csv"

OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# Load Data
# ------------------------------
df_clean = pd.read_csv(CLEAN_DATA)
df_topics = pd.read_csv(TOPIC_DATA)
df_labels = pd.read_csv(LABELS_DATA)

# Validate required columns
required_clean = {"url", "country", "clean_text"}
required_topics = {"url", "topic"}
required_labels = {"topic_id", "assigned_label"}

if not required_clean.issubset(df_clean.columns):
    raise ValueError(f"Missing columns in CLEAN_DATA: {required_clean - set(df_clean.columns)}")

if not required_topics.issubset(df_topics.columns):
    raise ValueError(f"Missing columns in TOPIC_DATA: {required_topics - set(df_topics.columns)}")

if not required_labels.issubset(df_labels.columns):
    raise ValueError(f"Missing columns in LABELS_DATA: {required_labels - set(df_labels.columns)}")

# Merge datasets
df = (
    df_clean.merge(df_topics[["url", "topic"]], on="url", how="left")
            .merge(df_labels[["topic_id", "assigned_label"]], 
                   left_on="topic", right_on="topic_id", how="left")
)

# Ensure no missing labels
df["assigned_label"].fillna("Unlabeled", inplace=True)

# ------------------------------
# 1. Country–Topic Distribution (Raw Counts)
# ------------------------------
pivot_counts = df.pivot_table(
    index="country",
    columns="assigned_label",
    values="clean_text",
    aggfunc="count",
    fill_value=0
)

pivot_counts.to_csv(f"{OUTPUT_DIR}/country_topic_counts.csv")
print("✔ Saved → country_topic_counts.csv")

# ------------------------------
# 2. Country–Topic Proportion Matrix (Normalized)
# ------------------------------
pivot_props = pivot_counts.div(pivot_counts.sum(axis=1), axis=1)

pivot_props.to_csv(f"{OUTPUT_DIR}/country_topic_proportions.csv")
print("✔ Saved → country_topic_proportions.csv")

# ------------------------------
# 3. Heatmap: Normalized Topic Proportions
# ------------------------------
plt.figure(figsize=(16, 9))
sns.heatmap(pivot_props, annot=True, cmap="Blues", fmt=".2f")
plt.title("Country vs Topic Distribution (Proportions)", fontsize=16)
plt.ylabel("Country")
plt.xlabel("Topic")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/country_topic_heatmap_proportions.png", dpi=300)
plt.show()

print("✔ Saved → country_topic_heatmap_proportions.png")

# ------------------------------
# 4. Heatmap: Raw Counts
# ------------------------------
plt.figure(figsize=(16, 9))
sns.heatmap(pivot_counts, annot=True, cmap="Greens", fmt="d")
plt.title("Country vs Topic Distribution (Raw Counts)", fontsize=16)
plt.ylabel("Country")
plt.xlabel("Topic")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/country_topic_heatmap_counts.png", dpi=300)
plt.show()

print("✔ Saved → country_topic_heatmap_counts.png")

# ------------------------------
# 5. Overall Topic Frequency Bar Chart
# ------------------------------
topic_counts = df["assigned_label"].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(y=topic_counts.index, x=topic_counts.values, palette="viridis")
plt.title("Overall Topic Frequency Across All Universities", fontsize=16)
plt.xlabel("Count")
plt.ylabel("Topic")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/topic_frequency_bar.png", dpi=300)
plt.show()

print("✔ Saved → topic_frequency_bar.png")

# ------------------------------
# 6. Topic Ranking for Each Country
# ------------------------------
ranking_output = f"{OUTPUT_DIR}/country_topic_rankings.txt"

with open(ranking_output, "w") as f:
    for country in pivot_counts.index:
        f.write(f"\n===== {country} =====\n")
        ranked = pivot_counts.loc[country].sort_values(ascending=False)
        for topic, count in ranked.items():
            f.write(f"{topic}: {count}\n")

print("✔ Saved → country_topic_rankings.txt")

# ------------------------------
# 7. Chi-Square Test (Country x Topic)
# ------------------------------
chi2, p, dof, expected = chi2_contingency(pivot_counts)

chi_output = f"{OUTPUT_DIR}/chi_square_test.txt"
with open(chi_output, "w") as f:
    f.write(f"Chi-Square Statistic: {chi2}\n")
    f.write(f"p-value: {p}\n")
    f.write(f"Degrees of freedom: {dof}\n")

print("✔ χ² Test Saved → chi_square_test.txt")

# ------------------------------
# 8. Save final merged dataset
# ------------------------------
df.to_csv(f"{OUTPUT_DIR}/final_topic_country_dataset.csv", index=False)
print("✔ Saved → final_topic_country_dataset.csv")

print("\n🎉 FULL COUNTRY–TOPIC ANALYSIS COMPLETED SUCCESSFULLY!")
