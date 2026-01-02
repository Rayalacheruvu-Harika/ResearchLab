# -------------------------------------------------
# Chi-Square Test: Country × Topic Distribution
# -------------------------------------------------

import pandas as pd
from scipy.stats import chi2_contingency
import os

# -------------------------------------------------
# Paths
# -------------------------------------------------
TOPIC_RESULTS = "data/bert_topic_model_results.csv"
TOPIC_LABELS = "data/bert_topic_labels.csv"
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Load data
# -------------------------------------------------
df_topics = pd.read_csv(TOPIC_RESULTS)
df_labels = pd.read_csv(TOPIC_LABELS)

# -------------------------------------------------
# Merge topic names
# -------------------------------------------------
df = df_topics.merge(
    df_labels,
    left_on="topic",
    right_on="topic_id",
    how="left"
)

df["topic_name"] = df["topic_name"].fillna("Unlabeled")

# -------------------------------------------------
# Build contingency table
# -------------------------------------------------
contingency = pd.crosstab(
    df["country"],
    df["topic_name"]
)

contingency.to_csv(f"{OUTPUT_DIR}/chi_square_contingency_table.csv")

# -------------------------------------------------
# Run Chi-square test
# -------------------------------------------------
chi2, p_value, dof, expected = chi2_contingency(contingency)

# Save results
with open(f"{OUTPUT_DIR}/chi_square_results.txt", "w") as f:
    f.write("Chi-Square Test: Country × Topic Distribution\n")
    f.write("------------------------------------------------\n")
    f.write(f"Chi-square statistic: {chi2:.4f}\n")
    f.write(f"Degrees of freedom: {dof}\n")
    f.write(f"p-value: {p_value:.6f}\n")

print("\n✅ Chi-square test completed")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.6f}")
print("Results saved to analysis_results/")
