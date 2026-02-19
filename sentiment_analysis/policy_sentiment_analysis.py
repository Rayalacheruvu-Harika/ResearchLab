import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import chi2_contingency

# =====================================================
# PATHS
# =====================================================
INPUT_FILE = "data/final_clean_dataset.csv"
OUT_DATA_FILE = "analysis_results/rq3/policy_sentiment_university.csv"
OUT_FIG_DIR = "analysis_results/rq3/figures"

os.makedirs(os.path.dirname(OUT_DATA_FILE), exist_ok=True)
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# =====================================================
# POLICY-ORIENTED SENTIMENT LABELS
# (Delegated MERGED into Conditional)
# =====================================================
POLICY_SENTIMENT = {
    "Supportive": [
        "encourage", "support", "enable", "enhance", "explore",
        "innovation", "assist", "augment", "opportunity"
    ],
    "Conditional": [
        "provided", "provided that", "subject to", "must",
        "guidelines", "appropriate", "responsible", "ethical",
        "in accordance", "with permission",
        # delegation cues merged here
        "instructor decides", "faculty decide",
        "course instructor", "module instructor",
        "syllabus", "course level", "departmental policy",
        "refer to instructor"
    ],
    "Restrictive": [
        "prohibited", "not allowed", "unauthorized", "misconduct",
        "violation", "plagiarism", "penalty", "sanction", "breach"
    ],
    "Risk-Focused": [
        "risk", "risks", "threat", "threats", "harm",
        "misuse", "danger", "concern", "academic integrity risk"
    ],
    "Neutral": [
        "policy", "procedure", "definition", "applies",
        "outlined", "described", "refer to"
    ]
}

LABEL_ORDER = [
    "Supportive",
    "Conditional",
    "Restrictive",
    "Risk-Focused",
    "Neutral"
]

# =====================================================
# SENTIMENT SCORING FUNCTION
# =====================================================
def score_policy_sentiment(text):
    text = str(text).lower()
    return {
        label: sum(text.count(term) for term in terms)
        for label, terms in POLICY_SENTIMENT.items()
    }

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(INPUT_FILE)

# =====================================================
# DOCUMENT-LEVEL SENTIMENT SCORING
# =====================================================
sentiment_df = df["guideline_text"].apply(
    lambda t: pd.Series(score_policy_sentiment(t))
)

df = pd.concat([df, sentiment_df], axis=1)

# =====================================================
# UNIVERSITY-LEVEL AGGREGATION
# =====================================================
uni_sentiment = (
    df.groupby(["url", "country"])[LABEL_ORDER]
      .sum()
      .reset_index()
)

uni_sentiment["dominant_sentiment"] = uni_sentiment[LABEL_ORDER].idxmax(axis=1)

# Save university-level results
uni_sentiment.to_csv(OUT_DATA_FILE, index=False)
print(f"✓ Saved university-level policy sentiment → {OUT_DATA_FILE}")

# =====================================================
# COUNTRY-LEVEL % STACKED BAR CHART
# =====================================================
country_counts = (
    uni_sentiment.groupby(["country", "dominant_sentiment"])
    .size()
    .unstack(fill_value=0)
)

country_counts = country_counts.reindex(columns=LABEL_ORDER, fill_value=0)

country_pct = country_counts.div(
    country_counts.sum(axis=1), axis=0
) * 100

country_pct.plot(
    kind="bar",
    stacked=True,
    figsize=(11, 6)
)

plt.title(
    "Policy-Oriented Sentiment of LLM Policies by Country\n(University Level)"
)
plt.xlabel("Country")
plt.ylabel("Percentage of Universities (%)")
plt.xticks(rotation=0)
plt.legend(title="Policy Sentiment", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(
    f"{OUT_FIG_DIR}/policy_sentiment_by_country_percentage.png",
    dpi=300
)
plt.show()

# =====================================================
# OVERALL SENTIMENT DISTRIBUTION (PERCENTAGE)
# =====================================================
overall_counts = (
    uni_sentiment["dominant_sentiment"]
    .value_counts()
    .reindex(LABEL_ORDER, fill_value=0)
)

overall_pct = overall_counts / overall_counts.sum() * 100

plt.figure(figsize=(8, 5))
overall_pct.plot(kind="bar")

plt.title(
    "Overall Policy-Oriented Sentiment of LLM Policies\n(University Level)"
)
plt.xlabel("Policy Sentiment")
plt.ylabel("Percentage of Universities (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    f"{OUT_FIG_DIR}/policy_sentiment_overall_percentage.png",
    dpi=300
)
plt.show()

# =====================================================
# STATISTICAL VALIDATION
# =====================================================
print("\n=== STATISTICAL VALIDATION ===")

contingency = pd.crosstab(
    uni_sentiment["country"],
    uni_sentiment["dominant_sentiment"]
).reindex(columns=LABEL_ORDER, fill_value=0)

chi2, p_value, dof, expected = chi2_contingency(contingency)

print(f"Chi-square statistic: {chi2:.3f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.5f}")

# Effect size: Cramér’s V
n = contingency.values.sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

print(f"Cramér’s V: {cramers_v:.3f}")

if p_value < 0.05:
    print("✓ Sentiment distributions differ significantly across countries.")
else:
    print("✓ No statistically significant difference in sentiment distributions across countries.")
