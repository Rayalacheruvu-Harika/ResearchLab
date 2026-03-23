import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import CONFIG

AFFECTIVE_PALETTE = {
    "Cautionary":           "#2E86AB",
    "Opportunity-Oriented": "#44BBA4",
    "Threat-Oriented":      "#1E3A5F",
}


INPUT = CONFIG["data"]["clean"]
OUT_DATA = CONFIG["data"]["affective_language_university"]
OUT_FIG = "analysis_results/rq1/figures"

os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# -----------------------------
# Affective Lexicon
# -----------------------------
AFFECTIVE_LEXICON = {
    "Threat-Oriented": [
        "risk", "misuse", "violation", "misconduct", "plagiarism",
        "unauthorized", "cheating", "penalty", "sanction", "breach"
    ],
    "Opportunity-Oriented": [
        "support", "enhance", "assist", "improve", "innovation",
        "opportunity", "augment", "enable", "explore"
    ],
    "Cautionary": [
        "responsible", "appropriate", "careful", "guidance",
        "ethical", "transparent"
    ]
}

def score_affect(text):
    text = str(text).lower()
    return {k: sum(text.count(w) for w in v) for k, v in AFFECTIVE_LEXICON.items()}

df = pd.read_csv(INPUT)

# -----------------------------
# Document-level affect scoring
# -----------------------------
scores = df["guideline_text"].apply(lambda x: pd.Series(score_affect(x)))
df = pd.concat([df, scores], axis=1)

# -----------------------------
# 🔑 University-level aggregation
# -----------------------------
uni_affect = (
    df.groupby(["url", "country"])[list(AFFECTIVE_LEXICON.keys())]
      .sum()
      .reset_index()
)

uni_affect["dominant_affect"] = uni_affect[
    list(AFFECTIVE_LEXICON.keys())
].idxmax(axis=1)

# Save university-level output
uni_affect.to_csv(CONFIG["data"]["affective_language_university"], index=False)
print(f"Saved: {CONFIG['data']['affective_language_university']}")

# ============================================================
# VISUALIZATION 1: Overall affective framing (University level)
# ============================================================
plt.figure(figsize=(8, 5))
sns.countplot(
    data=uni_affect,
    y="dominant_affect",
    order=uni_affect["dominant_affect"].value_counts().index,
    hue="dominant_affect",
    palette=AFFECTIVE_PALETTE,
    legend=False
)
plt.title("Overall Affective Framing of LLM Policies (University Level)")
plt.xlabel("Number of Universities")
plt.ylabel("Affective Tone")
plt.tight_layout()
plt.savefig(CONFIG["data"]["affective_overall_university"], dpi=300)
plt.close()


# ============================================================
# VISUALIZATION 2: Dominant affect by country (University level)
# ============================================================
dominant_country = (
    uni_affect.groupby(["country", "dominant_affect"])
              .size()
              .reset_index(name="count")
)

dominant_country = dominant_country.loc[
    dominant_country.groupby("country")["count"].idxmax()
]

plt.figure(figsize=(10, 5))
sns.barplot(
    data=dominant_country,
    x="country",
    y="count",
    hue="dominant_affect",
    palette=AFFECTIVE_PALETTE
)
plt.title("Dominant Affective Tone by Country (University Level)")
plt.xlabel("Country")
plt.ylabel("Number of Universities")
plt.tight_layout()
plt.savefig(CONFIG["data"]["affective_by_country_university"], dpi=300)
plt.close()

# ============================================================
# 🔹 NEW VISUAL 3: STACKED BAR (Country × Affect)
# ============================================================
stacked = (
    uni_affect.groupby(["country", "dominant_affect"])
              .size()
              .unstack(fill_value=0)
)

stacked.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=[AFFECTIVE_PALETTE.get(col, "#999999") for col in stacked.columns]
)
plt.title("Distribution of Affective Tone Across Universities by Country")
plt.xlabel("Country")
plt.ylabel("Number of Universities")
plt.legend(title="Affective Tone")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(CONFIG["data"]["affective_stacked_university"], dpi=300)
plt.close()
