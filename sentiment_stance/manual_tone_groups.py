import os
import pandas as pd
import matplotlib.pyplot as plt
from config import CONFIG

HUMAN_TONE_PALETTE = {
    "neutral":               "#A8DADC",  
    "allowed-with-conditions":"#2E86AB",  
    "risk-awareness":        "#44BBA4",  
    "restrictive":           "#1E3A5F",  
    "allowed":               "#48CAE4",  
}

LABEL_FILE = CONFIG["data"]["sentiment_manual"]
META_FILE = CONFIG["data"]["clean"]

os.makedirs(os.path.dirname(CONFIG["data"]["overall_tone_distribution"]), exist_ok=True)
labels_df = pd.read_excel(LABEL_FILE)
labels_df.columns = ["url", "label"]

meta_df = pd.read_csv(META_FILE)[["url", "country"]]

df = labels_df.merge(meta_df, on="url", how="left")
print(" Data loaded")

# ==============================
# SPLIT MULTI-LABEL CELLS
# ==============================
df["label_list"] = df["label"].astype(str).str.split(",")
df = df.explode("label_list")
df["label_clean"] = df["label_list"].str.strip().str.lower()

# ==============================
# HUMAN SEMANTIC ALIGNMENT
# ==============================
def align_tone(label):
    l = label.lower()

    if any(w in l for w in ["restrictive", "disciplinary", "enforcement", "judicial"]):
        return "restrictive"

    if any(w in l for w in [
        "risk", "caution", "ethical", "integrity", "security",
        "compliance", "responsibility"
    ]):
        return "risk-awareness"

    if any(w in l for w in [
        "policy", "governance", "regulatory", "supervisory",
        "directive", "authoritative", "institutional",
        "guidance", "instructional", "framework", "strategic",
        "brand-protective"
    ]):
        return "allowed-with-conditions"

    if any(w in l for w in [
        "supportive", "enabling", "empowering", "student-friendly",
        "optimistic", "visionary", "inclusive", "promotional",
        "confidence", "trust"
    ]):
        return "allowed"

    if any(w in l for w in [
        "academic", "academically", "research", "educational",
        "informative", "formal", "technical", "explanatory",
        "balanced", "calm", "neutral", "undertone"
    ]):
        return "neutral"

    return "neutral"


df["tone_group"] = df["label_clean"].apply(align_tone)
print(" Semantic alignment complete")

# ==============================
# UNIVERSITY-LEVEL WEIGHTED + SUPPORT AGGREGATION
# (RESTRICTIVE EXEMPTED)
# ==============================
TONE_WEIGHTS = {
    "restrictive": 3.0,
    "allowed-with-conditions": 2.5,
    "risk-awareness": 2.0,
    "allowed": 1.5,
    "neutral": 1.0
}

MIN_SUPPORT_RATIO = 0.30  # applies to all EXCEPT restrictive

def assign_weighted_supported_tone(tone_series):
    counts = tone_series.value_counts()
    total = counts.sum()

    scores = {}

    for tone, weight in TONE_WEIGHTS.items():
        freq = counts.get(tone, 0)

        # Restrictive: presence is sufficient
        if tone == "restrictive" and freq > 0:
            scores[tone] = freq * weight
            continue

        # Other tones: require minimum support
        if freq / total >= MIN_SUPPORT_RATIO:
            scores[tone] = freq * weight

    if scores:
        return max(scores, key=scores.get)
    else:
        return "neutral"


university_tone = (
    df.groupby(["url", "country"])["tone_group"]
      .agg(assign_weighted_supported_tone)
      .reset_index()
)

print(f" Aggregated to {len(university_tone)} universities")

# ==============================
# OVERALL DISTRIBUTION
# ==============================
tone_order = [
    "neutral",
    "allowed-with-conditions",
    "risk-awareness",
    "restrictive",
    "allowed"
]

tone_counts = (
    university_tone["tone_group"]
    .value_counts()
    .reindex(tone_order, fill_value=0)
)

assert tone_counts.sum() == len(university_tone)

plt.figure(figsize=(8, 5))
tone_counts.plot(kind="bar")
plt.title("Overall Policy Tone Distribution (Manual)")
plt.xlabel("Policy Tone")
plt.ylabel("Number of Universities")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(CONFIG["data"]["overall_tone_distribution"], dpi=300)
plt.close()

# ==============================
# COUNTRY-WISE DISTRIBUTION
# ==============================
country_tone = pd.pivot_table(
    university_tone,
    index="country",
    columns="tone_group",
    aggfunc="size",
    fill_value=0
)[tone_order]

country_tone.plot(
    kind="bar",
    stacked=True,
    figsize=(11, 6),
    color=[HUMAN_TONE_PALETTE.get(col, "#A0AEC0") for col in country_tone.columns]
)

plt.title("Country-wise Policy Tone Distribution (Human-annotated, University-Level)")
plt.xlabel("Country")
plt.ylabel("Number of Universities")
plt.xticks(rotation=0)
plt.legend(title="Policy Tone", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(CONFIG["data"]["country_tone_distribution"], dpi=300)
plt.close()
print(" Analysis complete. Restrictive tone preserved and balanced.")
