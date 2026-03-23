import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from config import CONFIG

POLICY_TONE_PALETTE = {
    "allowed-with-conditions": "#A8DADC",
    "allowed":                 "#2E86AB",
    "neutral":                 "#44BBA4",
    "risk-awareness":          "#1E3A5F",
    "restrictive":             "#48CAE4",
}

DATA_PATH = CONFIG["data"]["clean"]
OUTPUT_FILE = CONFIG["data"]["policy_tone_distilbert"]
BAR_FIG_PATH = CONFIG["data"]["policy_tone_overall_bar"]
STACKED_FIG_PATH = CONFIG["data"]["policy_tone_stacked_by_country"]

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(os.path.dirname(BAR_FIG_PATH), exist_ok=True)


df = pd.read_csv(DATA_PATH)

# =====================================================
# POLICY TONE LABELS
# =====================================================
LABELS = [
    "allowed-with-conditions",
    "allowed",
    "neutral",
    "risk-awareness",
    "restrictive"
]

# =====================================================
# ZERO-SHOT CLASSIFIER
# =====================================================
classifier = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
    device=-1  # CPU
)

# =====================================================
# RUN CLASSIFICATION
# =====================================================
tone_labels = []

for text in tqdm(df["clean_text"].astype(str).tolist()):
    result = classifier(
        text,
        LABELS,
        multi_label=False
    )
    tone_labels.append(result["labels"][0])

df["policy_tone_label"] = tone_labels

# =====================================================
# SAVE CSV
# =====================================================
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved: {OUTPUT_FILE}")

# =====================================================
# NORMAL BAR CHART (OVERALL POLICY TONE)
# =====================================================
overall_counts = (
    df["policy_tone_label"]
    .value_counts()
    .reindex(LABELS, fill_value=0)
)

overall_counts.plot(
    kind="bar",
    figsize=(8, 5),
    color=[POLICY_TONE_PALETTE.get(l, "#A0AEC0") for l in overall_counts.index]
)

plt.title("Overall Policy Tone Distribution")
plt.xlabel("Policy Tone")
plt.ylabel("Number of Policies")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(BAR_FIG_PATH, dpi=300)
plt.close() 

print(f"Saved: {BAR_FIG_PATH}")

# =====================================================
# STACKED BAR CHART (Country × Policy Tone)
# =====================================================
pivot_df = pd.pivot_table(
    df,
    index="country",
    columns="policy_tone_label",
    aggfunc="size",
    fill_value=0
)

pivot_df = pivot_df.reindex(columns=LABELS, fill_value=0)

pivot_df.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 7),
    color=[POLICY_TONE_PALETTE.get(col, "#A0AEC0") for col in pivot_df.columns]
)

plt.title("Policy Tone Distribution by Country")
plt.xlabel("Country")
plt.ylabel("Count of Policies")
plt.xticks(rotation=45, ha="right")
plt.legend(
    title="Policy Tone",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.tight_layout()

plt.savefig(STACKED_FIG_PATH, dpi=300)
plt.close()
print(f"Saved: {STACKED_FIG_PATH}")