import pandas as pd
import spacy
from spacy.matcher import Matcher
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------
# Load spaCy + Matcher
# -----------------------------
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

matcher.add("Governance", [[{"LOWER": {"IN": ["policy", "regulation", "governance"]}}]])
matcher.add("Pedagogical", [[{"LOWER": {"IN": ["learning", "teaching", "assessment"]}}]])
matcher.add("Threat", [[{"LOWER": {"IN": ["plagiarism", "misconduct", "violation"]}}]])

# -----------------------------
# Paths
# -----------------------------
INPUT = "data/final_clean_dataset.csv"
OUT_DATA = "analysis_results/rq1/policy_frames.csv"
OUT_FIG = "analysis_results/rq1/figures"

os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# -----------------------------
# Frame detection
# -----------------------------
def detect_policy_frames(text):
    doc = nlp(str(text))
    frames = {nlp.vocab[m_id].text for m_id, _, _ in matcher(doc)}
    return "; ".join(frames) if frames else "Unspecified"

# -----------------------------
# Analysis
# -----------------------------
df = pd.read_csv(INPUT)
df["policy_frame"] = df["guideline_text"].apply(detect_policy_frames)
df.to_csv(OUT_DATA, index=False)

# -----------------------------
# 🔑 UNIVERSITY-LEVEL + COUNTRY AGGREGATION
# -----------------------------
expanded = df.assign(
    frame=df["policy_frame"].str.split("; ")
).explode("frame")

plot_df = (
    expanded
    .groupby(["country", "frame"])
    .size()
    .reset_index(name="count")
)

# Pivot for stacked lollipop
pivot = plot_df.pivot(
    index="country",
    columns="frame",
    values="count"
).fillna(0)

# Order frames (consistent)
frames = ["Pedagogical", "Governance", "Threat", "Unspecified"]
pivot = pivot[frames]

# -----------------------------
# VISUALIZATION: STACKED LOLLIPOP
# -----------------------------
colors = {
    "Pedagogical": "#1E3A5F",
    "Governance": "#2E86AB",
    "Threat": "#44BBA4",
    "Unspecified": "#A0AEC0"
}

countries = pivot.index.tolist()
cum = np.zeros(len(countries))

plt.figure(figsize=(9, 4))

for frame in frames:
    values = pivot[frame].values

    plt.hlines(
        y=countries,
        xmin=cum,
        xmax=cum + values,
        linewidth=8,
        color=colors[frame],
        label=frame
    )

    plt.plot(
        cum + values,
        countries,
        "o",
        color=colors[frame],
        markersize=6
    )

    cum += values

plt.xlabel("Number of Universities")
plt.ylabel("Country")
plt.title("University-Level Policy Frames by Country")
plt.legend(title="Policy Frame", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.savefig(f"{OUT_FIG}/policy_frames_stacked_lollipop.png", dpi=300)
plt.show()