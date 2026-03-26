import pandas as pd
import seaborn as sns
import spacy
from spacy.matcher import Matcher
import matplotlib.pyplot as plt
import numpy as np
import os
from config import CONFIG
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
INPUT = CONFIG["data"]["clean"]
OUT_DATA = CONFIG["data"]["policy_frames"]
OUT_FIG = os.path.dirname(CONFIG["data"]["policy_frames_figure"])

os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

def detect_policy_frames(text):
    doc = nlp(str(text))
    frames = {nlp.vocab[m_id].text for m_id, _, _ in matcher(doc)}
    return "; ".join(frames) if frames else "Unspecified"

# -----------------------------
# Analysis
# -----------------------------
df = pd.read_csv(INPUT)
df["policy_frame"] = df["guideline_text"].apply(detect_policy_frames)
df.to_csv(CONFIG["data"]["policy_frames"], index=False)

# -----------------------------
# Visualization
# -----------------------------
# -----------------------------
# University-level + Country aggregation
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

pivot = plot_df.pivot(
    index="country",
    columns="frame",
    values="count"
).fillna(0)

frames = ["Pedagogical", "Governance", "Threat", "Unspecified"]
pivot = pivot[[f for f in frames if f in pivot.columns]]

# -----------------------------
# Stacked Lollipop Chart
# -----------------------------
colors = {
    "Pedagogical": "#1E3A5F",
    "Governance":  "#2E86AB",
    "Threat":      "#44BBA4",
    "Unspecified": "#A0AEC0"
}

countries = pivot.index.tolist()
cum = np.zeros(len(countries))

plt.figure(figsize=(9, 4))

for frame in pivot.columns:
    values = pivot[frame].values
    plt.hlines(y=countries, xmin=cum, xmax=cum + values,
               linewidth=8, color=colors[frame], label=frame)
    plt.plot(cum + values, countries, "o", color=colors[frame], markersize=6)
    cum += values

plt.xlabel("Number of Universities")
plt.ylabel("Country")
plt.title("University-Level Policy Frames by Country")
plt.legend(title="Policy Frame", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(CONFIG["data"]["policy_frames_lollipop"], dpi=300)
plt.close()
# -----------------------------
# Simple Bar Chart (Overall)
# -----------------------------


plt.figure(figsize=(8, 5))
sns.countplot(
    data=expanded,
    y="frame",
    hue="frame",
    order=expanded["frame"].value_counts().index,
    palette=colors,
    legend=False
)
plt.title("Distribution of Policy Frames")
plt.xlabel("Number of Universities")
plt.ylabel("Policy Frame")
plt.tight_layout()
plt.savefig(CONFIG["data"]["policy_frames_figure"], dpi=300)
plt.close()
print(f"Saved: {CONFIG['data']['policy_frames']}")
print(f"Saved: {CONFIG['data']['policy_frames_lollipop']}")
print(f"Saved: {CONFIG['data']['policy_frames_figure']}")