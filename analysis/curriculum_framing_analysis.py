import pandas as pd
import spacy
from spacy.matcher import Matcher
import seaborn as sns
import matplotlib.pyplot as plt
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
# Visualization
# -----------------------------
expanded = df.assign(
    frame=df["policy_frame"].str.split("; ")
).explode("frame")

plt.figure(figsize=(8, 5))
sns.countplot(
    data=expanded,
    y="frame",
    order=expanded["frame"].value_counts().index,
    palette="Pastel2"
)
plt.title("Distribution of Policy Frames")
plt.xlabel("Number of Universities")
plt.ylabel("Policy Frame")
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/policy_frames.png", dpi=300)
plt.show()
