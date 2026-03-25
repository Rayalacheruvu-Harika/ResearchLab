import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import CONFIG

FRAME_PALETTE = {
    "Regulatory / Governance":  "#1E3A5F",
    "Pedagogical Support":      "#2E86AB",
    "Technological Enablement": "#44BBA4",
    "Risk & Compliance":        "#A8DADC",
    "Control & Integrity":      "#48CAE4",
    "Unmapped Frame": "#A0AEC0"
}

# ---------------------------------
# Paths
# ---------------------------------
INPUT_FILE = CONFIG["data"]["university_framing_profiles"]
OUTPUT_DIR = "analysis_results/framing"
os.makedirs(OUTPUT_DIR, exist_ok=True)


df = pd.read_csv(INPUT_FILE)

# ---------------------------------
# Expand multi-frame column
# ---------------------------------
df_exp = df.assign(
    frame=df["frame"].str.split("; ")
).explode("frame")


# =================================
# B. Overall Frame Distribution
# =================================
frame_counts = df_exp["frame"].value_counts().reset_index()
frame_counts.columns = ["frame", "count"]

plt.figure(figsize=(10, 6))
sns.barplot(
    data=frame_counts,
    y="frame",
    x="count",
    hue="frame",
    palette=FRAME_PALETTE,
    legend=False
)
plt.title("Overall Distribution of LLM Policy Frames")
plt.xlabel("Number of Universities")
plt.ylabel("Frame")
plt.tight_layout()
plt.savefig(CONFIG["data"]["overall_frame_distribution"], dpi=300)
plt.close()
print(f"Saved: {CONFIG['data']['overall_frame_distribution']}")

# =================================
# A. Dominant Frame per Country
# =================================
dominant = (
    df_exp.groupby(["country", "frame"])
          .size()
          .reset_index(name="count")
)

dominant = dominant.loc[
    dominant.groupby("country")["count"].idxmax()
]

plt.figure(figsize=(10, 6))
sns.barplot(
    data=dominant,
    x="country",
    y="count",
    hue="frame",
    palette=FRAME_PALETTE
)
plt.title("Dominant LLM Policy Frame by Country")
plt.ylabel("Number of Universities")
plt.xlabel("Country")
plt.tight_layout()
plt.savefig(CONFIG["data"]["dominant_frame_by_country"], dpi=300)
plt.close()
print(f"Saved: {CONFIG['data']['dominant_frame_by_country']}")