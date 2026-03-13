import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

FRAME_PALETTE = {
    "Regulatory / Governance":  "#1E3A5F",  # deep navy
    "Pedagogical Support":      "#2E86AB",  # steel blue
    "Technological Enablement": "#44BBA4",  # teal
     "Risk & Compliance":        "#A8DADC",  # light blue
     "Control & Integrity":      "#48CAE4",  # sky blue
}

# ---------------------------------
# Paths
# ---------------------------------
INPUT_FILE = "analysis_results/framing/university_framing_profiles.csv"
OUTPUT_DIR = "analysis_results/framing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------
# Load data
# ---------------------------------
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
    palette=FRAME_PALETTE
)
plt.title("Overall Distribution of LLM Policy Frames")
plt.xlabel("Number of Universities")
plt.ylabel("Frame")
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/overall_frame_distribution.png",
    dpi=300
)
plt.show()

print("✓ Saved overall frame distribution plot")

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
plt.savefig(
    f"{OUTPUT_DIR}/dominant_frame_by_country.png",
    dpi=300
)
plt.show()

print("✓ Saved dominant frame plot")

