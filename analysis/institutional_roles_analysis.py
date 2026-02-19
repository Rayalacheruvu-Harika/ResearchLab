import pandas as pd
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# -----------------------------
# Load spaCy
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Paths
# -----------------------------
INPUT = "data/final_clean_dataset.csv"
OUT_DATA = "analysis_results/rq1/institutional_roles_university.csv"
OUT_FIG = "analysis_results/rq1/figures"

os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# -----------------------------
# Role Lexicons
# -----------------------------
INSTITUTION = {"university", "institution", "college"}
INSTRUCTOR  = {"faculty", "instructor", "lecturer", "professor"}
STUDENT     = {"student", "students"}

def classify_role(text):
    doc = nlp(str(text))
    subjects = {tok.text.lower() for tok in doc if tok.dep_ == "nsubj"}

    if subjects & INSTITUTION:
        return "Institution-Led"
    if subjects & INSTRUCTOR:
        return "Instructor-Led"
    if subjects & STUDENT:
        return "Student-Responsible"
    return "Unspecified"

# -----------------------------
# Load & classify (document-level)
# -----------------------------
df = pd.read_csv(INPUT)
df["role_assumption"] = df["guideline_text"].apply(classify_role)

# -----------------------------
# 🔑 UNIVERSITY-LEVEL AGGREGATION
# -----------------------------
uni_roles = (
    df.groupby(["url", "country", "role_assumption"])
      .size()
      .reset_index(name="count")
)

# Dominant role per university
uni_roles = uni_roles.loc[
    uni_roles.groupby("url")["count"].idxmax()
]

# Save output
uni_roles.to_csv(OUT_DATA, index=False)
print(f"✓ Saved university-level institutional roles → {OUT_DATA}")
# -----------------------------
# VISUALIZATION: COUNTRY SMALL MULTIPLES (MANUAL, STABLE)
# -----------------------------

plot_df = (
    uni_roles
    .groupby(["country", "role_assumption"])
    .size()
    .reset_index(name="count")
)

roles = [
    "Institution-Led",
    "Instructor-Led",
    "Student-Responsible",
    "Unspecified"
]

countries = sorted(plot_df["country"].unique())

# Role-based colors (used in legend)
role_palette = {
    "Institution-Led": "#1f77b4",
    "Instructor-Led": "#ff7f0e",
    "Student-Responsible": "#2ca02c",
    "Unspecified": "#d62728"
}

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()

for i, country in enumerate(countries):
    ax = axes[i]
    data = plot_df[plot_df["country"] == country]

    values = [
        data.loc[data["role_assumption"] == r, "count"].sum()
        for r in roles
    ]

    # Bars colored by role (legend explains roles)
    ax.bar(
        roles,
        values,
        color=[role_palette[r] for r in roles],
        edgecolor="black"
    )

    ax.set_title(country)
    ax.set_ylim(0, 6)

    # remove all x ticks
    ax.set_xticks([])
    ax.set_xlabel("")

# remove empty subplot if exists
for j in range(len(countries), len(axes)):
    fig.delaxes(axes[j])

# Global labels
fig.suptitle(
    "Responsible Actor for LLM Policy Governance (University Level)",
    fontsize=14
)

fig.text(0.04, 0.5, "Number of Universities", va="center", rotation="vertical")

# Legend instead of ticks
legend_handles = [
    Patch(facecolor=color, edgecolor="black", label=role)
    for role, color in role_palette.items()
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 0.02)
)

# Layout (space for title + legend)
plt.tight_layout(rect=[0.05, 0.08, 1, 0.93])

plt.savefig(
    f"{OUT_FIG}/institutional_roles_small_multiples_legend.png",
    dpi=300
)
plt.show()