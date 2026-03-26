import pandas as pd
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
from config import CONFIG
# -----------------------------
# Load spaCy
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# Paths
# -----------------------------
INPUT = CONFIG["data"]["clean"]
OUT_DATA = CONFIG["data"]["institutional_roles_university"]
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
STUDENT     = {"student", "students"}

role_palette = {
    "Institution-Led": "#1E3A5F",
    "Instructor-Led": "#2E86AB",
    "Student-Responsible": "#44BBA4",
    "Unspecified": "#48CAE4"
}

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
uni_roles.to_csv(CONFIG["data"]["institutional_roles_university"], index=False)
print(f"Saved university-level institutional roles  {OUT_DATA}")

# -----------------------------
# VISUALIZATION: GROUPED BAR CHART
# -----------------------------
plt.figure(figsize=(10, 6))

sns.countplot(
    data=uni_roles,
    x="country",
    hue="role_assumption",
    order=sorted(uni_roles["country"].unique()),
    hue_order=list(role_palette.keys()),
    palette=role_palette
)

plt.title("Institutional Role Assumptions in LLM Policies (University Level)")
plt.xlabel("Country")
plt.ylabel("Number of Universities")
plt.xticks(rotation=0)
plt.legend(title="Responsible Actor")

plt.tight_layout()
plt.savefig(CONFIG["data"]["institutional_roles_by_country"], dpi=300)
plt.close()
# -----------------------------
# VISUALIZATION 2: Country Small Multiples
# -----------------------------
plot_df = (
    uni_roles
    .groupby(["country", "role_assumption"])
    .size()
    .reset_index(name="count")
)

roles = list(role_palette.keys())
countries = sorted(plot_df["country"].unique())

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
axes = axes.flatten()

for i, country in enumerate(countries):
    ax = axes[i]
    data = plot_df[plot_df["country"] == country]
    values = [data.loc[data["role_assumption"] == r, "count"].sum() for r in roles]
    ax.bar(roles, values, color=[role_palette[r] for r in roles], edgecolor="black")
    ax.set_title(country)
    ax.set_ylim(0, 6)
    ax.set_xticks([])

for j in range(len(countries), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Responsible Actor for LLM Policy Governance (University Level)", fontsize=14)
fig.text(0.04, 0.5, "Number of Universities", va="center", rotation="vertical")

legend_handles = [Patch(facecolor=c, edgecolor="black", label=r) for r, c in role_palette.items()]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(rect=[0.05, 0.08, 1, 0.93])
plt.savefig(CONFIG["data"]["institutional_roles_small_multiples"], dpi=300)
plt.close()
print(f"Saved: {CONFIG['data']['institutional_roles_university']}")
print(f"Saved: {CONFIG['data']['institutional_roles_by_country']}")
print(f"Saved: {CONFIG['data']['institutional_roles_small_multiples']}")