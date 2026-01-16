import pandas as pd
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
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
# VISUALIZATION: GROUPED BAR CHART
# -----------------------------
plt.figure(figsize=(10, 6))

sns.countplot(
    data=uni_roles,
    x="country",
    hue="role_assumption",
    order=sorted(uni_roles["country"].unique()),
    hue_order=[
        "Institution-Led",
        "Instructor-Led",
        "Student-Responsible",
        "Unspecified"
    ]
)

plt.title("Institutional Role Assumptions in LLM Policies (University Level)")
plt.xlabel("Country")
plt.ylabel("Number of Universities")
plt.xticks(rotation=0)
plt.legend(title="Responsible Actor")

plt.tight_layout()
plt.savefig(f"{OUT_FIG}/institutional_roles_by_country_grouped.png", dpi=300)
plt.show()