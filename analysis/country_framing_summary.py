import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

#palette for heatmap (blue gradient)
blue_cmap = LinearSegmentedColormap.from_list(
    "custom_blues",
    ["#A8DADC", "#2E86AB", "#1E3A5F"]  # light → mid → dark blue
)

# -----------------------------
# Paths
# -----------------------------
AFFECT = "analysis_results/rq1/affective_language_university.csv"
ROLES  = "analysis_results/rq1/institutional_roles_university.csv"
FRAMES = "analysis_results/rq1/policy_frames.csv"

OUT_DATA = "analysis_results/rq1/rq1_country_summary.csv"
OUT_FIG  = "analysis_results/rq1/figures"

os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# -----------------------------
# Load files
# -----------------------------
aff = pd.read_csv(AFFECT)
rol = pd.read_csv(ROLES)
frm = pd.read_csv(FRAMES)

# -----------------------------
# Merge at document level
# -----------------------------
df = (
    aff[["url", "country", "dominant_affect"]]
    .merge(
        rol[["url", "role_assumption"]],
        on="url",
        how="left"
    )
    .merge(
        frm[["url", "policy_frame"]],
        on="url",
        how="left"
    )
)

# -----------------------------
# Country-level dominant category
# -----------------------------
summary = (
    df.groupby("country")
      .agg({
          "dominant_affect": lambda x: x.value_counts().idxmax(),
          "role_assumption": lambda x: x.value_counts().idxmax(),
          "policy_frame": lambda x: x.value_counts().idxmax()
      })
      .reset_index()
)

summary.to_csv(OUT_DATA, index=False)
print(f"✓ Saved RQ1 country summary → {OUT_DATA}")

# -----------------------------
# Encode categories for heatmap
# -----------------------------
heatmap_data = summary.set_index("country").copy()

for col in ["dominant_affect", "role_assumption", "policy_frame"]:
    heatmap_data[col + "_code"] = (
        heatmap_data[col]
        .astype("category")
        .cat.codes
    )

# -----------------------------
# OLD-STYLE HEATMAP (TABLE-LIKE)
# -----------------------------
plt.figure(figsize=(10, 4))
sns.heatmap(
    heatmap_data[
        ["dominant_affect_code", "role_assumption_code", "policy_frame_code"]
    ],
    annot=heatmap_data[
        ["dominant_affect", "role_assumption", "policy_frame"]
    ],
    fmt="",
    cmap= blue_cmap,
    cbar=False,
    linewidths=0.5
)

plt.title("Country-Level Framing Summary")
plt.ylabel("Country")
plt.xlabel("Framing Dimension")
plt.tight_layout()
plt.savefig(f"{OUT_FIG}/rq1_country_summary_heatmap.png", dpi=300)
plt.show()
