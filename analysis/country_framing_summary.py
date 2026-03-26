import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from config import CONFIG
from matplotlib.colors import LinearSegmentedColormap

blue_cmap = LinearSegmentedColormap.from_list(
    "custom_blues",
    ["#A8DADC", "#2E86AB", "#1E3A5F"]
)

AFFECT = CONFIG["data"]["affective_language_university"]
ROLES  = CONFIG["data"]["institutional_roles_university"]
FRAMES = CONFIG["data"]["policy_frames"]

OUT_DATA = CONFIG["data"]["rq1_country_summary"]

os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
os.makedirs(os.path.dirname(CONFIG["data"]["rq1_country_summary_heatmap"]), exist_ok=True)

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

summary.to_csv(CONFIG["data"]["rq1_country_summary"], index=False)
print(f"Saved RQ1 country summary {OUT_DATA}")

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
    cmap=blue_cmap,
    cbar=False,
    linewidths=0.5
)

plt.title("Country-Level Framing Summary")
plt.ylabel("Country")
plt.xlabel("Framing Dimension")
plt.tight_layout()
plt.savefig(CONFIG["data"]["rq1_country_summary_heatmap"], dpi=300)
plt.close()