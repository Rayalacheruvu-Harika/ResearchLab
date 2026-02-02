import pandas as pd
import matplotlib.pyplot as plt
import os

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LABEL_FILE = os.path.join(BASE_DIR, "sentiment_manual.xlsx")
META_FILE = os.path.join(BASE_DIR, "../data/final_clean_dataset.csv")

OUT_DIR = os.path.join(BASE_DIR, "../analysis_results")
FIG_DIR = os.path.join(OUT_DIR, "Manual_sentiment")

os.makedirs(FIG_DIR, exist_ok=True)

# ==============================
# LOAD DATA
# ==============================
labels_df = pd.read_excel(LABEL_FILE)

if not {"url", "label"}.issubset(labels_df.columns):
    raise ValueError("Excel must contain 'url' and 'label' columns")

meta_df = pd.read_csv(META_FILE)[["url", "country"]]

# Merge country info
df = labels_df.merge(meta_df, on="url", how="left")

print("✓ Loaded manual labels with country metadata")

# ==============================
# SPLIT MULTI-LABEL CELLS
# ==============================
df["label_list"] = df["label"].str.split(",")
df = df.explode("label_list")
df["label_clean"] = df["label_list"].str.strip()

# ==============================
# OVERALL LABEL FREQUENCY (TOP 10)
# ==============================
label_counts = df["label_clean"].value_counts()
top10_labels = label_counts.head(10)

top10_labels.to_csv(
    os.path.join(OUT_DIR, "top10_manual_labels.csv"),
    header=["count"]
)

# ==============================
# BAR CHART: OVERALL
# ==============================
top10_labels.plot(
    kind="bar",
    figsize=(10, 5),
    color="tab:blue"
)

plt.title("Top 10 Manual Policy Tones (All Universities)")
plt.xlabel("Manual Label")
plt.ylabel("Count of Occurrences")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

FIG_PATH = os.path.join(FIG_DIR, "top10_manual_labels.png")
plt.savefig(FIG_PATH, dpi=300)
plt.show()

print("✓ Overall label frequency plot saved")

# ==============================
# COUNTRY × LABEL DISTRIBUTION
# (Country-wise Top Labels)
# ==============================

# Full country × label table
country_label_counts = pd.pivot_table(
    df,
    index="country",
    columns="label_clean",
    aggfunc="size",
    fill_value=0
)

# ------------------------------
# Get top N labels PER country
# ------------------------------
TOP_N = 10

top_labels_per_country = set()

for country in country_label_counts.index:
    top_labels = (
        country_label_counts.loc[country]
        .sort_values(ascending=False)
        .head(TOP_N)
        .index
    )
    top_labels_per_country.update(top_labels)

# Convert set to sorted list (for consistent plotting)
top_labels_per_country = sorted(top_labels_per_country)

# Filter table to country-wise top labels
country_label_counts_top = country_label_counts[top_labels_per_country]

# ==============================
# STACKED BAR: TOP 10 PER COUNTRY ONLY
# ==============================

TOP_N = 7

# Build a filtered table where each country keeps only its own top 10 labels
filtered_rows = []

for country in country_label_counts.index:
    top_labels = (
        country_label_counts.loc[country]
        .sort_values(ascending=False)
        .head(TOP_N)
    )
    df_top = top_labels.reset_index()
    df_top.columns = ["label_clean", "count"]
    df_top["country"] = country
    filtered_rows.append(df_top)

plot_df = pd.concat(filtered_rows)

# Pivot back for stacked bar plot
plot_table = plot_df.pivot(
    index="country",
    columns="label_clean",
    values="count"
).fillna(0)

plot_table.plot(
    kind="bar",
    stacked=True,
    figsize=(13, 6)
)

plt.title("Top 7 Manual Policy Tones per Country")
plt.xlabel("Country")
plt.ylabel("Label Count")
plt.xticks(rotation=0)
plt.legend(
    title="Manual Policy Tone",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
plt.tight_layout()

FIG_PATH = os.path.join(FIG_DIR, "manual_labels_by_country_top10_each.png")
plt.savefig(FIG_PATH, dpi=300)
plt.show()

print("✓ Country-wise top 10 stacked chart saved")