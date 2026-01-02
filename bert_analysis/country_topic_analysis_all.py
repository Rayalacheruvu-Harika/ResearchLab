# -------------------------------------------------
# Country-Level Multi-Topic Analysis (ALL CHARTS)
# -------------------------------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
import os

# -------------------------------------------------
# Paths
# -------------------------------------------------
INPUT_FILE = "analysis_results/university_top_topics.csv"
OUTPUT_DIR = "analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Load Data
# -------------------------------------------------
df = pd.read_csv(INPUT_FILE)

# =================================================
# A. Country → Topic Profile Table (SAVE)
# =================================================
df.to_csv(f"{OUTPUT_DIR}/country_topic_profile_table.csv", index=False)

# =================================================
# B. Heatmap: Topic Importance (Top-3 Weighted)
# =================================================

# Convert to long format
long_df = pd.melt(
    df,
    id_vars=["country"],
    value_vars=["primary_topic", "secondary_topic", "tertiary_topic"],
    var_name="rank",
    value_name="topic"
)

# Assign weights
rank_weights = {
    "primary_topic": 3,
    "secondary_topic": 2,
    "tertiary_topic": 1
}
long_df["weight"] = long_df["rank"].map(rank_weights)

# Pivot table
heatmap_df = long_df.pivot_table(
    index="country",
    columns="topic",
    values="weight",
    aggfunc="sum",
    fill_value=0
)

heatmap_df.to_csv(f"{OUTPUT_DIR}/country_topic_heatmap_data.csv")

plt.figure(figsize=(14, 7))
sns.heatmap(
    heatmap_df,
    annot=True,
    cmap="Blues",
    linewidths=0.5
)
plt.title("Country × Topic Importance (Top-3 Topics Weighted)")
plt.xlabel("Policy Topic")
plt.ylabel("Country")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/heatmap_topic_importance.png", dpi=300)
plt.show()

# =================================================
# C. Topic Co-Occurrence Network
# =================================================

edges = []

for _, row in df.iterrows():
    topics = [
        row["primary_topic"],
        row["secondary_topic"],
        row["tertiary_topic"]
    ]
    for pair in combinations(topics, 2):
        edges.append(pair)

edge_df = pd.DataFrame(edges, columns=["source", "target"])
edge_counts = edge_df.value_counts().reset_index(name="weight")

G = nx.Graph()
for _, row in edge_counts.iterrows():
    G.add_edge(row["source"], row["target"], weight=row["weight"])

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue")
nx.draw_networkx_edges(
    G,
    pos,
    width=[d["weight"] for (_, _, d) in G.edges(data=True)]
)
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Topic Co-Occurrence Network Across Countries")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/topic_cooccurrence_network.png", dpi=300)
plt.show()

# =================================================
# D. Bar Chart: Dominant PRIMARY Topics
# =================================================

primary_counts = df["primary_topic"].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(
    x=primary_counts.values,
    y=primary_counts.index,
    palette="viridis"
)

plt.title("Dominant Primary AI Policy Topics by Country")
plt.xlabel("Number of Countries")
plt.ylabel("Primary Topic")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/primary_topic_bar_chart.png", dpi=300)
plt.show()

print("\n✅ ALL ANALYSES COMPLETED SUCCESSFULLY")
print("📁 Outputs saved in:", OUTPUT_DIR)
