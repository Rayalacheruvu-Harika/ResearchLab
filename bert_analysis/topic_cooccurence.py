# -------------------------------------------------
# Topic Co-Occurrence Analysis (University Level)
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
TOPIC_RESULTS = "data/bert_topic_model_results.csv"
TOPIC_LABELS = "data/bert_topic_labels.csv"
OUTPUT_DIR = "analysis_results/topic_cooccurence"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Load data
# -------------------------------------------------
df_topics = pd.read_csv(TOPIC_RESULTS)
df_labels = pd.read_csv(TOPIC_LABELS)

# -------------------------------------------------
# Validate
# -------------------------------------------------
required_topics = {"country", "topic"}
required_labels = {"topic_id", "topic_name"}

if not required_topics.issubset(df_topics.columns):
    raise ValueError("Topic results must contain: country, topic")

if not required_labels.issubset(df_labels.columns):
    raise ValueError("Labels file must contain: topic_id, topic_name")

# -------------------------------------------------
# Merge topic names
# -------------------------------------------------
df = df_topics.merge(
    df_labels,
    left_on="topic",
    right_on="topic_id",
    how="left"
)

df["topic_name"] = df["topic_name"].fillna("Unlabeled")

# -------------------------------------------------
# Build university → topic sets
# -------------------------------------------------
uni_topics = (
    df.groupby("country")["topic_name"]
      .apply(lambda x: set(x))
      .reset_index()
)

# -------------------------------------------------
# Generate co-occurrence pairs
# -------------------------------------------------
pairs = []

for _, row in uni_topics.iterrows():
    topics = list(row["topic_name"])
    for pair in combinations(topics, 2):
        pairs.append(tuple(sorted(pair)))

co_df = pd.DataFrame(pairs, columns=["topic_1", "topic_2"])

# -------------------------------------------------
# Count co-occurrences
# -------------------------------------------------
co_counts = (
    co_df.value_counts()
         .reset_index(name="count")
)

os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)

co_counts.to_csv(f"{OUTPUT_DIR}/topic_cooccurrence_pairs.csv", index=False)

# -------------------------------------------------
# Create co-occurrence matrix
# -------------------------------------------------
matrix = co_counts.pivot_table(
    index="topic_1",
    columns="topic_2",
    values="count",
    fill_value=0
)

# Make symmetric
matrix = matrix.add(matrix.T, fill_value=0)

matrix.to_csv(f"{OUTPUT_DIR}/topic_cooccurrence_matrix.csv")

# -------------------------------------------------
# Heatmap visualization
# -------------------------------------------------
plt.figure(figsize=(14, 10))
sns.heatmap(
    matrix,
    cmap="Blues",
    linewidths=0.5,
    annot=True
)
plt.title("Topic Co-Occurrence Heatmap (University Level)")
plt.xlabel("Policy Topic")
plt.ylabel("Policy Topic")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/topic_cooccurrence_heatmap.png", dpi=300)
plt.show()

# -------------------------------------------------
# Network graph visualization
# -------------------------------------------------
G = nx.Graph()

for _, row in co_counts.iterrows():
    G.add_edge(
        row["topic_1"],
        row["topic_2"],
        weight=row["count"]
    )

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue")
nx.draw_networkx_edges(
    G,
    pos,
    width=[d["weight"] for (_, _, d) in G.edges(data=True)]
)
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Topic Co-Occurrence Network Across Universities")
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/topic_cooccurrence_network.png", dpi=300)
plt.show()

print("\n✅ Topic co-occurrence analysis completed")
print("📁 Outputs saved in:", OUTPUT_DIR)
