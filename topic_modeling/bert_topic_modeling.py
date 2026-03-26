import random
random.seed(42)
import numpy as np
np.random.seed(42)
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
import os
import numpy as np
from config import CONFIG


if __name__ == "__main__":
    # Paths
    INPUT_FILE = CONFIG["data"]["clean"]
    OUTPUT_TOPICS_FILE = CONFIG["data"]["bert_results"]
    OUTPUT_SUMMARY_FILE = CONFIG["data"]["bert_summary"]
    MODEL_SAVE_DIR = CONFIG["paths"]["models"]

    print("=" * 80)
    print("BERT TOPIC MODELING - 7 TOPICS")
    print("=" * 80)

    # Load data
    df = pd.read_csv(INPUT_FILE)
    texts = df["clean_text"].astype(str).tolist()
    print(f"Loaded {len(texts)} documents")

    # Embedding Model
    print("Initializing embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # CUSTOM UMAP + HDBSCAN → prevents collapsing topics
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )

    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )

    # CREATE BERTopic WITH FORCED 7 TOPICS
    print("Creating BERTopic model with 7 topics...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=CONFIG["bert"]["num_topics"],  # From config
        min_topic_size=CONFIG["bert"]["min_topic_size"],  # From config
        verbose=True
    )


    # Fit model
    print("Fitting BERTopic model...")
    topics, probabilities = topic_model.fit_transform(texts)
    df["topic"] = topics
    if probabilities is None:
        df["topic_probability"] = 0.0
    else:
        probabilities = np.array(probabilities)
        # Case 1: probabilities is 1D → already max probability
        if probabilities.ndim == 1:
            df["topic_probability"] = probabilities
        # Case 2: probabilities is 2D → take max across topics
        else:
            df["topic_probability"] = probabilities.max(axis=1)

    df.to_csv(OUTPUT_TOPICS_FILE, index=False)
    print(f"Saved: {OUTPUT_TOPICS_FILE}")

    # Save summary
    topic_summary = topic_model.get_topic_info()
    topic_summary.to_csv(OUTPUT_SUMMARY_FILE, index=False)
    print(f"Saved summary: {OUTPUT_SUMMARY_FILE}")

    # Save model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    topic_model.save(f"{MODEL_SAVE_DIR}/bertopic_model.pkl")
    print("Model saved")

    print("=" * 80)
    print("BERT TOPIC MODELING COMPLETED")
    print("=" * 80)
