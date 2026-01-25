import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG
from config.logger import setup_logger

import pandas as pd
import gensim
import pyLDAvis
import pyLDAvis.gensim
import re
from gensim import corpora
from gensim.models import CoherenceModel
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

logger = setup_logger(__name__)

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("LDA TOPIC MODELING - 6 TOPICS FIXED")
    logger.info("=" * 80)

    # 1. LOAD DATA
    df = pd.read_csv(CONFIG["data"]["clean"])
    needed_cols = [
        "url", "country", "document_type", "clean_text",
        "tokens", "word_count", "sentence_count", "ai_term_count"
    ]
    df = df[needed_cols]
    df["tokens"] = df["tokens"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    logger.info("Data loaded successfully")

    # 2. DICTIONARY & CORPUS
    dictionary = corpora.Dictionary(df["tokens"])
    valid_rows = []
    corpus = []
    for idx, tokens in enumerate(df["tokens"]):
        bow = dictionary.doc2bow(tokens)
        if len(bow) > 0:
            corpus.append(bow)
            valid_rows.append(idx)
    df = df.iloc[valid_rows].reset_index(drop=True)
    logger.info("Dictionary and corpus created")

    # 3. FORCE EXACTLY 6 TOPICS (NO OPTIMIZATION)
    best_k = 6
    logger.info(f"Using fixed number of topics: {best_k}")

    # 4. TRAIN FINAL MODEL
    logger.info("Training final LDA model...")
    lda_model = gensim.models.LdaModel(
        corpus=corpus, id2word=dictionary, num_topics=best_k,
        passes=20, random_state=42
    )
    logger.info("LDA model trained")

    # 5. DOMINANT TOPIC PER DOCUMENT
    dominant_topics = []
    dominant_probs = []
    second_topics = []
    second_probs = []
    for bow in corpus:
        topics = lda_model.get_document_topics(bow)
        topics_sorted = sorted(topics, key=lambda x: x[1], reverse=True)
        dominant_topics.append(topics_sorted[0][0])
        dominant_probs.append(topics_sorted[0][1])
        if len(topics_sorted) > 1:
            second_topics.append(topics_sorted[1][0])
            second_probs.append(topics_sorted[1][1])
        else:
            second_topics.append(None)
            second_probs.append(None)
    assert len(df) == len(dominant_topics), "ERROR: df and topic lists are misaligned"
    df["lda_topic"] = dominant_topics
    df["lda_topic_probability"] = dominant_probs
    df["lda_second_topic"] = second_topics
    df["lda_second_probability"] = second_probs
    logger.info("Topics assigned to documents")

    # 6. PRINT TOP WORDS PER TOPIC
    topics = lda_model.print_topics(num_words=15)
    with open(CONFIG["data"]["lda_topics"], "w") as f:
        for topic_num, words in topics:
            f.write(f"Topic {topic_num}: {words}\n")
    logger.info(f"Saved: {CONFIG['data']['lda_topics']}")

    # 7. PYLDAVIS VISUALIZATION
    try:
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, CONFIG["data"]["lda_viz"])
        logger.info(f"Saved: {CONFIG['data']['lda_viz']}")
    except ImportError:
        logger.warning("pyLDAvis not installed")

    # 8. HUMAN-READABLE TOPIC LABELS (ONLY 6)
    topic_labels = {
        0: "Miscellaneous / Low-Frequency Content",
        1: "General Guidance on Generative AI Tools",
        2: "Responsible Use, Risks, and Policy Frameworks",
        3: "Teaching, Learning, and Assessment Guidance",
        4: "Student Use of Generative AI in Coursework",
        5: "Academic Integrity, Misconduct, and Plagiarism"
    }
    df["lda_topic"] = df["lda_topic"].astype(int)
    df["topic_label"] = df["lda_topic"].map(topic_labels)
    logger.info("Topic labels assigned")

    # 9. TOPIC SUMMARY TABLE
    topic_summary = {
        "topic_num": [], "topic_label": [], "top_words": [],
        "num_documents": [], "avg_probability": []
    }
    for topic_num, words in lda_model.print_topics(num_words=12):
        topic_docs = df[df["lda_topic"] == topic_num]
        topic_summary["topic_num"].append(topic_num)
        topic_summary["topic_label"].append(topic_labels.get(topic_num, "Unknown"))
        topic_summary["top_words"].append(words)
        topic_summary["num_documents"].append(len(topic_docs))
        topic_summary["avg_probability"].append(topic_docs["lda_topic_probability"].mean())

    summary_df = pd.DataFrame(topic_summary)
    summary_df.to_csv(CONFIG["data"]["lda_summary"], index=False)
    logger.info(f"Saved: {CONFIG['data']['lda_summary']}")

    # 10. FINAL DATASET
    df.to_csv(CONFIG["data"]["lda_results"], index=False)
    logger.info(f"Saved: {CONFIG['data']['lda_results']}")

    # =============================================================================
    # 11. EXTENDED ANALYSIS: UNIVERSITY → POLICIES → TOPICS
    # =============================================================================
    logger.info("=" * 80)
    logger.info("EXTENDED ANALYSIS: UNIVERSITY -> POLICIES -> TOPICS")
    logger.info("=" * 80)

    def extract_university_domain(url: str):
        m = re.search(r"https?://(?:www\.)?([^/]+)", str(url))
        return m.group(1) if m else None

    df["university_domain"] = df["url"].apply(extract_university_domain)
    df["university_name"] = df["university_domain"].apply(
        lambda x: x.split(".")[0].upper() if isinstance(x, str) else None
    )

    university_stats = []
    for uni in sorted(df["university_name"].dropna().unique()):
        uni_data = df[df["university_name"] == uni]
        num_policies = len(uni_data)
        unique_topics = sorted(set(uni_data["lda_topic"].tolist()))
        num_unique_topics = len(unique_topics)
        topic_distribution = uni_data["lda_topic"].value_counts().sort_index().to_dict()
        university_stats.append({
            "university": uni,
            "num_policies": num_policies,
            "topics_assigned": unique_topics,
            "num_unique_topics": num_unique_topics,
            "topic_distribution": topic_distribution,
        })

    logger.info("=" * 80)
    logger.info("SUMMARY: Universities with Multiple Topics")
    logger.info("=" * 80)
    logger.info(f"{'University':<15} {'Policies':<12} {'#Topics':<10} {'Topics':<30}")
    logger.info("-" * 70)
    for stat in sorted(university_stats, key=lambda x: x["num_policies"], reverse=True):
        uni = stat["university"]
        policies = stat["num_policies"]
        n_topics = stat["num_unique_topics"]
        topic_ids = str(stat["topics_assigned"])
        logger.info(f"{uni:<15} {policies:<12} {n_topics:<10} {topic_ids:<30}")

    uni_topic_matrix = pd.crosstab(df["university_name"], df["lda_topic"])
    logger.info("=" * 80)
    logger.info("MATRIX: Which Topics Does Each University Cover?")
    logger.info("=" * 80)
    logger.info(str(uni_topic_matrix))

    logger.info("=" * 80)
    logger.info("SAVING EXTENDED ANALYSIS RESULTS")
    logger.info("=" * 80)

    uni_policy_topic_df = df[[
        "university_domain", "university_name", "url", "lda_topic",
        "topic_label", "lda_topic_probability"
    ]].copy()
    uni_policy_topic_df.to_csv("data/university_policy_topic_mapping.csv", index=False)
    logger.info("Saved: data/university_policy_topic_mapping.csv")

    uni_stats_df = pd.DataFrame([{
        "university": stat["university"],
        "num_policies": stat["num_policies"],
        "num_unique_topics": stat["num_unique_topics"],
        "topics": ",".join(map(str, stat["topics_assigned"])),
        "topic_distribution": str(stat["topic_distribution"]),
    } for stat in university_stats])
    uni_stats_df.to_csv("data/university_topic_statistics.csv", index=False)
    logger.info("Saved: data/university_topic_statistics.csv")

    uni_topic_matrix.to_csv("data/university_topic_matrix.csv")
    logger.info("Saved: data/university_topic_matrix.csv")

    logger.info("=" * 80)
    logger.info("STATISTICAL SUMMARY")
    logger.info("=" * 80)

    multi_topic_unis = [s for s in university_stats if s["num_unique_topics"] > 1]
    multi_policy_unis = [s for s in university_stats if s["num_policies"] > 1]
    n_unis = len(university_stats)
    n_docs = len(df)

    logger.info(f"Total Universities: {n_unis}")
    logger.info(f"Total Policies: {n_docs}")
    logger.info(f"Fixed Topics: {best_k}")
    logger.info(f"Average Policies per University: {n_docs / n_unis:.1f}")
    logger.info(f"Average Topics per University: {df.groupby('university_name')['lda_topic'].nunique().mean():.1f}")
    logger.info(f"Universities with >1 topic: {len(multi_topic_unis)} ({len(multi_topic_unis)/n_unis*100:.1f}%)")
    logger.info(f"Universities with >1 policy: {len(multi_policy_unis)} ({len(multi_policy_unis)/n_unis*100:.1f}%)")

    logger.info("=" * 80)
    logger.info("COMPLETE - 6 TOPIC LDA ANALYSIS FINISHED")
    logger.info("=" * 80)
    logger.info("ALL FILES CREATED:")
    logger.info(f" - {CONFIG['data']['lda_topics']}")
    logger.info(f" - {CONFIG['data']['lda_viz']}")
    logger.info(f" - {CONFIG['data']['lda_summary']}")
    logger.info(f" - {CONFIG['data']['lda_results']}")
    logger.info(" - data/university_policy_topic_mapping.csv")
    logger.info(" - data/university_topic_statistics.csv")
    logger.info(" - data/university_topic_matrix.csv")
