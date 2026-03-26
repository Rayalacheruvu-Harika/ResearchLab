from config import CONFIG

import pandas as pd
import gensim
import pyLDAvis
import pyLDAvis.gensim
import re
from gensim import corpora
from gensim.models import CoherenceModel
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__ == "__main__":
    print("=" * 80)
    print("LDA TOPIC MODELING - 6 TOPICS FIXED")
    print("=" * 80)

    # 1. LOAD DATA
    df = pd.read_csv(CONFIG["data"]["clean"])
    needed_cols = [
        "url", "country", "document_type", "clean_text",
        "tokens", "word_count", "sentence_count", "ai_term_count"
    ]
    df = df[needed_cols]
    df["tokens"] = df["tokens"].apply(lambda x: eval(x) if isinstance(x, str) else x)
    print("Data loaded successfully")
    # Load trained LDA model
    lda_model = gensim.models.LdaModel.load(str(CONFIG["data"]["lda_model"]))
    print(f"Loaded LDA model from {CONFIG['data']['lda_model']}") 

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
    print("Dictionary and corpus created")

    # 3. FORCE EXACTLY 6 TOPICS (NO OPTIMIZATION)
    best_k = CONFIG["lda"]["num_topics"]
    print(f"Using fixed number of topics: {best_k}")

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
    print("Topics assigned to documents")

    # 6. PRINT TOP WORDS PER TOPIC
    topics = lda_model.print_topics(num_words=15)
    with open(CONFIG["data"]["lda_topics"], "w") as f:
        for topic_num, words in topics:
            f.write(f"Topic {topic_num}: {words}\n")
    print(f"Saved: {CONFIG['data']['lda_topics']}")

    # 7. PYLDAVIS VISUALIZATION
    try:
        vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
        pyLDAvis.save_html(vis, CONFIG["data"]["lda_viz"])
        print(f"Saved: {CONFIG['data']['lda_viz']}")
    except ImportError:
        print("pyLDAvis not installed")

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
    print("Topic labels assigned")

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
    print(f"Saved: {CONFIG['data']['lda_summary']}")

    # 10. FINAL DATASET
    df.to_csv(CONFIG["data"]["lda_results"], index=False)
    print(f"Saved: {CONFIG['data']['lda_results']}")

    # =============================================================================
    # 11. EXTENDED ANALYSIS: UNIVERSITY → POLICIES → TOPICS
    # =============================================================================
    print("=" * 80)
    print("EXTENDED ANALYSIS: UNIVERSITY -> POLICIES -> TOPICS")
    print("=" * 80)

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

    print("=" * 80)
    print("SUMMARY: Universities with Multiple Topics")
    print("=" * 80)
    print(f"{'University':<15} {'Policies':<12} {'#Topics':<10} {'Topics':<30}")
    print("-" * 70)
    for stat in sorted(university_stats, key=lambda x: x["num_policies"], reverse=True):
        uni = stat["university"]
        policies = stat["num_policies"]
        n_topics = stat["num_unique_topics"]
        topic_ids = str(stat["topics_assigned"])
        print(f"{uni:<15} {policies:<12} {n_topics:<10} {topic_ids:<30}")

    uni_topic_matrix = pd.crosstab(df["university_name"], df["lda_topic"])
    print("=" * 80)
    print("MATRIX: Which Topics Does Each University Cover?")
    print("=" * 80)
    print(str(uni_topic_matrix))

    print("=" * 80)
    print("SAVING EXTENDED ANALYSIS RESULTS")
    print("=" * 80)

    uni_policy_topic_df = df[[
        "university_domain", "university_name", "url", "lda_topic",
        "topic_label", "lda_topic_probability"
    ]].copy()
    uni_policy_topic_df.to_csv(CONFIG["data"]["uni_policy_topic_mapping"], index=False)
    print(f"Saved: {CONFIG['data']['uni_policy_topic_mapping']}")

    uni_stats_df = pd.DataFrame([{
        "university": stat["university"],
        "num_policies": stat["num_policies"],
        "num_unique_topics": stat["num_unique_topics"],
        "topics": ",".join(map(str, stat["topics_assigned"])),
        "topic_distribution": str(stat["topic_distribution"]),
    } for stat in university_stats])
    uni_stats_df.to_csv(CONFIG["data"]["uni_topic_statistics"], index=False)
    print(f"Saved: {CONFIG['data']['uni_topic_statistics']}")

    uni_topic_matrix.to_csv(CONFIG["data"]["uni_topic_matrix"])
    print(f"Saved: {CONFIG['data']['uni_topic_matrix']}")

    print("=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)

    multi_topic_unis = [s for s in university_stats if s["num_unique_topics"] > 1]
    multi_policy_unis = [s for s in university_stats if s["num_policies"] > 1]
    n_unis = len(university_stats)
    n_docs = len(df)

    print(f"Total Universities: {n_unis}")
    print(f"Total Policies: {n_docs}")
    print(f"Fixed Topics: {best_k}")
    print(f"Average Policies per University: {n_docs / n_unis:.1f}")
    print(f"Average Topics per University: {df.groupby('university_name')['lda_topic'].nunique().mean():.1f}")
    print(f"Universities with >1 topic: {len(multi_topic_unis)} ({len(multi_topic_unis)/n_unis*100:.1f}%)")
    print(f"Universities with >1 policy: {len(multi_policy_unis)} ({len(multi_policy_unis)/n_unis*100:.1f}%)")

    print("=" * 80)
    print("COMPLETE - 6 TOPIC LDA ANALYSIS FINISHED")
    print("=" * 80)
    print("ALL FILES CREATED:")
    print(f" - {CONFIG['data']['lda_topics']}")
    print(f" - {CONFIG['data']['lda_viz']}")
    print(f" - {CONFIG['data']['lda_summary']}")
    print(f" - {CONFIG['data']['lda_results']}")
    print(f" - {CONFIG['data']['uni_policy_topic_mapping']}")
    print(f" - {CONFIG['data']['uni_topic_statistics']}")
    print(f" - {CONFIG['data']['uni_topic_matrix']}")
