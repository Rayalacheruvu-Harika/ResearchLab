"""
Improved Topic Labeling Script (Final Version)
---------------------------------------------
Outputs:
- topic_id
- topic_name (clean human topic label)
- top_keywords
- example texts

Ensures:
- NO "Other"
- NO "Unlabeled Topic"
- Every topic gets a readable label
"""

import pandas as pd
from bertopic import BERTopic
from config import CONFIG
from config.logger import setup_logger

logger = setup_logger(__name__)

# Category Rules → High-level labels
CATEGORY_RULES = {
    "policy": "AI Governance",
    "regulation": "AI Governance",
    "governance": "AI Governance",
    "accountability": "AI Governance",
    "learning": "Teaching & Learning",
    "teaching": "Teaching & Learning",
    "education": "Teaching & Learning",
    "assessment": "Assessment Policies",
    "exam": "Assessment Policies",
    "grading": "Assessment Policies",
    "genai": "GenAI Tools",
    "chatgpt": "GenAI Tools",
    "llm": "GenAI Tools",
    "ai tool": "GenAI Tools",
    "privacy": "Data Privacy",
    "security": "Data Privacy",
    "data": "Data Privacy",
    "integrity": "Academic Integrity",
    "plagiarism": "Academic Integrity",
    "misconduct": "Academic Integrity",
}


def assign_category(keywords):
    """Assign category based on keywords"""
    words = [w.lower() for w in keywords]
    for w in words:
        for key, category in CATEGORY_RULES.items():
            if key in w:
                return category
    # Fallback: temporarily None (NOT "Other")
    return None


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("BERT TOPIC LABELING")
    logger.info("=" * 80)

    # Paths
    MODEL_PATH = CONFIG["paths"]["bertopic_model"]
    DATA_PATH = CONFIG["data"]["bert_results"]
    OUTPUT_PATH = CONFIG["data"]["bert_labels"]

    # Load BERTopic model + dataset
    logger.info("Loading BERTopic model...")
    topic_model = BERTopic.load(MODEL_PATH)
    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    topic_info = topic_model.get_topic_info()
    topic_ids = [t for t in topic_info["Topic"] if t != -1]  # remove noise cluster

    # Generate labels
    labels = []
    logger.info("Generating topic labels...")
    for topic_id in topic_ids:
        topic_keywords = topic_model.get_topic(topic_id)
        keywords = [w[0] for w in topic_keywords[:10]]
        top_keywords = ", ".join(keywords)

        # Step 1 — try to assign a meaningful category
        category = assign_category(keywords)

        # Step 2 — If no category detected, create a clean name from keywords
        if category is None:
            main_word = keywords[0].capitalize()
            category = f"{main_word} Policy"

        # Step 3 — Create final human-readable topic name
        topic_name = category  # cleaned and readable

        # Example documents
        samples = df[df["topic"] == topic_id]["clean_text"].head(3).tolist()

        labels.append({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "top_keywords": top_keywords,
            "example_1": samples[0] if len(samples) > 0 else "",
            "example_2": samples[1] if len(samples) > 1 else "",
            "example_3": samples[2] if len(samples) > 2 else "",
        })

    # Save final label file
    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    logger.info("Topic labeling completed")
    logger.info(f"Saved: {OUTPUT_PATH}")
    logger.info("=" * 80)
