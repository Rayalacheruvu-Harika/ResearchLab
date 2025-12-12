"""
label_topics.py

This script:
- Loads a BERTopic model
- Loads the topic modelling results CSV (with clean_text + topic columns)
- Automatically assigns human-readable labels to each topic
- Uses both predefined labels + keyword matching heuristics
- Saves a clean topic-label dataframe to disk
"""

import pandas as pd
from bertopic import BERTopic

# -----------------------------
# Paths (EDIT IF NEEDED)
# -----------------------------
MODEL_PATH = "models/bertopic_model/bertopic_model.pkl"
DATA_PATH = "../data/topic_model_results.csv"
OUTPUT_PATH = "../data/topic_labels.csv"


# ----------------------------------------
# EXPANDED MASTER LABEL DICTIONARY
# ----------------------------------------
PREDEFINED_LABELS = {

    # Academic Integrity Themes
    "integrity": "Academic Integrity & Misconduct Prevention",
    "misconduct": "Academic Integrity & Misconduct Prevention",
    "plagiarism": "Academic Integrity & Plagiarism Detection",
    "cheating": "Assessment Misconduct & Cheating Prevention",
    "unauthorized": "Unauthorized AI Use Restrictions",

    # Ethical / Responsible AI
    "ethical": "Ethical & Responsible AI Use",
    "responsible": "Ethical & Responsible AI Use",
    "bias": "Bias, Fairness & Accountability",
    "fairness": "Bias, Fairness & Accountability",
    "transparency": "Transparency & Disclosure Requirements",
    "accountability": "AI Governance, Accountability & Compliance",

    # Assessment & Evaluation
    "assessment": "Assessment Rules & AI Restrictions",
    "exam": "Assessment Rules & AI Restrictions",
    "grading": "Assessment & Evaluation Policies",
    "evaluate": "Assessment & Evaluation Policies",

    # Teaching / Learning / Pedagogy
    "learning": "AI for Teaching & Learning Support",
    "teaching": "AI for Teaching & Learning Support",
    "support": "AI Literacy, Instruction & Training",
    "pedagogy": "AI-Enhanced Pedagogy & Instruction",
    "feedback": "AI Feedback, Editing & Writing Assistance",

    # Governance / Policy Framework
    "policy": "Institutional AI Policy & Governance",
    "governance": "Institutional AI Policy & Governance",
    "regulation": "Regulation, Compliance & Legal Risk",
    "standards": "AI Policy Standards & Implementation",

    # Privacy / Data Security
    "privacy": "Data Security, Privacy & Institutional Risk",
    "security": "Data Security, Privacy & Institutional Risk",
    "data": "Data Management & AI Usage Safety",
    "protection": "Student Data Protection Obligations",

    # Tools / Technology
    "chatgpt": "Generative AI Tools: Opportunities & Risks",
    "llm": "Large Language Model Usage Guidelines",
    "software": "AI Detection Tools & Monitoring Systems",
    "detection": "AI Detection Tools & Monitoring Systems",

    # Capacity / Training / Infrastructure
    "training": "Staff & Student AI Training Initiatives",
    "capacity": "AI Capacity Building & Workforce Preparedness",
    "infrastructure": "AI Infrastructure & Institutional Readiness",
    "evaluation": "Monitoring, Compliance & Review Processes",
}


# ------------------------------------------------------
# Keyword Matching Function
# ------------------------------------------------------
def match_label_by_keywords(keywords):
    """
    Assign the most appropriate label based on keyword matching.
    """
    for word in keywords:
        word_lower = word.lower()
        for key, label in PREDEFINED_LABELS.items():
            if key in word_lower:
                return label
    return None


# -----------------------------
# Load the BERTopic model
# -----------------------------
print("\nLoading BERTopic model from:", MODEL_PATH)
topic_model = BERTopic.load(MODEL_PATH)

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

if "topic" not in df.columns:
    raise ValueError("Dataset must contain a 'topic' column.")

if "clean_text" not in df.columns:
    raise ValueError("Dataset must contain a 'clean_text' column.")

topic_info = topic_model.get_topic_info()
topic_ids = topic_info["Topic"].tolist()


# -----------------------------
# Generate Labels
# -----------------------------
labels = []

print("\nGenerating labels for topics...")

for topic_id in topic_ids:

    if topic_id == -1:
        continue  # skip outlier/no-topic cluster

    # Retrieve top words for topic
    words = topic_model.get_topic(topic_id)
    if words is None:
        continue

    keyword_list = [w[0] for w in words[:10]]
    top_keywords = ", ".join(keyword_list)

    # Step 1: Keyword-based label assignment
    auto_label = match_label_by_keywords(keyword_list)

    # Step 2: Fallback to topic ID mapping (if exists)
    if auto_label is None:
        auto_label = f"Topic {topic_id} (Uncategorized)"

    # Retrieve example texts
    sample_docs = df[df["topic"] == topic_id]["clean_text"].head(3).tolist()

    labels.append({
        "topic_id": topic_id,
        "assigned_label": auto_label,
        "top_keywords": top_keywords,
        "example_1": sample_docs[0] if len(sample_docs) > 0 else "",
        "example_2": sample_docs[1] if len(sample_docs) > 1 else "",
        "example_3": sample_docs[2] if len(sample_docs) > 2 else "",
    })


# -----------------------------
# Save Final Label Table
# -----------------------------
labels_df = pd.DataFrame(labels)
labels_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("\n✓ Topic labeling completed")
print("✓ Saved labeled topic file →", OUTPUT_PATH)
print("--------------------------------------------------------\n")

