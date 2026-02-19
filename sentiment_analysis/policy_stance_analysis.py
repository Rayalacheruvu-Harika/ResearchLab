import pandas as pd
import re
from transformers import pipeline

# Non-overlapping stance labels (best for university policy analysis)
STANCE_LABELS = [
    "SUPPORTIVE",
    "CAUTIONARY",
    "RESTRICTIVE",
    "PROHIBITIVE",
    "INFORMATIONAL"
]

# Restrictiveness score for your "Policy Restrictiveness Index"
RESTRICTIVENESS_SCORE = {
    "SUPPORTIVE": 1,
    "INFORMATIONAL": 2,
    "CAUTIONARY": 3,
    "RESTRICTIVE": 4,
    "PROHIBITIVE": 5
}

def split_into_chunks(text, max_chunk_chars=700):
    """
    Splits long policy text into smaller chunks so the model
    doesn't miss important sections.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # basic sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) + 1 <= max_chunk_chars:
            current += (" " + s).strip()
        else:
            if current:
                chunks.append(current)
            current = s

    if current:
        chunks.append(current)

    return chunks

def classify_policy_stance(
    input_csv="data/final_clean_dataset.csv",
    output_csv="data/final_clean_dataset_with_stance.csv",
    text_col="clean_text"
):
    df = pd.read_csv(input_csv)
    df[text_col] = df[text_col].fillna("")

    # zero-shot classifier (no manual training data needed)
    clf = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    final_labels = []
    final_scores = []
    restrictiveness = []

    for doc in df[text_col]:
        chunks = split_into_chunks(doc)

        # if empty, assign informational
        if not chunks:
            final_labels.append("INFORMATIONAL")
            final_scores.append(0.0)
            restrictiveness.append(RESTRICTIVENESS_SCORE["INFORMATIONAL"])
            continue

        label_votes = {l: 0 for l in STANCE_LABELS}
        label_conf_sum = {l: 0.0 for l in STANCE_LABELS}

        for chunk in chunks:
            result = clf(
                chunk,
                candidate_labels=STANCE_LABELS,
                hypothesis_template="This university guideline is {}."
            )

            top_label = result["labels"][0]
            top_score = result["scores"][0]

            label_votes[top_label] += 1
            label_conf_sum[top_label] += top_score

        # majority vote (interpretable + stable)
        best_label = max(label_votes, key=label_votes.get)

        # average confidence of winning label
        avg_conf = label_conf_sum[best_label] / max(label_votes[best_label], 1)

        final_labels.append(best_label)
        final_scores.append(avg_conf)
        restrictiveness.append(RESTRICTIVENESS_SCORE[best_label])

    df["policy_stance_label"] = final_labels
    df["policy_stance_confidence"] = final_scores
    df["restrictiveness_index"] = restrictiveness

    df.to_csv(output_csv, index=False)

    print("✅ Saved stance results to:", output_csv)
    print("\n📌 Stance label distribution:")
    print(df["policy_stance_label"].value_counts())
    print("\n📌 Average restrictiveness index:", df["restrictiveness_index"].mean())

if __name__ == "__main__":
    classify_policy_stance()