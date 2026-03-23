import pandas as pd
import re
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from config import CONFIG

INPUT_CSV = CONFIG["data"]["clean"]
OUTPUT_CSV = CONFIG["data"]["clean_sentiment_stance"]
TEXT_COL = "clean_text"

STANCE_LABELS = [
    "SUPPORTIVE",      # encourages usage
    "CAUTIONARY",      # warns but allows
    "RESTRICTIVE",     # conditional / limited usage
    "PROHIBITIVE",     # banned / misconduct
    "INFORMATIONAL"    # neutral policy description
]


# ----------------------------
# Helper: chunk long docs
# ----------------------------
def split_into_chunks(text, max_chunk_chars=700):
    """
    Split long policy text into smaller sentence-based chunks
    so BERT/Transformers can process properly.
    """
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []

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


# ----------------------------
# VADER Sentiment
# ----------------------------
def vader_sentiment(text, vader):
    scores = vader.polarity_scores(text)
    # scores: {'neg':..., 'neu':..., 'pos':..., 'compound':...}
    return scores


# ----------------------------
# BERT Sentiment (Transformers)
# ----------------------------
def bert_sentiment(text, sentiment_pipe):
    """
    Returns label + confidence.
    Example label: POSITIVE / NEGATIVE (depends on model)
    """
    out = sentiment_pipe(text[:1200])[0]  # truncate to avoid max length issues
    return out["label"], float(out["score"])


# ----------------------------
# BERT Stance (Zero-shot)
# ----------------------------
def bert_stance(text, stance_pipe):
    chunks = split_into_chunks(text)
    if not chunks:
        return "INFORMATIONAL", 0.0

    votes = {label: 0 for label in STANCE_LABELS}
    conf_sum = {label: 0.0 for label in STANCE_LABELS}

    for chunk in chunks:
        result = stance_pipe(
            chunk,
            candidate_labels=STANCE_LABELS,
            hypothesis_template="This university policy stance is {}."
        )

        top_label = result["labels"][0]
        top_score = float(result["scores"][0])

        votes[top_label] += 1
        conf_sum[top_label] += top_score

    final_label = max(votes, key=votes.get)
    avg_conf = conf_sum[final_label] / max(votes[final_label], 1)

    return final_label, avg_conf


def main():
    print(" Loading dataset:", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    # ---- VADER setup ----
    vader = SentimentIntensityAnalyzer()

    # ---- Transformer pipelines ----
    print(" Loading BERT sentiment model...")
    sentiment_pipe = pipeline("sentiment-analysis")

    print(" Loading BERT stance (zero-shot) model...")
    stance_pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # ---- Run analysis ----
    vader_neg, vader_neu, vader_pos, vader_compound = [], [], [], []
    bert_sent_labels, bert_sent_scores = [], []
    stance_labels, stance_scores = [], []

    for i, text in enumerate(df[TEXT_COL].tolist(), start=1):
        # VADER
        vs = vader_sentiment(text, vader)
        vader_neg.append(vs["neg"])
        vader_neu.append(vs["neu"])
        vader_pos.append(vs["pos"])
        vader_compound.append(vs["compound"])

        # BERT Sentiment
        b_label, b_score = bert_sentiment(text, sentiment_pipe)
        bert_sent_labels.append(b_label)
        bert_sent_scores.append(b_score)

        # BERT Stance
        s_label, s_score = bert_stance(text, stance_pipe)
        stance_labels.append(s_label)
        stance_scores.append(s_score)

        if i % 5 == 0:
            print(f"Processed {i}/{len(df)} docs...")

    # ---- Save columns ----
    df["vader_neg"] = vader_neg
    df["vader_neu"] = vader_neu
    df["vader_pos"] = vader_pos
    df["vader_compound"] = vader_compound

    df["bert_sentiment_label"] = bert_sent_labels
    df["bert_sentiment_confidence"] = bert_sent_scores

    df["policy_stance_label"] = stance_labels
    df["policy_stance_confidence"] = stance_scores

    df.to_csv(OUTPUT_CSV, index=False)
    print("\n Saved output:", OUTPUT_CSV)

    print("\n VADER compound summary:")
    print(df["vader_compound"].describe())

    print("\n BERT Sentiment label counts:")
    print(df["bert_sentiment_label"].value_counts())

    print("\n Policy stance label counts:")
    print(df["policy_stance_label"].value_counts())


if __name__ == "__main__":
    main()