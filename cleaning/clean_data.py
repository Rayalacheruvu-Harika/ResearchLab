import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG
from config.logger import setup_logger

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = setup_logger(__name__)

nltk.download("stopwords")
nltk.download("wordnet")

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Starting text preprocessing pipeline")
    logger.info("=" * 80)
    
    df = pd.read_csv(CONFIG["data"]["merged"])
    logger.info(f"✅ Loaded {len(df)} documents from merged_dataset.csv")

    # PRESERVE NEGATIONS FOR SENTIMENT ANALYSIS
    standard_stop_words = set(stopwords.words("english"))
    negation_words = set(CONFIG["preprocessing"]["negation_words"])
    custom_stop_words = standard_stop_words - negation_words
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        words = [w for w in text.split() if w not in custom_stop_words and len(w) > 2]
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    logger.info("Starting text preprocessing...")
    df["clean_text"] = df["guideline_text"].apply(preprocess)
    df["tokens"] = df["clean_text"].str.split()
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    logger.info(f"✅ Preprocessed {len(df)} documents")

    ai_terms = CONFIG["preprocessing"]["ai_terms"]
    df["ai_term_count"] = df["clean_text"].apply(lambda t: sum(t.count(a) for a in ai_terms))
    logger.info("✅ Calculated AI term counts")

    logger.info("Saving preprocessed data...")
    df.to_csv(CONFIG["data"]["clean"], index=False, encoding="utf-8")
    logger.info(f"✅ Saved preprocessed data to data/final_clean_dataset.csv")

    logger.info("=" * 80)
    logger.info("Text preprocessing completed successfully")
    logger.info("=" * 80)
