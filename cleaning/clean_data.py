import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_excel("data/merged_dataset.xlsx")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

df["clean_text"] = df["guideline_text"].apply(preprocess)
df["tokens"] = df["clean_text"].str.split()
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

ai_terms = ["ai", "artificial intelligence", "chatgpt", "llm", "genai", "machine learning"]
df["ai_term_count"] = df["clean_text"].apply(lambda t: sum(t.count(a) for a in ai_terms))

df.to_csv("data/final_clean_dataset.csv", index=False, encoding="utf-8")
print("✔ Saved → data/final_clean_dataset.csv")
