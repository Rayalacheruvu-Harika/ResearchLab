import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pathlib import Path

# --------------------
# Load data
# --------------------
df = pd.read_csv("data/final_clean_dataset.csv")

# Convert stringified tokens -> list
texts = df["tokens"].apply(eval).tolist()

# --------------------
# Build corpus
# --------------------
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# --------------------
# Train LDA
# --------------------
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=6,          # MUST match your existing LDA
    passes=10,
    random_state=42,
    alpha="auto",
    eta="auto"
)

# --------------------
# Save model
# --------------------
Path("lda_analysis").mkdir(exist_ok=True)
lda_model.save("lda_analysis/lda_model.gensim")

print("✅ LDA model saved to lda_analysis/lda_model.gensim")
