import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pathlib import Path
from config import CONFIG
# --------------------
# Load data
# --------------------
df = pd.read_csv(CONFIG["data"]["clean"])

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
    num_topics=CONFIG["lda"]["num_topics"],
    passes=CONFIG["lda"]["passes"],
    random_state=CONFIG["lda"]["random_state"],
    alpha="auto",
    eta="auto"
)

# --------------------
# --------------------
# Save model
# --------------------
lda_dir = Path(CONFIG["data"]["lda_model"]).parent
lda_dir.mkdir(parents=True, exist_ok=True)

lda_model.save(str(CONFIG["data"]["lda_model"]))
dictionary.save(str(lda_dir / "dictionary.pkl"))
 

print(f"LDA model saved to {CONFIG['data']['lda_model']}")
