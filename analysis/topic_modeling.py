import pandas as pd
import gensim
import pyLDAvis
import pyLDAvis.gensim


from gensim import corpora
from gensim.models import CoherenceModel
from collections import defaultdict

# 1. LOAD DATA
df = pd.read_excel("./final.xlsx")

needed_cols = [
    "url",
    "country",
    "document_type",
    "clean_text",
    "tokens",
    "word_count",
    "sentence_count",
    "ai_term_count"
]

df = df[needed_cols]

# Convert tokens (string) back to list
df["tokens"] = df["tokens"].apply(lambda x: eval(x) if isinstance(x, str) else x)

# 2. DICTIONARY & CORPUS
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(text) for text in df['tokens']]

# 3. FUNCTION: COMPUTE COHERENCE
def compute_coherence(k):
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        passes=10,
        random_state=42
    )
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=df['tokens'],
        dictionary=dictionary,
        coherence='c_v',
        processes=1
    )
    return coherence_model.get_coherence()

# 4. FIND BEST NUMBER OF TOPICS
topic_range = range(2, 10)
coherence_scores = []

for k in topic_range:
    score = compute_coherence(k)
    coherence_scores.append((k, score))
    print(f"K={k}, Coherence={score}")

best_k = max(coherence_scores, key=lambda x: x[1])[0]
print("\nBest number of topics:", best_k)

# 5. TRAIN FINAL MODEL
lda_model = gensim.models.LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=best_k,
    passes=20,
    random_state=42
)

# 6. DOMINANT TOPIC PER DOCUMENT
def get_dominant_topic(bow):
    topics = lda_model.get_document_topics(bow)
    if topics:
        return max(topics, key=lambda x: x[1])[0]
    return None

df["dominant_topic"] = [get_dominant_topic(b) for b in corpus]

# 7. PRINT TOP WORDS PER TOPIC
topics = lda_model.print_topics(num_words=15)

with open("topics.txt", "w") as f:
    for topic_num, words in topics:
        f.write(f"Topic {topic_num}: {words}\n")

print("Saved: topics.txt")

# 8. PYLDAVIS VISUALIZATION
try:
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, "lda_visualization.html")
    print("Saved: lda_visualization.html")

except ImportError:
    print("pyLDAvis not installed. Run: pip install pyLDAvis")

# 9. ADD HUMAN-READABLE TOPIC LABELS (EDIT NAMES LATER)
topic_labels = {
    0: "Miscellaneous / Low-Frequency Content",
    1: "General Guidance on Generative AI Tools",
    2: "Responsible Use, Risks, and Policy Frameworks",
    3: "Teaching, Learning, and Assessment Guidance",
    4: "Student Use of Generative AI in Coursework",
    5: "Academic Integrity, Misconduct, and Plagiarism"
}

df["topic_label"] = df["dominant_topic"].map(topic_labels)

# 10. SHOW SAMPLE DOCUMENTS PER TOPIC

topic_docs = defaultdict(list)

for i, bow in enumerate(corpus):
    t = df.loc[i, "dominant_topic"]
    snippet = df.loc[i, "clean_text"][:200]
    topic_docs[t].append(snippet)

for t in topic_docs:
    print(f"\nSAMPLE DOCUMENTS FOR TOPIC {t}")
    for text in topic_docs[t][:3]:
        print("\n", text)

# 11. TOPIC SUMMARY TABLE
topic_summary = {
    "topic_num": [],
    "topic_label": [],
    "top_words": []
}

for topic_num, words in lda_model.print_topics(num_words=12):
    topic_summary["topic_num"].append(topic_num)
    topic_summary["topic_label"].append(topic_labels.get(topic_num, ""))
    topic_summary["top_words"].append(words)

summary_df = pd.DataFrame(topic_summary)
summary_df.to_csv("topic_summary.csv", index=False)
print("Saved: topic_summary.csv")

# 12. FINAL DATASET
df.to_csv("topic_modelled_policies.csv", index=False)
print("\nSaved: topic_modelled_policies.csv")
