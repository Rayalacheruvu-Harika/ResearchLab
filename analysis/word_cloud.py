import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# -------------------------
# CONFIG
# -------------------------
INPUT_CSV = "../data/final_clean_dataset.csv"
TEXT_COL = "clean_text"
OUTPUT_IMG = "../analysis_results/overall_policy_wordcloud.png"

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(INPUT_CSV)
texts = df[TEXT_COL].dropna().astype(str)

# Combine all policy text into one corpus
full_text = " ".join(texts)

# Optional: add domain-specific stopwords
custom_stopwords = set([
    "university", "universities", "students", "staff",
    "use", "using", "used", "may", "must", "should",
    "ai", "artificial", "intelligence", "generative"
])

stopwords = STOPWORDS.union(custom_stopwords)

# -------------------------
# Generate word cloud
# -------------------------
wordcloud = WordCloud(
    width=1400,
    height=700,
    background_color="white",
    stopwords=stopwords,
    max_words=100,
    collocations=False,
    colormap="GnBu"
).generate(full_text)

# -------------------------
# Plot & save
# -------------------------
plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)
plt.show()

print(f"✅ Word cloud saved to: {OUTPUT_IMG}")
