# LLM Policy Analysis – Research Lab

A comprehensive research pipeline for analyzing Generative AI and Large Language Model (LLM) policy guidelines across universities in multiple countries using advanced NLP and statistical techniques.

---

## 📋 Table of Contents

- Project Overview
- Research Questions
- Methodology
- System Requirements
- Setup & Installation
- Pipeline Architecture
- How to Run the Project
- Results & Evaluation
- Output Files
- Repository Structure
- Research Context & Citation
- Contributing
- License & Contact

---

## 🎯 Project Overview

This project investigates how universities across different countries frame, govern, and regulate the use of Generative AI and LLMs. Using topic modeling, policy framing analysis, and statistical evaluation, the study identifies similarities and differences in institutional AI governance.

**Scope & Scale**

- 45 universities
- 5 countries: Germany, UK, USA, Canada, Australia
- 1,200+ policy documents
- Dual topic modeling: LDA (6 topics) and BERTopic (7 topics)
- 13 quantitative evaluation metrics


---

## 🛠️ Methodology

### Data Collection

- Web scraping using **Selenium** for dynamic pages
- HTML extraction with **Trafilatura**
- PDF extraction via **PDFMiner**
- Manual correction for protected or missing pages

### Data Processing

- Text cleaning and normalization
- Tokenization and lemmatization
- Negation handling and AI-term detection

### Analysis Techniques

| Technique          | Purpose                      | Output                   |
| ------------------ | ---------------------------- | ------------------------ |
| LDA                | Probabilistic topic modeling | 6 interpretable topics   |
| BERTopic           | Transformer-based clustering | 7 semantic topics        |
| Chi-Square Test    | Country × Topic association | Statistical significance |
| Coherence (c_v)    | Topic quality                | 0–1 scale               |
| Clustering Metrics | Topic separation & quality   | Silhouette, DB Index     |
| Policy Framing     | Rhetorical analysis          | Governance frames        |

---

## 💻 System Requirements

- Python **3.9 – 3.11** (recommended: 3.10)
- OS: Windows, macOS, or Linux
- Minimum 8 GB RAM (16 GB recommended for BERTopic)
- Internet access (for scraping stage only)

---

## 📦 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Rayalacheruvu-Harika/ResearchLab.git
cd ResearchLab
git checkout dash
```

### 2. Create and Activate a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages include:

- `gensim` – LDA topic modeling
- `bertopic` – Transformer-based topic modeling
- `scikit-learn` – Evaluation metrics
- `selenium`, `trafilatura`, `pdfminer` – Data extraction
- `pandas`, `numpy`, `matplotlib` – Data handling & visualization

### 4. Configure Paths

Edit `config/config.yaml` to match your local directory structure:

```yaml
data:
  clean: "data/final_clean_dataset.csv"
  lda_results: "data/lda_topic_model_results.csv"
  bert_results: "data/bert_topic_model_results.csv"

paths:
  models: "models/bertopic_model"
  analysis: "analysis_results"
```

---

## 🔄 Pipeline Architecture

```
Data Extraction (HTML/PDF)
        ↓
Manual Verification
        ↓
Text Cleaning & Preprocessing
        ↓
   ┌───────────┬
   ↓           ↓   
  LDA         BERTopic   
   ↓           ↓   
Evaluation Metrics (13 total)
        ↓
Policy Framing Analysis
        ↓
Results, Visualizations & Paper
```

---

## 🚀 How to Run the Project

### Run the Full Pipeline (Recommended)

Execute the following commands in order:

```bash
python extraction/extract_data.py
python manual/merge_manual_data.py
python cleaning/clean_data.py
python lda_analysis/lda_uni_topic_modeling.py
python bert_analysis/bert_topic_modeling.py
python bert_analysis/bert_topic_labeling.py
python framing_analysis/framing.py
python framing_analysis/framing_interpret.py
python run_evaluation.py
```

---

## 📊 Results & Evaluation Summary

### LDA (6 Topics)

- Chi-Square: **p = 0.001** (significant country differences)
- Coherence: 0.341
- Perplexity: -7.08
- Topic Diversity: 18.23

### BERTopic (7 Topics)

- Chi-Square: **p = 0.135** (not significant)
- Coherence: 0.457 (34% higher than LDA)
- Topic Diversity: 22.38
- Davies–Bouldin Index: 2.91 (better separation)

### Country-Level Consistency (Cosine Similarity)

- UK: Highest consistency (0.552)
- Australia: Most diverse policies (0.368)

---

## 📁 Output Files

All quantitative results are stored in `analysis_results/` as `.csv`, `.txt`, and `.json` files, including:

- Chi-square statistics
- Topic coherence scores
- Clustering quality metrics
- Consolidated evaluation reports

Trained models are saved under:

- `lda_analysis/lda_model.gensim`
- `models/bertopic_model/`
