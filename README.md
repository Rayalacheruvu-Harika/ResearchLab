# LLM Policy Analysis Dataset

This project extracts and analyzes Generative AI policy guidelines from global universities.

# Setup Guide

**1. Create Virtual Environment**

cd "C:\Users\harik\Documents\Research_Lab\LLM_Policy_Analysis"

python -m venv .venv

.venv\Scripts\activate

**2. Install required libraries**

pip install -r requirements.txt

## Pipeline
1. Data extraction using Selenium + Trafilatura + PDFMiner
2. Manual correction of protected/missing pages
3. NLP-based cleaning and preprocessing
4. NLP-based computational analysis - baseline Topic modeling using LDA 
5. NLP-based computational analysis - advanced Topic modeling using Bertopic
6. Multi-topic profiling and Topic co-occurence
7. Policy Framing analysis
8. Evaluation

## How to Run
1. python extraction/extract_data.py
2. python manual/merge_manual_data.py
3. python cleaning/clean_data.py
4. python lda_analysis/lda_uni_topic_modeling.py
5. python bert_analysis/bert_topic_modeling.py
6. python bert_analysis/bert_topic_labeling.py
7. python bert_analysis/country_distribution.py
8. python bert_analysis/university_multi_topic_profiling.py
9. python bert_analysis/topic_cooccurence.py
10. python bert_analysis/country_topic_analysis_all.py
11. python bert_analysis/uni_multi_topic_profiling.py
12. python framing_analysis/framing.py
13. python framing_analysis/framing_interpret.py
14. python evaluation/chi_square_country_topic.py


