# LLM Policy Analysis Dataset

This project extracts and analyzes Generative AI policy guidelines from global universities.

**#Setup Guide**

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

## How to Run
1. python extraction/extract_data.py
2. python manual/merge_manual_data.py
3. python cleaning/clean_data.py



LLM_POLICY_ANALYSIS/
│
├──1. data/
│   ├──1. raw_extracted_data.xlsx         # Automatically scraped
│   ├──2. manual_corrected_data.xlsx      # Human-filled missing policies
│   ├──3. merged_dataset.csv              # Combined dataset
│   ├──4. final_clean_dataset.csv         # After text cleaning
│
├──2. extraction/
│   └──1. extract_data.py             # Web scraping script
│
├──3. manual/
│   └──1. merge_manual_data.py            # Insert manually corrected text
│
├──4. cleaning/
│   └──1. clean_data.py                   # Preprocessing & NLP cleaning
│
│
├──5. README.md
└──6. requirements.txt
