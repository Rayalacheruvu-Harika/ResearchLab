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
python extraction/extract_data.py

python manual/merge_manual_data.py

python cleaning/clean_data.py



LLM_POLICY_ANALYSIS/
│
├── data/
│   ├── raw_extracted_data.xlsx         # Automatically scraped
│   ├── manual_corrected_data.xlsx      # Human-filled missing policies
│   ├── merged_dataset.csv              # Combined dataset
│   ├── final_clean_dataset.csv         # After text cleaning
│
├── extraction/
│   └── extract_data.py             # Web scraping script
│
├── manual/
│   └── merge_manual_data.py            # Insert manually corrected text
│
├── cleaning/
│   └── clean_data.py                   # Preprocessing & NLP cleaning
│
│
├── README.md
└── requirements.txt
