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
├── data/
│   ├── raw_extracted_data.xlsx
│   ├── manual_corrected_data.xlsx
│   ├── merged_dataset.csv
│   ├── final_clean_dataset.csv
│
├── extraction/
│   └── extract_policies.py
│
├── manual/
│   └── merge_manual_data.py
│
├── cleaning/
│   └── clean_data.py
│
├── README.md
└── requirements.txt

