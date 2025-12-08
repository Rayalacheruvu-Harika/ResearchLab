# LLM Policy Analysis Dataset

This project extracts and analyzes Generative AI policy guidelines from global universities.

## Pipeline
1. Data extraction using Selenium + Trafilatura + PDFMiner
2. Manual correction of protected/missing pages
3. NLP-based cleaning and preprocessing

## How to Run
python extraction/extract_data.py
python manual/merge_manual_data.py
python cleaning/clean_data.py
