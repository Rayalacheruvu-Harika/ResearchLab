import trafilatura
import pandas as pd
import re
import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime
from io import BytesIO
from pdfminer.high_level import extract_text as pdf_extract

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

# -------------------------------
# Setup Selenium
# -------------------------------
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                          options=chrome_options)

def accept_cookies():
    for key in ["Accept", "Agree", "Continue", "Allow"]:
        try:
            btn = driver.find_element(By.XPATH, f"//button[contains(text(),'{key}')]")
            btn.click()
            return
        except:
            pass

# -------------------------------
# Helpers
# -------------------------------
def clean_text(t):
    return re.sub(r"\s+", " ", t).strip() if t else ""

def detect_country(url):
    u = url.lower()
    if ".ac.uk" in u: return "United Kingdom"
    if ".edu.au" in url or ".monash.edu" in url: return "Australia"
    if ".ca" in u: return "Canada"
    if ".edu" in u: return "United States"
    if ".de" in u: return "Germany"
    return "Unknown"

def extract_all(url):
    print("\nExtracting:", url)

    # PDF
    if url.lower().endswith(".pdf"):
        try:
            pdf_bytes = requests.get(url).content
            raw = pdf_extract(BytesIO(pdf_bytes))
            return clean_text(raw), "PDF"
        except:
            pass

    # Trafilatura
    try:
        html = trafilatura.fetch_url(url)
        txt = trafilatura.extract(html)
        if txt:
            return clean_text(txt), "Trafilatura"
    except:
        pass

    # Selenium
    try:
        driver.get(url)
        time.sleep(4)
        accept_cookies()
        soup = BeautifulSoup(driver.page_source, "html.parser")
        for x in soup(["script", "style", "nav", "footer", "header"]):
            x.extract()
        txt = clean_text(soup.get_text(" "))
        if len(txt) > 150:
            return txt, "Selenium"
    except:
        pass

    # Requests fallback
    try:
        resp = requests.get(url, timeout=12)
        soup = BeautifulSoup(resp.text, "html.parser")
        txt = clean_text(soup.get_text(" "))
        return txt, "Requests"
    except:
        return "", "Failed"


# -------------------------------
# Load URLs
# -------------------------------
with open("data/urls.txt", "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f.readlines() if line.strip()]

# -------------------------------
# Run extraction
# -------------------------------
results = []

for url in urls:
    text, method = extract_all(url)
    results.append({
        "url": url,
        "country": detect_country(url),
        "guideline_text": text,
        "word_count": len(text.split()),
        "status": method,
        "date_accessed": datetime.today().strftime("%Y-%m-%d")
    })

df = pd.DataFrame(results)
df.to_excel("data/raw_extracted_data.xlsx", index=False)
print("✔ Saved → data/raw_extracted_data.xlsx")

driver.quit()
