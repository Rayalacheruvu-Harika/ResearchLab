import pandas as pd
from datetime import datetime
from pathlib import Path
from config import CONFIG


project_root = Path(__file__).resolve().parents[1]  

raw_path = project_root / CONFIG["data"]["raw_genai"]
manual_path = project_root / CONFIG["data"]["manual_corrected"]

df = pd.read_csv(raw_path)
manual_df = pd.read_excel(manual_path)  # Your file CELL 2


def detect_document_type(text):
    if "policy" in text.lower(): return "Policy"
    if "guidance" in text.lower() or "guideline" in text.lower() or "recommendations" in text.lower(): return "Guideline"
    return "Unknown"

def detect_country(url):
    url = url.lower()
    if ".ac.uk" in url: return "United Kingdom"
    if ".ca" in url: return "Canada"
    if ".edu.au" in url or ".monash.edu" in url: return "Australia"
    if ".edu" in url: return "United States"
    if ".de" in url or ".tu.berlin" in url: return "Germany"
    return "Unknown"

def detect_tools(text):
    tools = ["ai", "chatgpt","gpt","bard","claude","gemini","copilot","llm","generative ai","genai","ai tool"]
    return ", ".join([t for t in tools if t in text.lower()])

def detect_target_audience(text):
    groups = {
        "Students": ["student"],
        "Faculty": ["faculty","professor","instructor"],
        "Researchers": ["research","researcher"],
        "Staff": ["staff","employee"]
    }
    found = [g for g, keys in groups.items() if any(k in text.lower() for k in keys)]
    return ", ".join(found) if found else "Unknown"


updated = 0

for _, row in manual_df.iterrows():
    url = row["url"]
    text = str(row["guideline_text"])

    mask = df["url"] == url
    if mask.any():
        df.loc[mask, "country"] = detect_country(url)
        df.loc[mask, "document_type"] =  detect_document_type(text)
        df.loc[mask, "tool_mentioned"] =  detect_tools(text)
        df.loc[mask, "target_audience"] =  detect_target_audience(text)
        df.loc[mask, "guideline_text"] = text
        df.loc[mask, "status"] = "Manual"
        df.loc[mask, "word_count"] = len(text.split())
        df.loc[mask, "sentence_count"] = text.count('.')
        df.loc[mask,"date_accessed"] = datetime.today().strftime("%Y-%m-%d")
        updated += 1

print(f"Manually updated rows: {updated}")

output_path = project_root / CONFIG["data"]["merged"]
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Saved --- {output_path}")
