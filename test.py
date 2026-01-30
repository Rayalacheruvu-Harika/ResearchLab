import sys
import os
from pathlib import Path
import pandas as pd
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load CSV file
csv_path = os.path.join(project_root, "data/final_clean_dataset.csv")
df = pd.read_csv(csv_path)

# Convert DataFrame to list of dictionaries
data_list = df.to_dict(orient='records')

# Write to JSON file
json_path = os.path.join(project_root, "data/final_clean_dataset.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data_list, f, indent=2, ensure_ascii=False)

print(f"✔ CSV converted to JSON: {json_path}")
print(f"✔ Total records: {len(data_list)}")