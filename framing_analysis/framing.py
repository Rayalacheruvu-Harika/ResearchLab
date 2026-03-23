import pandas as pd
import os
from config import CONFIG
# --------------------------------------------------
# Paths
# --------------------------------------------------
INPUT = CONFIG["data"]["uni_multi_topic_profiles"]
OUTPUT = CONFIG["data"]["university_framing_profiles"]

# --------------------------------------------------
# FRAME DEFINITIONS (INSIDE CODE – as requested)
# --------------------------------------------------
FRAME_MAP = {
    "AI Governance": "Regulatory / Governance",
    "Assessment Policies": "Control & Integrity",
    "Academic Integrity": "Control & Integrity",
    "Data Privacy": "Risk & Compliance",
    "Teaching & Learning": "Pedagogical Support",
    "GenAI Tools": "Technological Enablement"
}


df = pd.read_csv(INPUT)

# --------------------------------------------------
# Expand multi-topic column
# --------------------------------------------------
df_expanded = df.assign(
    topic_name=df["multi_topics"].str.split("; ")
).explode("topic_name")

# --------------------------------------------------
# Map topics → frames
# --------------------------------------------------
df_expanded["frame"] = df_expanded["topic_name"].map(FRAME_MAP)

# Safety check (should not trigger if labels are correct)
df_expanded["frame"] = df_expanded["frame"].fillna("Unmapped Frame")

# --------------------------------------------------
# Aggregate frames per university
# --------------------------------------------------
final = (
    df_expanded.groupby(["url", "country"])["frame"]
    .apply(lambda x: "; ".join(sorted(set(x))))
    .reset_index()
)

# --------------------------------------------------
# Save output
# --------------------------------------------------
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
final.to_csv(OUTPUT, index=False)
print("RQ1 Framing analysis completed")
print("Saved →", OUTPUT)
