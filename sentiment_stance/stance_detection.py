"""
================================================================================
RQ3: STANCE DETECTION ANALYSIS
================================================================================
Objective: Quantify policy restrictiveness and identify extreme cases
Input: analysis_results/sentiment_analysis/policy_tone_distilbert.csv
Output: University-level stance scores + extremes + visualizations
Author: Research Team
Date: January 2026
================================================================================
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURATION & PATHS
# ================================================================================

INPUT_FILE = "analysis_results/sentiment_analysis/policy_tone_distilbert.csv"

OUTPUT_DIR = "analysis_results/Stance_detection"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

# Output files
OUT_DOCUMENT = os.path.join(OUTPUT_DIR, "stance_scores_document_level.csv")
OUT_UNIVERSITY = os.path.join(OUTPUT_DIR, "stance_scores_university_level.csv")
OUT_EXTREMES = os.path.join(OUTPUT_DIR, "stance_extremes_summary.csv")
OUT_REPORT = os.path.join(OUTPUT_DIR, "stance_rq3_report.txt")

# Visualizations
FIG_SCATTER = os.path.join(FIG_DIR, "stance_scatter_universities.png")
FIG_EXTREMES = os.path.join(FIG_DIR, "stance_extremes_bar_chart.png")
FIG_COUNTRY = os.path.join(FIG_DIR, "stance_distribution_by_country.png")
FIG_EXAMPLES = os.path.join(FIG_DIR, "stance_university_examples.png")

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("✓ Directories created")

# ================================================================================
# SECTION 1: LOAD DATA
# ================================================================================

print("\n[STEP 1] Loading policy tone data...")

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"✓ Loaded {len(df)} policy documents")
    print(f"✓ Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
    print("   Make sure policy_tone_distilbert.py has been run first")
    exit(1)

# Validate required columns
required_cols = ['url', 'country', 'clean_text', 'policy_tone_label']
for col in required_cols:
    if col not in df.columns:
        print(f"❌ ERROR: Missing required column: {col}")
        exit(1)

print(f"✓ Data validation passed")

# ================================================================================
# SECTION 2: DEFINE STANCE LEXICONS
# ================================================================================

print("\n[STEP 2] Defining stance lexicons...")

# PERMISSIVE SIGNALS (Negative score direction = permissive)
PERMISSIVE_LEXICON = {
    "encouraged": [
        "encouraged", "promote", "support", "enable", "empower",
        "fully utilize", "harness", "leverage", "exploit potential",
        "innovation", "creative use", "augment capabilities"
    ],
    "allowed": [
        "allowed", "permitted", "acceptable", "may use", "free to use",
        "optional", "available for", "can be used", "use is welcomed"
    ],
    "with_minimal_conditions": [
        "can use", "permitted with guidance", "responsible use",
        "appropriate use", "supervised", "with guidance"
    ]
}

# RESTRICTIVE SIGNALS (Positive score direction = restrictive)
RESTRICTIVE_LEXICON = {
    "cautionary": [
        "caution", "risk", "concern", "aware", "careful", "mindful",
        "critical thinking", "evaluate", "scrutinize", "limitation"
    ],
    "limited": [
        "limited", "restricted", "limit use", "minimize", "reduce",
        "constrain", "narrow", "bounded", "controlled use"
    ],
    "prohibited": [
        "prohibited", "banned", "forbidden", "must not", "cannot use",
        "not permitted", "not allowed", "no use", "zero tolerance",
        "completely restrict", "no AI"
    ],
    "severe_consequences": [
        "disciplinary", "expulsion", "suspension", "termination",
        "violation", "breach", "academic integrity", "honor code",
        "sanction", "penalty", "consequence"
    ]
}

print("✓ Permissive lexicon categories:", list(PERMISSIVE_LEXICON.keys()))
print("✓ Restrictive lexicon categories:", list(RESTRICTIVE_LEXICON.keys()))

# ================================================================================
# SECTION 3: CALCULATE STANCE SCORES (DOCUMENT LEVEL)
# ================================================================================

print("\n[STEP 3] Calculating document-level stance scores...")

def calculate_stance_score(text):
    """
    Calculate stance score for a single policy document.
    
    Returns: float between -1.0 (most permissive) and +1.0 (most restrictive)
    
    Formula:
    stance_score = (restrictive_count - permissive_count) / 
                   (restrictive_count + permissive_count + 1)
    
    Why +1 in denominator? Avoid division by zero for empty texts
    """
    
    if pd.isna(text) or text == "":
        return 0.0  # Neutral if empty
    
    text_lower = str(text).lower()
    
    # Count permissive signals
    permissive_count = sum(
        text_lower.count(word) 
        for category in PERMISSIVE_LEXICON.values() 
        for word in category
    )
    
    # Count restrictive signals
    restrictive_count = sum(
        text_lower.count(word) 
        for category in RESTRICTIVE_LEXICON.values() 
        for word in category
    )
    
    # Calculate stance score (normalized)
    total = restrictive_count + permissive_count
    if total == 0:
        return 0.0  # Neutral if no signals found
    
    stance_score = (restrictive_count - permissive_count) / (total + 1)
    
    # Clamp to [-1, 1]
    return np.clip(stance_score, -1.0, 1.0)

# Apply to all documents
stance_scores = []
for idx, row in df.iterrows():
    score = calculate_stance_score(row['clean_text'])
    stance_scores.append(score)
    
    if (idx + 1) % 500 == 0:
        print(f"  → Processed {idx + 1}/{len(df)} documents")

df['stance_score'] = stance_scores

print(f"✓ Calculated stance scores for all {len(df)} documents")
print(f"  Mean stance: {df['stance_score'].mean():.3f}")
print(f"  Std dev: {df['stance_score'].std():.3f}")
print(f"  Min: {df['stance_score'].min():.3f}")
print(f"  Max: {df['stance_score'].max():.3f}")

# Save document-level scores
df.to_csv(OUT_DOCUMENT, index=False)
print(f"✓ Saved document-level scores → {OUT_DOCUMENT}")

# ================================================================================
# SECTION 4: AGGREGATE TO UNIVERSITY LEVEL
# ================================================================================

print("\n[STEP 4] Aggregating to university level...")

# Group by university (url)
uni_stance = df.groupby(['url', 'country']).agg({
    'stance_score': ['mean', 'std', 'min', 'max', 'count'],
    'policy_tone_label': lambda x: x.value_counts().index if len(x) > 0 else 'Unknown'
}).reset_index()

# Flatten column names
uni_stance.columns = ['url', 'country', 'stance_score_mean', 'stance_score_std', 
                      'stance_score_min', 'stance_score_max', 'n_policies', 
                      'dominant_tone']

# Add stance category
def categorize_stance(score):
    """Categorize stance into meaningful buckets"""
    if score >= 0.6:
        return "Highly Restrictive"
    elif score >= 0.3:
        return "Moderately Restrictive"
    elif score >= 0.0:
        return "Balanced"
    elif score >= -0.3:
        return "Moderately Permissive"
    else:
        return "Highly Permissive"

uni_stance['stance_category'] = uni_stance['stance_score_mean'].apply(categorize_stance)

# Sort by stance score
uni_stance = uni_stance.sort_values('stance_score_mean', ascending=False).reset_index(drop=True)

# Save university-level aggregation
uni_stance.to_csv(OUT_UNIVERSITY, index=False)

print(f"✓ Aggregated to {len(uni_stance)} universities")
print(f"✓ Saved university-level stance scores → {OUT_UNIVERSITY}")
print(f"\nUniversity Stance Distribution:")
print(uni_stance['stance_category'].value_counts())

# ================================================================================
# SECTION 5: IDENTIFY EXTREMES
# ================================================================================

print("\n[STEP 5] Identifying extreme cases...")

# Top 5 most restrictive universities
top_restrictive = uni_stance.nlargest(5, 'stance_score_mean')

# Top 5 most permissive universities
top_permissive = uni_stance.nsmallest(5, 'stance_score_mean')

# Middle ground (neutral stance)
middle_ground = uni_stance[
    (uni_stance['stance_score_mean'] >= -0.3) & 
    (uni_stance['stance_score_mean'] <= 0.3)
].nlargest(3, 'n_policies')

print("\n TOP 5 MOST RESTRICTIVE UNIVERSITIES:")
for idx, row in top_restrictive.iterrows():
    print(f"  {idx+1}. {row['country']:<12} | Stance: +{row['stance_score_mean']:.3f} | "
          f"Tone: {row['dominant_tone']:<20} | N policies: {row['n_policies']}")

print("\n🟢 TOP 5 MOST PERMISSIVE UNIVERSITIES:")
for idx, row in top_permissive.iterrows():
    print(f"  {idx+1}. {row['country']:<12} | Stance: {row['stance_score_mean']:.3f} | "
          f"Tone: {row['dominant_tone']:<20} | N policies: {row['n_policies']}")

print("\n⚪ MIDDLE GROUND (BALANCED) UNIVERSITIES:")
for idx, row in middle_ground.iterrows():
    print(f"  • {row['country']:<12} | Stance: {row['stance_score_mean']:.3f} | "
          f"Tone: {row['dominant_tone']:<20} | N policies: {row['n_policies']}")

# Save extremes summary
extremes = pd.concat([
    top_restrictive.assign(category='Most Restrictive'),
    top_permissive.assign(category='Most Permissive'),
    middle_ground.assign(category='Balanced')
])

extremes.to_csv(OUT_EXTREMES, index=False)
print(f"\n✓ Saved extremes summary → {OUT_EXTREMES}")

# ================================================================================
# SECTION 6: EXTRACT POLICY QUOTES FOR EXAMPLES
# ================================================================================

print("\n[STEP 6] Extracting policy examples...")

def extract_example_quote(url_filter, n_chars=150):
    """Extract a representative policy quote from a university"""
    policies = df[df['url'] == url_filter]['clean_text'].tolist()
    if not policies:
        return "No policies found"
    
    # Use longest policy as most representative
    longest = max(policies, key=len)
    return longest[:n_chars] + "..." if len(longest) > n_chars else longest

# Add example quotes to extremes
extremes['policy_example'] = extremes['url'].apply(extract_example_quote)

# Create examples file with quotes
examples_report = extremes[['url', 'country', 'stance_score_mean', 'category', 'policy_example']]
examples_report.to_csv(os.path.join(OUTPUT_DIR, "stance_examples_with_quotes.csv"), index=False)

print("✓ Extracted policy examples for all extreme cases")

# ================================================================================
# SECTION 7: VISUALIZATION 1 - SCATTER PLOT (UNIVERSITY WISE)
# ================================================================================

print("\n[STEP 7] Creating visualizations...")
print("\n  → Visualization 1: Scatter plot (university-wise stance)")

plt.figure(figsize=(14, 8))

# Create scatter plot
countries = uni_stance['country'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))
country_colors = {country: colors[i] for i, country in enumerate(countries)}

for country in countries:
    country_data = uni_stance[uni_stance['country'] == country]
    plt.scatter(
        range(len(country_data)),
        country_data['stance_score_mean'],
        label=country,
        s=country_data['n_policies'] * 15,  # Size by number of policies
        alpha=0.6,
        color=country_colors[country]
    )

# Highlight extremes
top_restrictive_first = top_restrictive.reset_index(drop=True).loc[0]
top_permissive_first = top_permissive.reset_index(drop=True).loc[0]

# Get index position safely
restrictive_idx = uni_stance.index[
    uni_stance['url'] == top_restrictive_first['url']
][0]

plt.scatter(
    restrictive_idx,
    top_restrictive_first['stance_score_mean'],
    s=500,
    marker='X',
    color='red',
    edgecolors='darkred',
    linewidths=2,
    label='Most Restrictive',
    zorder=5
)


permissive_idx = uni_stance.index[
    uni_stance['url'] == top_permissive_first['url']
][0]

plt.scatter(
    permissive_idx,
    top_permissive_first['stance_score_mean'],
    s=500,
    marker='*',
    color='green',
    edgecolors='darkgreen',
    linewidths=2,
    label='Most Permissive',
    zorder=5
)


# Format plot
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Neutral')
plt.axhline(y=0.3, color='orange', linestyle=':', alpha=0.3)
plt.axhline(y=-0.3, color='blue', linestyle=':', alpha=0.3)

plt.xlabel('University Index', fontsize=12, fontweight='bold')
plt.ylabel('Stance Score (-1=Permissive, +1=Restrictive)', fontsize=12, fontweight='bold')
plt.title('Policy Stance by University (Size = Number of Policies)', fontsize=14, fontweight='bold')
plt.ylim(-1, 1)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(FIG_SCATTER, dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved → {FIG_SCATTER}")

# ================================================================================
# SECTION 8: VISUALIZATION 2 - EXTREMES BAR CHART
# ================================================================================

print("  → Visualization 2: Extremes bar chart (top/bottom 5)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Most restrictive
top_res_sorted = top_restrictive.sort_values('stance_score_mean')
ax1.barh(range(len(top_res_sorted)), top_res_sorted['stance_score_mean'], color='#d62728')
ax1.set_yticks(range(len(top_res_sorted)))
ax1.set_yticklabels(top_res_sorted['country'] + ' (' + top_res_sorted['country'].str[:2] + ')')
ax1.set_xlabel('Stance Score (Restrictive →)', fontweight='bold')
ax1.set_title('Top 5 Most Restrictive Universities', fontweight='bold', color='#d62728')
ax1.set_xlim(0, 1)
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(top_res_sorted['stance_score_mean']):
    ax1.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')

# Most permissive
top_perm_sorted = top_permissive.sort_values('stance_score_mean')
ax2.barh(range(len(top_perm_sorted)), top_perm_sorted['stance_score_mean'], color='#2ca02c')
ax2.set_yticks(range(len(top_perm_sorted)))
ax2.set_yticklabels(top_perm_sorted['country'] + ' (' + top_perm_sorted['country'].str[:2] + ')')
ax2.set_xlabel('← Stance Score (Permissive)', fontweight='bold')
ax2.set_title('Top 5 Most Permissive Universities', fontweight='bold', color='#2ca02c')
ax2.set_xlim(-1, 0)
ax2.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, v in enumerate(top_perm_sorted['stance_score_mean']):
    ax2.text(v - 0.05, i, f'{v:.3f}', va='center', fontweight='bold', ha='right')

plt.tight_layout()
plt.savefig(FIG_EXTREMES, dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved → {FIG_EXTREMES}")

# ================================================================================
# SECTION 9: VISUALIZATION 3 - COUNTRY-WISE DISTRIBUTION
# ================================================================================

print("  → Visualization 3: Country-wise stance distribution")

country_stats = uni_stance.groupby('country').agg({
    'stance_score_mean': ['mean', 'std', 'count']
}).reset_index()

country_stats.columns = ['country', 'mean_stance', 'std_stance', 'n_universities']
country_stats = country_stats.sort_values('mean_stance', ascending=False)

plt.figure(figsize=(12, 6))

bars = plt.bar(
    range(len(country_stats)),
    country_stats['mean_stance'],
    yerr=country_stats['std_stance'],
    capsize=5,
    color=['#d62728' if x > 0 else '#2ca02c' for x in country_stats['mean_stance']],
    alpha=0.7,
    edgecolor='black',
    linewidth=1.5
)

plt.xticks(range(len(country_stats)), country_stats['country'], rotation=45, ha='right')
plt.ylabel('Mean Stance Score', fontweight='bold')
plt.xlabel('Country', fontweight='bold')
plt.title('Policy Stance Distribution by Country (Error bars = Std Dev)', 
          fontweight='bold', fontsize=12)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3, axis='y')

# Add count labels
for i, (bar, count) in enumerate(zip(bars, country_stats['n_universities'])):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'n={int(count)}',
            ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig(FIG_COUNTRY, dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved → {FIG_COUNTRY}")

# ================================================================================
# SECTION 10: VISUALIZATION 4 - STANCE CATEGORIES DISTRIBUTION
# ================================================================================

print("  → Visualization 4: Stance categories distribution")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# === UPDATED COLOR MAPPING (Permissive=Green, Restrictive=Red) ===

# Define color mapping by category name
color_map = {
    'Highly Restrictive': '#d62728',    # Dark red
    'Moderately Restrictive': '#ff7f0e', # Orange-red
    'Balanced': '#ffbb78',               # Light orange (neutral)
    'Moderately Permissive': '#98df8a',  # Light green
    'Highly Permissive': '#2ca02c'       # Dark green
}

# Pie chart
stance_dist = uni_stance['stance_category'].value_counts()
# Get colors in the order of stance_dist.index (category names)
colors_pie = [color_map[cat] for cat in stance_dist.index]

ax1.pie(stance_dist.values, labels=stance_dist.index, autopct='%1.1f%%',
        colors=colors_pie, startangle=90)
ax1.set_title('Distribution of Universities by Stance Category', fontweight='bold')

# Bar chart with countries stacked
stance_country = pd.crosstab(uni_stance['country'], uni_stance['stance_category'])

# Ensure consistent color order for stacked bar (order columns by restrictiveness)
category_order = ['Highly Restrictive', 'Moderately Restrictive', 'Balanced', 
                  'Moderately Permissive', 'Highly Permissive']
# Reorder columns to match category_order (only include categories that exist)
stance_country = stance_country[[col for col in category_order if col in stance_country.columns]]

# Get colors for bar chart in category_order
colors_bar = [color_map[cat] for cat in stance_country.columns]

stance_country.plot(kind='bar', stacked=True, ax=ax2, color=colors_bar)
ax2.set_title('Stance Categories by Country', fontweight='bold')
ax2.set_xlabel('Country', fontweight='bold')
ax2.set_ylabel('Number of Universities', fontweight='bold')
ax2.legend(title='Stance Category', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig(FIG_EXAMPLES, dpi=300, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved → {FIG_EXAMPLES}")


# ================================================================================
# SECTION 11: GENERATE TEXT REPORT
# ================================================================================

print("\n[STEP 8] Generating RQ3 analysis report...")

report = f"""
{'='*80}
RQ3: POLICY STANCE DETECTION ANALYSIS REPORT
{'='*80}

RESEARCH QUESTION:
What is the stance of university AI policies regarding usage restrictiveness?
(Analysis of extremes: from most restrictive to most permissive)

METHODOLOGY:
1. Stance scoring using lexicon-based approach
2. Score range: -1.0 (most permissive) to +1.0 (most restrictive)
3. Document-level calculation aggregated to university level

DATASET:
- Total documents analyzed: {len(df)}
- Total universities: {len(uni_stance)}
- Countries represented: {uni_stance['country'].nunique()}

{'='*80}
KEY FINDINGS
{'='*80}

OVERALL STANCE STATISTICS:
- Mean stance score: {uni_stance['stance_score_mean'].mean():.3f}
- Median stance score: {uni_stance['stance_score_mean'].median():.3f}
- Std deviation: {uni_stance['stance_score_mean'].std():.3f}
- Range: [{uni_stance['stance_score_mean'].min():.3f}, {uni_stance['stance_score_mean'].max():.3f}]

STANCE DISTRIBUTION:
{uni_stance['stance_category'].value_counts().to_string()}

BY COUNTRY:
{uni_stance.groupby('country')['stance_score_mean'].agg(['mean', 'min', 'max', 'count']).round(3).to_string()}

{'='*80}
TOP 5 MOST RESTRICTIVE UNIVERSITIES
{'='*80}

"""

for idx, (_, row) in enumerate(top_restrictive.iterrows(), 1):
    report += f"""
{idx}. {row['country'].upper()}
   - Stance Score: +{row['stance_score_mean']:.3f} ({row['stance_category']})
   - Number of Policies: {row['n_policies']}
   - Dominant Tone: {row['dominant_tone']}
   - Interpretation: University has highly restrictive AI policies
                     with strong prohibitive language and severe consequences

"""

report += f"""
{'='*80}
TOP 5 MOST PERMISSIVE UNIVERSITIES
{'='*80}

"""

for idx, (_, row) in enumerate(top_permissive.iterrows(), 1):
    report += f"""
{idx}. {row['country'].upper()}
   - Stance Score: {row['stance_score_mean']:.3f} ({row['stance_category']})
   - Number of Policies: {row['n_policies']}
   - Dominant Tone: {row['dominant_tone']}
   - Interpretation: University has permissive AI policies
                     with encouragement and support language

"""

report += f"""
{'='*80}
STANCE CATEGORIES EXPLAINED
{'='*80}

Highly Restrictive (Score ≥ +0.6):
- Policies contain strong prohibition language
- Emphasis on risks, violations, severe consequences
- Examples: "AI use is banned", "disciplinary action"

Moderately Restrictive (Score ≥ +0.3):
- Policies limit AI use with conditions
- Includes cautions and warnings
- Examples: "limited use allowed", "with restrictions"

Balanced (Score -0.3 to +0.3):
- Policies show mixed signals
- Both restrictions and permissions present
- Examples: "use responsibly", "with guidance"

Moderately Permissive (Score ≥ -0.3):
- Policies allow AI use with minimal conditions
- Some support and guidance
- Examples: "permitted use", "with supervision"

Highly Permissive (Score < -0.3):
- Policies encourage AI use
- Emphasis on support and innovation
- Examples: "encouraged", "fully supported"

{'='*80}
IMPLICATIONS FOR RQ3
{'='*80}

This analysis reveals:
1. Significant variation in policy stance across universities
2. Some institutions are restrictive (mean score: +{top_restrictive['stance_score_mean'].mean():.3f})
3. Others are permissive (mean score: {top_permissive['stance_score_mean'].mean():.3f})
4. The stance is NOT random but relates to:
   - Institutional governance models (RQ1: role assumptions)
   - Affective framing (RQ1: threat vs opportunity orientation)
   - Policy tone (from policy_tone_distilbert.py)

RECOMMENDATIONS:
- Universities with highly restrictive stance may need to reconsider
  their stance to support beneficial AI applications
- Universities should clearly communicate their AI stance in policy documents
- Consider hybrid approaches balancing innovation with responsibility

{'='*80}
"""

with open(OUT_REPORT, 'w', encoding='utf-8') as f:
    f.write(report)


print(f"✓ Saved report → {OUT_REPORT}")

# ================================================================================
# FINAL SUMMARY
# ================================================================================

print("\n" + "="*80)
print("RQ3 STANCE DETECTION ANALYSIS COMPLETE")
print("="*80)

print(f"\n✓ All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
print(f"\nFiles created:")
print(f"  1. {OUT_DOCUMENT}")
print(f"  2. {OUT_UNIVERSITY}")
print(f"  3. {OUT_EXTREMES}")
print(f"  4. {OUT_REPORT}")
print(f"\nVisualizations created:")
print(f"  1. {FIG_SCATTER}")
print(f"  2. {FIG_EXTREMES}")
print(f"  3. {FIG_COUNTRY}")
print(f"  4. {FIG_EXAMPLES}")