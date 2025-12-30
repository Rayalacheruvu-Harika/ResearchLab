import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/final_clean_dataset.csv")

print("\n" + "="*80)
print("VERIFICATION: NEGATIONS PRESERVED?")
print("="*80)

# List of negation words to check for
negations = ['not', 'no', 'never', 'cannot', 'neither', 'nor']

# Count documents containing each negation
print("\nNegation word frequency in cleaned data:")
print("-" * 80)

for neg in negations:
    count = df['clean_text'].str.contains(neg, case=False).sum()
    percentage = (count / len(df)) * 100
    print(f"'{neg:10}': {count:3} documents ({percentage:5.1f}%)")

# Show sample documents with negations
print("\n" + "="*80)
print("SAMPLE DOCUMENTS WITH NEGATIONS:")
print("="*80)

# Find documents that contain "not"
docs_with_not = df[df['clean_text'].str.contains('not', case=False)]

if len(docs_with_not) > 0:
    print(f"\nFound {len(docs_with_not)} documents containing 'not'\n")
    
    for idx, (i, row) in enumerate(docs_with_not.head(3).iterrows()):
        print(f"\nExample {idx+1}:")
        print(f"  URL: {row['url']}")
        print(f"  Original: {row['guideline_text'][:100]}...")
        print(f"  Cleaned:  {row['clean_text'][:100]}...")
        
        # Check if negation is present
        has_not = 'not' in row['clean_text'].lower()
        print(f"  Contains 'not': {'✅ YES' if has_not else '❌ NO'}")
else:
    print("\n⚠️  WARNING: No documents with 'not' found!")
    print("This might indicate the fix didn't work properly.")

# Statistics
print("\n" + "="*80)
print("OVERALL STATISTICS:")
print("="*80)

total_docs = len(df)
docs_with_any_negation = df['clean_text'].str.contains('|'.join(negations), case=False).sum()

print(f"\nTotal documents: {total_docs}")
print(f"Documents with negations: {docs_with_any_negation}")
print(f"Percentage: {(docs_with_any_negation/total_docs)*100:.1f}%")

if docs_with_any_negation > 0:
    print("\n✅ SUCCESS! Negations are being preserved!")
else:
    print("\n❌ PROBLEM! No negations found in cleaned text.")
    print("The fix may not have been applied correctly.")

print("\n" + "="*80)
