import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import os

# -----------------------------------
# File path
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_PATH = os.path.join(BASE_DIR, "annotations.csv")

# -----------------------------------
# Load dataset
# -----------------------------------
df = pd.read_csv(FILE_PATH)

# Human annotators
human_cols = ["Harika", "Momena", "Siya", "Sreshta", "Suraya"]

# Model prediction column
model_col = "Model Annotation"

# -----------------------------------
# Clean labels
# -----------------------------------
for col in human_cols + [model_col]:
    df[col] = df[col].astype(str).str.strip()

# -----------------------------------
# Majority vote from humans
# -----------------------------------
df["human_majority"] = df[human_cols].mode(axis=1)[0]

y_true = df["human_majority"]
y_pred = df[model_col]

# -----------------------------------
# Get all class labels
# -----------------------------------
labels = sorted(list(set(y_true) | set(y_pred)))

# -----------------------------------
# Confusion Matrix
# -----------------------------------
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print("\nConfusion Matrix:\n")
print(cm_df)

# -----------------------------------
# Accuracy
# -----------------------------------
accuracy = np.trace(cm) / np.sum(cm)

# -----------------------------------
# Precision, Recall, F1 per class
# -----------------------------------
precision_list = []
recall_list = []
f1_list = []

for i in range(len(labels)):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP

    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

# Weighted averages
weights = cm.sum(axis=1) / cm.sum()

precision = np.sum(np.array(precision_list) * weights)
recall = np.sum(np.array(recall_list) * weights)
f1 = np.sum(np.array(f1_list) * weights)

# -----------------------------------
# Cohen's Kappa
# -----------------------------------
kappa = cohen_kappa_score(y_true, y_pred)

# -----------------------------------
# Print metrics
# -----------------------------------
print("\nEvaluation Metrics:\n")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"Cohen's Kappa  : {kappa:.4f}")