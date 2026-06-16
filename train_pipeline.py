"""
Disease Prediction using Logistic Regression
Full training pipeline: EDA -> preprocessing -> model training -> evaluation -> save artifacts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

ASSETS = "assets"
MODELS = "models"
os.makedirs(ASSETS, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# -----------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------
df = pd.read_csv("data/Blood_samples_dataset_balanced_2_f_.csv")
print("Shape:", df.shape)
print(df.head())

FEATURE_COLS = [c for c in df.columns if c != "Disease"]
TARGET_COL = "Disease"

# -----------------------------------------------------------------
# 2. EDA
# -----------------------------------------------------------------
# Class distribution
plt.figure(figsize=(7, 4))
order = df[TARGET_COL].value_counts().index
sns.countplot(data=df, x=TARGET_COL, order=order, hue=TARGET_COL, palette="viridis", legend=False)
plt.title("Disease Class Distribution")
plt.xlabel("")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{ASSETS}/class_distribution.png")
plt.close()

# Correlation heatmap of features
plt.figure(figsize=(13, 10))
corr = df[FEATURE_COLS].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.3,
            cbar_kws={"shrink": 0.7})
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{ASSETS}/correlation_heatmap.png")
plt.close()

# Distribution of a few key features by disease
key_feats = ["Glucose", "Hemoglobin", "HbA1c", "Platelets"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, feat in zip(axes.flat, key_feats):
    sns.boxplot(data=df, x=TARGET_COL, y=feat, hue=TARGET_COL, ax=ax, palette="viridis", legend=False)
    ax.set_title(f"{feat} by Disease")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(f"{ASSETS}/feature_distributions.png")
plt.close()

# -----------------------------------------------------------------
# 3. Preprocessing
# -----------------------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL])
X = df[FEATURE_COLS].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# 4. Train Logistic Regression (multinomial)
# -----------------------------------------------------------------
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs",
    C=1.0,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# -----------------------------------------------------------------
# 5. Evaluation
# -----------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_text = classification_report(y_test, y_pred, target_names=le.classes_)
print(f"\nTest Accuracy: {acc:.4f}\n")
print(report_text)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Confusion Matrix (Accuracy = {acc:.2%})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{ASSETS}/confusion_matrix.png")
plt.close()

# ROC curves (one-vs-rest, multiclass)
y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))
plt.figure(figsize=(7, 6))
for i, cls in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(f"{ASSETS}/roc_curves.png")
plt.close()

# Feature importance (coefficient magnitude, averaged across classes)
coef_df = pd.DataFrame(model.coef_, columns=FEATURE_COLS, index=le.classes_)
importance = coef_df.abs().mean(axis=0).sort_values(ascending=False)
plt.figure(figsize=(8, 9))
sns.barplot(x=importance.values, y=importance.index, hue=importance.index, palette="viridis", legend=False)
plt.title("Feature Importance (Mean |Coefficient| Across Classes)")
plt.xlabel("Mean Absolute Coefficient")
plt.tight_layout()
plt.savefig(f"{ASSETS}/feature_importance.png")
plt.close()

# -----------------------------------------------------------------
# 6. Save artifacts for the dashboard
# -----------------------------------------------------------------
joblib.dump(model, f"{MODELS}/logistic_regression_model.pkl")
joblib.dump(scaler, f"{MODELS}/scaler.pkl")
joblib.dump(le, f"{MODELS}/label_encoder.pkl")
joblib.dump(FEATURE_COLS, f"{MODELS}/feature_names.pkl")

metrics = {
    "accuracy": acc,
    "classification_report": report,
    "classes": list(le.classes_),
    "feature_importance": importance.to_dict(),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
with open(f"{MODELS}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save feature stats (for dashboard slider ranges/defaults)
feature_stats = df[FEATURE_COLS].describe().T[["min", "max", "mean"]].to_dict(orient="index")
with open(f"{MODELS}/feature_stats.json", "w") as f:
    json.dump(feature_stats, f, indent=2)

print("\nArtifacts saved to models/ and assets/")
