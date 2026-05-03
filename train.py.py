"""
Email Spam Classification
=========================
Classifies emails/SMS as spam or ham using:
- TF-IDF Vectorization
- Naive Bayes (MultinomialNB)
- Logistic Regression

Author: Mohit Masavarapu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

# ── 1. LOAD & EXPLORE DATA ──────────────────────────────────────────────────

print("=" * 60)
print("  EMAIL SPAM CLASSIFICATION")
print("=" * 60)

df = pd.read_csv("data/spam.csv")
print(f"\n[1] Dataset loaded: {df.shape[0]} samples")
print(f"    Ham  : {(df['label']=='ham').sum()}")
print(f"    Spam : {(df['label']=='spam').sum()}")

# ── 2. DATA CLEANING ─────────────────────────────────────────────────────────

df = df.dropna(subset=["message"])
df["message"] = df["message"].str.strip()
df["message_length"] = df["message"].str.len()
df["word_count"] = df["message"].str.split().str.len()
df["label_encoded"] = df["label"].map({"ham": 0, "spam": 1})

print(f"\n[2] After cleaning: {df.shape[0]} samples")

# ── 3. FEATURE EXTRACTION (TF-IDF) ──────────────────────────────────────────

X = df["message"]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\n[3] TF-IDF feature matrix: {X_train_tfidf.shape}")

# ── 4. TRAIN MODELS ──────────────────────────────────────────────────────────

print("\n[4] Training models...")

# Naive Bayes
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_acc  = accuracy_score(y_test, nb_pred)

# Logistic Regression
lr_model = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_pred = lr_model.predict(X_test_tfidf)
lr_acc  = accuracy_score(y_test, lr_pred)

print(f"    Naive Bayes Accuracy       : {nb_acc:.4f} ({nb_acc*100:.2f}%)")
print(f"    Logistic Regression Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")

# ── 5. HYPERPARAMETER TUNING ─────────────────────────────────────────────────

print("\n[5] Hyperparameter tuning (GridSearchCV)...")

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", sublinear_tf=True)),
    ("clf",   LogisticRegression(max_iter=1000, random_state=42))
])

param_grid = {
    "tfidf__max_features": [3000, 5000],
    "tfidf__ngram_range":  [(1, 1), (1, 2)],
    "clf__C":              [1.0, 5.0, 10.0],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_pred  = best_model.predict(X_test)
best_acc   = accuracy_score(y_test, best_pred)

print(f"    Best Params : {grid_search.best_params_}")
print(f"    Best CV Score: {grid_search.best_score_:.4f}")
print(f"    Test Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

# ── 6. EVALUATION ────────────────────────────────────────────────────────────

print("\n[6] Classification Report (Best Model):")
print(classification_report(y_test, best_pred, target_names=["Ham", "Spam"]))

cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
print(f"    5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 7. SAVE PLOTS ─────────────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Email Spam Classifier — Model Evaluation", fontsize=15, fontweight="bold")

# Confusion Matrix
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=axes[0])
axes[0].set_title("Confusion Matrix")
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# Model Accuracy Comparison
models  = ["Naive Bayes", "Logistic Reg", "Tuned LR"]
accs    = [nb_acc, lr_acc, best_acc]
colors  = ["#4C72B0", "#DD8452", "#55A868"]
bars = axes[1].bar(models, [a * 100 for a in accs], color=colors, edgecolor="black", width=0.5)
axes[1].set_ylim(80, 100)
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_title("Model Accuracy Comparison")
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{acc*100:.2f}%", ha="center", va="bottom", fontweight="bold")

# ROC Curve
y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
axes[2].plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {auc:.4f}")
axes[2].plot([0, 1], [0, 1], "k--", lw=1)
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].set_title("ROC Curve")
axes[2].legend(loc="lower right")

plt.tight_layout()
plt.savefig("models/evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[7] Evaluation plot saved → models/evaluation.png")

# ── 8. SAVE MODEL ────────────────────────────────────────────────────────────

with open("models/spam_classifier.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("[8] Model saved → models/spam_classifier.pkl")

print("\n" + "=" * 60)
print(f"  FINAL ACCURACY: {best_acc*100:.2f}%  |  AUC: {auc:.4f}")
print("=" * 60)
