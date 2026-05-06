# 📧 Email Spam Classification

A machine learning project that classifies emails and SMS messages as **spam** or **ham (not spam)** using Natural Language Processing techniques.

**Achieved ~98% accuracy** using TF-IDF vectorization with Logistic Regression, tuned via GridSearchCV.

## 📊 Results

![Evaluation](models/evaluation.png)
---

## 🚀 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Library | Scikit-learn |
| Feature Extraction | TF-IDF Vectorizer |
| Models | Naive Bayes, Logistic Regression |
| Tuning | GridSearchCV (5-Fold CV) |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## 📁 Project Structure

```
email-spam-classifier/
│
├── data/
│   ├── spam.csv              # Dataset (ham/spam messages)
│   └── generate_data.py      # Dataset generation script
│
├── models/
│   ├── spam_classifier.pkl   # Saved trained model
│   └── evaluation.png        # Confusion matrix + ROC curve plots
│
├── notebooks/
│   └── spam_classification.ipynb  # Full EDA + training walkthrough
│
├── train.py                  # End-to-end training pipeline
├── predict.py                # Inference script
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/mohitmasavarapu19/email-spam-classifier.git
cd email-spam-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```

### 4. Run predictions
```bash
# Demo predictions
python predict.py

# Classify a specific message
python predict.py --message "WINNER! You've been selected for a £900 prize!"
```

### 5. Explore the notebook
```bash
jupyter notebook notebooks/spam_classification.ipynb
```

---

## 🧠 ML Pipeline

```
Raw Text
   ↓
Data Cleaning  (strip whitespace, remove nulls)
   ↓
TF-IDF Vectorization  (max 5000 features, bigrams, sublinear TF)
   ↓
Model Training  (Naive Bayes + Logistic Regression)
   ↓
Hyperparameter Tuning  (GridSearchCV, 5-Fold CV)
   ↓
Evaluation  (Accuracy, Precision, Recall, F1, ROC-AUC)
   ↓
Saved Model (.pkl)
```

---

## 📊 Results

| Model | Accuracy |
|-------|----------|
| Naive Bayes | 98.85% |
| Logistic Regression | 98.85% |
| **Tuned LR (GridSearchCV)** | **~98%** |

**Best hyperparameters:**
- `tfidf__max_features`: 3000
- `tfidf__ngram_range`: (1, 1)
- `clf__C`: 5.0

**5-Fold Cross-Validation:** 98.57% ± 1.03%

---

## 🔍 How TF-IDF Works

**TF-IDF (Term Frequency–Inverse Document Frequency)** converts text into numerical feature vectors:

- **TF** — How often a word appears in a message
- **IDF** — How rare the word is across all messages
- Words like *"FREE"*, *"WIN"*, *"CLAIM"*, *"PRIZE"* are common in spam but rare in ham → high TF-IDF score → strong spam signal

---

## 📈 Evaluation Plots

Model evaluation charts are saved to `models/evaluation.png`:
- Confusion Matrix
- Model Accuracy Comparison
- ROC Curve (AUC score)

---

## 💡 Key Learnings

- TF-IDF with bigrams captures phrase-level spam signals (e.g., *"call now"*, *"free prize"*)
- `sublinear_tf=True` reduces the weight of very frequent words, improving generalization
- Logistic Regression outperforms Naive Bayes slightly on this dataset due to feature correlation handling
- GridSearchCV prevents overfitting by selecting hyperparameters on held-out CV folds

---

*Built as part of a machine learning portfolio project.*
