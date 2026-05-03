# 📧 Email Spam Classification

A machine learning project that classifies emails and SMS messages as **spam** or **ham (not spam)** using Natural Language Processing techniques.

🚀 Achieved **~98% accuracy** using TF-IDF vectorization with Logistic Regression, tuned via GridSearchCV.

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


email-spam-classifier/
│
├── data/
│ ├── spam.csv
│ └── generate_data.py
│
├── models/
│ ├── spam_classifier.pkl
│ └── evaluation.png
│
├── notebooks/
│ └── spam_classification.ipynb
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md


---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/mohitmasavarapu19/email-spam-classifier.git
cd email-spam-classifier
2. Install dependencies
pip install -r requirements.txt
3. Generate dataset
python data/generate_data.py
4. Train the model
python train.py
5. Run predictions
python predict.py
6. Test custom message
python predict.py --message "WINNER! You got a prize!"
🧠 ML Pipeline
Raw Text
   ↓
Data Cleaning
   ↓
TF-IDF Vectorization
   ↓
Model Training (NB + Logistic Regression)
   ↓
Hyperparameter Tuning (GridSearchCV)
   ↓
Evaluation (Accuracy, ROC-AUC)
   ↓
Saved Model (.pkl)
📊 Results
Accuracy: ~98%
Best Model: Logistic Regression
5-Fold CV: 98.57% ± 1.03%
📈 Evaluation

Model evaluation includes:

Confusion Matrix
ROC Curve
Accuracy comparison

Saved at:

models/evaluation.png
🚀 Demo

Example:

Input:

WINNER! You have been selected for a prize!

Output:

🚫 SPAM (confidence: ~99%)
💡 Key Learnings
TF-IDF highlights important spam keywords like free, win, claim
Logistic Regression handles feature correlation better than Naive Bayes
GridSearchCV improves model performance and avoids overfitting
📄 License

MIT License

👨‍💻 Author

Mohit Masavarapu
https://github.com/mohitmasavarapu19
