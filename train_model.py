"""
Fake News Detection - Model Training Script
---------------------------------------------
Dataset needed: "Fake and Real News Dataset" from Kaggle
Link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After downloading, you'll have two files: Fake.csv and True.csv
Place them in the same folder as this script before running.

Run with:  python train_model.py
"""

import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. LOAD DATA
# -----------------------------
print("Loading data...")
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0   # 0 = Fake
true_df["label"] = 1   # 1 = Real

df = pd.concat([fake_df, true_df], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Use title + text combined for stronger signal
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

print(f"Total articles: {len(df)}  |  Fake: {sum(df.label==0)}  Real: {sum(df.label==1)}")

# -----------------------------
# 2. CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text...")
df["content"] = df["content"].apply(clean_text)

# -----------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["content"], df["label"], test_size=0.2, random_state=42
)

# -----------------------------
# 4. TF-IDF VECTORIZATION
# -----------------------------
print("Vectorizing text (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -----------------------------
# 6. EVALUATE
# -----------------------------
y_pred = model.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc*100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# 7. SAVE MODEL + VECTORIZER
# -----------------------------
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nSaved model as 'fake_news_model.pkl' and vectorizer as 'tfidf_vectorizer.pkl'")
print("You can now run the app.py demo!")
