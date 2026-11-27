"""
baseline_tfidf.py -- text-only baseline: TF-IDF + logistic regression.

Usage:
    python baseline_tfidf.py
    python baseline_tfidf.py --data_dir data/mock
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=os.path.join("data", "mock"))
args = ap.parse_args()

# 1. Load the data and join all posts of each user into one document.
users = pd.read_csv(os.path.join(args.data_dir, "users.csv"))
posts = pd.read_csv(os.path.join(args.data_dir, "posts.csv"))
docs = posts.groupby("user_id")["text"].apply(lambda s: " ".join(s.dropna()))
users["text"] = users["user_id"].map(docs).fillna("")

train, val, test = (users[users["split"] == s] for s in ("train", "val", "test"))

# 2. TF-IDF: learn the vocabulary from TRAIN users only (avoids leakage).
#    Stop words are kept so that phrases like "not sad" survive.
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                      token_pattern=r"(?u)\b\w[\w']+\b")
X_train = vec.fit_transform(train["text"])
X_val = vec.transform(val["text"])
X_test = vec.transform(test["text"])

# 3. Tune regularisation strength C and decision threshold on VAL (best F1).
best = None
for C in (0.01, 0.1, 1, 10):
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, train["label"])
    prob = model.predict_proba(X_val)[:, 1]
    for t in np.arange(0.20, 0.81, 0.05):
        score = f1_score(val["label"], (prob >= t).astype(int))
        if best is None or score > best[0]:
            best = (score, C, t, model)

val_f1, C, threshold, model = best
print(f"best C={C}, threshold={threshold:.2f}, val F1={val_f1:.3f}")

# 4. Final score on TEST.
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= threshold).astype(int)
print("\nTEST RESULTS")
print(f"accuracy: {accuracy_score(test['label'], pred):.3f}")
print(f"F1:       {f1_score(test['label'], pred):.3f}")
print(f"ROC-AUC:  {roc_auc_score(test['label'], prob):.3f}")
print("confusion matrix (rows = true, columns = predicted; 0, 1):")
print(confusion_matrix(test["label"], pred))

# 5. What the model learned: highest and lowest weighted terms.
terms = vec.get_feature_names_out()
weights = model.coef_[0]
order = np.argsort(weights)
for title, idx in (("at-risk", order[::-1][:15]), ("control", order[:15])):
    print(f"\nTop terms for {title}:")
    for i in idx:
        print(f"  {terms[i]:<25} {weights[i]:+.3f}")
