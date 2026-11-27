"""
baseline_sbert.py -- text model: SBERT embeddings + logistic regression.

Usage:
    python baseline_sbert.py
    python baseline_sbert.py --data_dir data/mock
"""
import argparse
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=os.path.join("data", "mock"))
args = ap.parse_args()

users = pd.read_csv(os.path.join(args.data_dir, "users.csv"))
posts = pd.read_csv(os.path.join(args.data_dir, "posts.csv"))
posts = posts.dropna(subset=["text"]).reset_index(drop=True)

# 1. Embed every post separately (avoids cutting off long user histories).
model = SentenceTransformer("all-MiniLM-L6-v2")
post_vecs = model.encode(posts["text"].tolist(), batch_size=64,
                         show_progress_bar=True)

# 2. Average the post vectors of each user. Row i matches users row i.
#    Users with no posts keep a vector of zeros.
X = np.zeros((len(users), post_vecs.shape[1]), dtype=np.float32)
rows_of = posts.groupby("user_id").indices
for i, uid in enumerate(users["user_id"]):
    if uid in rows_of:
        X[i] = post_vecs[rows_of[uid]].mean(axis=0)

# Save the vectors: GraphSAGE will reuse them as text node features.
out = os.path.join(args.data_dir, "sbert_user_embeddings.npz")
np.savez(out, user_id=users["user_id"].to_numpy(dtype=str), embeddings=X)
print(f"saved {X.shape} embeddings to {out}")

y = users["label"].to_numpy()
tr, va, te = ((users["split"] == s).to_numpy() for s in ("train", "val", "test"))

# 3. Tune regularisation strength C and decision threshold on VAL (best F1).
best = None
for C in (0.01, 0.1, 1, 10, 100):
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(X[tr], y[tr])
    prob = clf.predict_proba(X[va])[:, 1]
    for t in np.arange(0.20, 0.81, 0.05):
        score = f1_score(y[va], (prob >= t).astype(int))
        if best is None or score > best[0]:
            best = (score, C, t, clf)

val_f1, C, threshold, clf = best
print(f"best C={C}, threshold={threshold:.2f}, val F1={val_f1:.3f}")

# 4. Final score on TEST.
prob = clf.predict_proba(X[te])[:, 1]
pred = (prob >= threshold).astype(int)
print("\nTEST RESULTS")
print(f"accuracy: {accuracy_score(y[te], pred):.3f}")
print(f"F1:       {f1_score(y[te], pred):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y[te], prob):.3f}")
print("confusion matrix (rows = true, columns = predicted; 0, 1):")
print(confusion_matrix(y[te], pred))