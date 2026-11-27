"""
graph_baseline.py -- structure-only baseline: centralities + logistic regression.

Usage:
    python graph_baseline.py
    python graph_baseline.py --data_dir data/mock
"""
import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=os.path.join("data", "mock"))
args = ap.parse_args()

users = pd.read_csv(os.path.join(args.data_dir, "users.csv"))
posts = pd.read_csv(os.path.join(args.data_dir, "posts.csv"))
comments = pd.read_csv(os.path.join(args.data_dir, "comments.csv"))

# 1. Build the graph: commenter -> post owner, weight = number of comments.
edges = comments.merge(posts[["post_id", "user_id"]], on="post_id")
edges = edges[edges["commenter_id"] != edges["user_id"]]
edges = (edges.groupby(["commenter_id", "user_id"]).size()
         .reset_index(name="weight"))

G = nx.DiGraph()
G.add_nodes_from(users["user_id"])          # keeps users with no connections
G.add_weighted_edges_from(edges.itertuples(index=False, name=None))
print(f"nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}, "
      f"isolated users: {nx.number_of_isolates(G)}")

# 2. Centrality features (labels are never used here).
print("computing centralities (betweenness can take a minute)...")
features = {
    "in_degree": dict(G.in_degree()),       # distinct users who reply to you
    "out_degree": dict(G.out_degree()),     # distinct users you reply to
    # Unweighted: NetworkX would read a weight as a distance, not a strength.
    "betweenness": nx.betweenness_centrality(G),
    "eigenvector": nx.eigenvector_centrality(G, max_iter=1000, weight="weight"),
}
feat = pd.DataFrame({name: users["user_id"].map(values)
                     for name, values in features.items()})

out = os.path.join(args.data_dir, "graph_features.csv")
pd.concat([users[["user_id"]], feat], axis=1).to_csv(out, index=False)
print(f"saved features to {out}")

print("\nAverage feature value per class (train users only):")
is_train = users["split"] == "train"
print(feat[is_train].groupby(users.loc[is_train, "label"]).mean().T)

# Degree counts are skewed, so compress them with log(1 + x).
X = feat.copy()
X[["in_degree", "out_degree"]] = np.log1p(X[["in_degree", "out_degree"]])
X = X.to_numpy()
y = users["label"].to_numpy()
tr, va, te = ((users["split"] == s).to_numpy() for s in ("train", "val", "test"))

# 3. Tune C and threshold on VAL (best F1). Scaling is fitted on train only.
best = None
for C in (0.01, 0.1, 1, 10, 100):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=2000))
    clf.fit(X[tr], y[tr])
    prob = clf.predict_proba(X[va])[:, 1]
    for t in np.arange(0.20, 0.81, 0.05):
        score = f1_score(y[va], (prob >= t).astype(int))
        if best is None or score > best[0]:
            best = (score, C, t, clf)

val_f1, C, threshold, clf = best
print(f"\nbest C={C}, threshold={threshold:.2f}, val F1={val_f1:.3f}")

# 4. Final score on TEST.
prob = clf.predict_proba(X[te])[:, 1]
pred = (prob >= threshold).astype(int)
print("\nTEST RESULTS")
print(f"accuracy: {accuracy_score(y[te], pred):.3f}")
print(f"F1:       {f1_score(y[te], pred):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y[te], prob):.3f}")
print("confusion matrix (rows = true, columns = predicted; 0, 1):")
print(confusion_matrix(y[te], pred))

print("\nFeature weights (positive = pushes towards at-risk):")
for name, w in zip(feat.columns, clf[-1].coef_[0]):
    print(f"  {name:<12} {w:+.3f}")