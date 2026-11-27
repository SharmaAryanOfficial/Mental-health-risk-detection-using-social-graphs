"""
graphsage.py -- GraphSAGE on SBERT text vectors + centrality features.

Run baseline_sbert.py and graph_baseline.py first (they save the features).

Usage:
    python graphsage.py                      # text + structure
    python graphsage.py --features text      # ablation: text only
    python graphsage.py --features struct    # ablation: structure only
    python graphsage.py --beta 2             # pick threshold by F2 (favours recall)
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, fbeta_score, precision_score,
                             recall_score, roc_auc_score)
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default=os.path.join("data", "mock"))
ap.add_argument("--features", choices=["all", "text", "struct"], default="all")
ap.add_argument("--beta", type=float, default=1.0)
args = ap.parse_args()


def path(name):
    return os.path.join(args.data_dir, name)


users = pd.read_csv(path("users.csv"))
posts = pd.read_csv(path("posts.csv"))
comments = pd.read_csv(path("comments.csv"))
ids = users["user_id"]
y_np = users["label"].to_numpy()
tr, va, te = ((users["split"] == s).to_numpy() for s in ("train", "val", "test"))

# 1. Node features, matched by user_id so rows cannot get mixed up.
emb = np.load(path("sbert_user_embeddings.npz"))
text = pd.DataFrame(emb["embeddings"], index=emb["user_id"]).loc[ids].to_numpy()
struct_df = pd.read_csv(path("graph_features.csv")).set_index("user_id").loc[ids]
struct_df[["in_degree", "out_degree"]] = np.log1p(
    struct_df[["in_degree", "out_degree"]])
struct = struct_df.to_numpy()

blocks = {"all": [text, struct], "text": [text], "struct": [struct]}
X = np.hstack(blocks[args.features])

# Standardise every column using TRAIN users only.
mean, std = X[tr].mean(axis=0), X[tr].std(axis=0)
std[std == 0] = 1
x = torch.tensor((X - mean) / std, dtype=torch.float)
y = torch.tensor(y_np, dtype=torch.long)
train_mask = torch.tensor(tr)

# 2. Edges: commenter <-> post owner, as row numbers.
edges = comments.merge(posts[["post_id", "user_id"]], on="post_id")
edges = edges[edges["commenter_id"] != edges["user_id"]]
row_of = {u: i for i, u in enumerate(ids)}
edge_index = torch.tensor(np.stack([
    edges["commenter_id"].map(row_of).to_numpy(),
    edges["user_id"].map(row_of).to_numpy(),
]), dtype=torch.long)
# Both directions, so each user learns from everyone they interact with.
edge_index = to_undirected(edge_index, num_nodes=len(users))


# 3. Two-layer GraphSAGE.
class SAGE(torch.nn.Module):
    def __init__(self, n_in, n_hidden=64):
        super().__init__()
        self.conv1 = SAGEConv(n_in, n_hidden)
        self.conv2 = SAGEConv(n_hidden, 2)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        return self.conv2(h, edge_index)


def run(seed):
    """Train once. Keep the epoch with the best VAL ROC-AUC (early stopping)."""
    torch.manual_seed(seed)
    model = SAGE(x.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    best_auc, best_prob, wait = -1.0, None, 0
    for _ in range(300):
        model.train()
        opt.zero_grad()
        out = model(x, edge_index)
        F.cross_entropy(out[train_mask], y[train_mask]).backward()  # train only
        opt.step()

        model.eval()
        with torch.no_grad():
            prob = model(x, edge_index).softmax(dim=1)[:, 1].numpy()
        auc = roc_auc_score(y_np[va], prob[va])
        if auc > best_auc:
            best_auc, best_prob, wait = auc, prob, 0
        else:
            wait += 1
            if wait == 30:
                break

    # Decision threshold chosen on VAL.
    t = max(np.arange(0.20, 0.81, 0.05),
            key=lambda t: fbeta_score(y_np[va], best_prob[va] >= t,
                                      beta=args.beta))
    pred = (best_prob[te] >= t).astype(int)
    truth = y_np[te]
    return {
        "accuracy": accuracy_score(truth, pred),
        "precision": precision_score(truth, pred, zero_division=0),
        "recall": recall_score(truth, pred),
        "F1": fbeta_score(truth, pred, beta=1),
        "ROC-AUC": roc_auc_score(truth, best_prob[te]),
    }


# 4. Neural networks vary between runs, so report 5 seeds.
results = pd.DataFrame([run(seed) for seed in range(5)])
print(f"features: {args.features}, threshold rule: F{args.beta:g}")
print("\nTEST RESULTS (mean +/- std over 5 seeds)")
for name in results.columns:
    print(f"{name:<10} {results[name].mean():.3f} +/- {results[name].std():.3f}")