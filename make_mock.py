"""
make_mock.py -- generate a SYNTHETIC Reddit-style dataset.

Purpose: let the whole pipeline (graph building, text baseline, graph baseline,
GraphSAGE) run end to end before real data is ready. Results on this data only
show that the CODE works. They say nothing about real people.

Output (in --out_dir, default data/mock):
    users.csv     one row per user:    user_id, label, split
    posts.csv     one row per post:    post_id, user_id, subreddit, text
    comments.csv  one row per comment: comment_id, commenter_id, post_id

The real-data pipeline will produce the same three files, so every later script
works on either dataset without changes.

Usage:
    python make_mock.py
    python make_mock.py --n_users 3000 --seed 7
"""
import argparse
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Text building blocks. Sentences are mild, everyday descriptions of low mood
# (sleep, energy, appetite, focus, interest, self-worth). Nothing about
# self-harm is generated on purpose.
# --------------------------------------------------------------------------- #
SYMPTOM = [
    "I have been feeling really down for weeks and nothing helps.",
    "I can't sleep at night and I am exhausted all day.",
    "No appetite lately, I barely eat anything.",
    "I can't focus on work or studies, my mind just drifts.",
    "I used to enjoy things but now I feel empty and numb.",
    "I feel worthless and like a failure to my family.",
    "I am so tired all the time even after sleeping.",
    "Everything feels pointless and I have lost interest in my hobbies.",
    "I feel alone even when I am around people.",
    "I keep crying for no reason and I don't know why.",
    "I feel hopeless about the future.",
    "I am restless and irritable, small things set me off.",
]
NEUTRAL = [
    "Watched a great movie over the weekend with friends.",
    "Anyone have tips for a beginner workout routine?",
    "Just finished a long coding session and grabbed some coffee.",
    "Trying to save money for a new laptop this semester.",
    "The new game update is surprisingly good.",
    "Had a busy week at work but things are going fine.",
    "Looking for book recommendations for the holidays.",
    "Went for a run this morning and it felt great.",
    "Does anyone know a good recipe for dinner tonight?",
    "Planning a trip with friends next month, any suggestions?",
    "Exams are coming up so I have been studying a lot.",
    "My team won the match yesterday, what a game!",
]
# Sentences that BOTH groups use equally often -> makes the task less trivial.
AMBIGUOUS = [
    "Didn't sleep well last night, need more coffee.",
    "Feeling a bit tired after the long week.",
    "Not sure what to do this weekend.",
    "Work has been stressful lately.",
    "Been thinking a lot about the future.",
    "Struggling to stay focused today.",
]

SUPPORT_SUBS = ["depression", "anxiety", "lonely", "selfimprovement"]
GENERAL_SUBS = ["CasualConversation", "gaming", "fitness", "movies",
                "personalfinance", "AskReddit"]


def make_users(n_users, rng):
    """Labels (1 = at-risk, 0 = control) and a fixed train/val/test split."""
    n_risk = int(n_users * 0.40)
    labels = np.array([1] * n_risk + [0] * (n_users - n_risk))
    rng.shuffle(labels)

    # Stratified split: 70% train / 15% val / 15% test inside each class.
    split = np.empty(n_users, dtype=object)
    for lab in (0, 1):
        idx = np.where(labels == lab)[0]
        rng.shuffle(idx)
        n_tr, n_va = int(0.70 * len(idx)), int(0.15 * len(idx))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:]] = "test"

    return pd.DataFrame({
        "user_id": [f"user_{i:04d}" for i in range(n_users)],
        "label": labels,
        "split": split,
    })


def make_posts(users, rng):
    """Each user writes 1+ posts. At-risk users use symptom sentences more often."""
    rows = []
    pid = 0
    for uid, lab in zip(users["user_id"], users["label"]):
        # Writing style: how often a sentence is a "symptom" sentence.
        # ~10% of users are "hard cases": at-risk users who write like controls
        # and controls who write like at-risk users.
        hard_case = rng.random() < 0.10
        use_risk_style = (lab == 1) != hard_case
        p_symptom = rng.beta(3, 5) if use_risk_style else rng.beta(1.2, 10)

        n_posts = 1 + rng.poisson(5)
        for _ in range(n_posts):
            # At-risk users post in support subreddits more often (metadata only!)
            in_support = rng.random() < (0.55 if lab == 1 else 0.12)
            sub = rng.choice(SUPPORT_SUBS if in_support else GENERAL_SUBS)
            sentences = []
            for _ in range(rng.integers(2, 5)):
                r = rng.random()
                if r < p_symptom:
                    sentences.append(rng.choice(SYMPTOM))
                elif r < p_symptom + 0.20:
                    sentences.append(rng.choice(AMBIGUOUS))
                else:
                    sentences.append(rng.choice(NEUTRAL))
            rows.append((f"post_{pid:05d}", uid, sub, " ".join(sentences)))
            pid += 1
    return pd.DataFrame(rows, columns=["post_id", "user_id", "subreddit", "text"])


def make_comments(users, posts, rng):
    """
    Reply graph signal (deliberately weak and noisy):
      - posts by at-risk users receive fewer comments
      - at-risk users comment a bit less overall
      - at-risk users are somewhat more likely to reply to other at-risk users
    """
    uid = users["user_id"].to_numpy()
    lab = users["label"].to_numpy()
    n = len(uid)
    label_of = dict(zip(uid, lab))

    # How "chatty" each user is (right-skewed, like real forums).
    chatty = rng.lognormal(mean=0.0, sigma=0.8, size=n)
    chatty = chatty * np.where(lab == 1, 0.75, 1.0)

    # Who comments on a CONTROL user's post / on an AT-RISK user's post.
    p_on_control = chatty / chatty.sum()
    boosted = chatty * np.where(lab == 1, 2.2, 1.0)
    p_on_risk = boosted / boosted.sum()

    rows = []
    cid = 0
    for post_id, owner in zip(posts["post_id"], posts["user_id"]):
        owner_label = label_of[owner]
        lam = 1.0 if owner_label == 1 else 1.6
        k = rng.poisson(lam)
        if k == 0:
            continue
        probs = p_on_risk if owner_label == 1 else p_on_control
        commenters = rng.choice(uid, size=k, p=probs)
        for c in commenters:
            if c == owner:          # no replying to yourself
                continue
            rows.append((f"comment_{cid:05d}", c, post_id))
            cid += 1
    return pd.DataFrame(rows, columns=["comment_id", "commenter_id", "post_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_users", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=os.path.join("data", "mock"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    users = make_users(args.n_users, rng)
    posts = make_posts(users, rng)
    comments = make_comments(users, posts, rng)

    os.makedirs(args.out_dir, exist_ok=True)
    users.to_csv(os.path.join(args.out_dir, "users.csv"), index=False)
    posts.to_csv(os.path.join(args.out_dir, "posts.csv"), index=False)
    comments.to_csv(os.path.join(args.out_dir, "comments.csv"), index=False)

    print(f"users:    {len(users):>6}  (at-risk: {int(users['label'].sum())}, "
          f"control: {int((users['label'] == 0).sum())})")
    print(f"posts:    {len(posts):>6}")
    print(f"comments: {len(comments):>6}")
    print("split sizes:", users["split"].value_counts().to_dict())
    print(f"saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
