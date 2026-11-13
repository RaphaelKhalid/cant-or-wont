"""Second computation of the TF-IDF text rival (words, what the model wrote). sklearn's Ridge in place of the hand-written
dual solve, task dummy variables in place of demeaning, and its own loading code.

    python audit_text_rival.py
"""
import hashlib
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    replies = rows(os.path.join(HERE, "data", "replies.jsonl"))
    doc = {}
    for p in (os.path.join(HERE, "data", "reply_texts.jsonl"),):
        for r in rows(p):
            if "reply" in r:
                doc[r["id"]] = (r.get("explanation") or "") + "\n" + (r.get("reply") or "")
    docs = [doc[r["id"]] for r in replies]
    on = np.array([float(r["cue"] == "on") for r in replies])
    fail = np.array([float(r["outcome"] == "fail") for r in replies])
    y = on * fail
    fold = np.array([int(hashlib.sha256(("row1d|fold|" + r["task"]).encode()).hexdigest()[:8], 16) % 5 for r in replies])
    score = np.zeros(len(replies))
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
        Xtr = vec.fit_transform([docs[i] for i in tr]).toarray()
        Xte = vec.transform([docs[i] for i in te]).toarray()
        mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
        model = Ridge(alpha=Xtr.shape[1], fit_intercept=True).fit((Xtr - mu) / sd, y[tr])
        ref = model.predict((Xtr - mu) / sd)[y[tr] == 0]
        score[te] = (model.predict((Xte - mu) / sd) - ref.mean()) / ref.std()
    z = score / score[fail == 0].std()
    tasks = sorted({r["task"] for r in replies})
    col = {t: i for i, t in enumerate(tasks)}
    tix = np.array([col[r["task"]] for r in replies])
    X = np.zeros((len(replies), 3 + len(tasks)))
    X[:, 0], X[:, 1], X[:, 2] = on, fail, y
    X[np.arange(len(replies)), 3 + tix] = 1.0
    cells = [(on == a) & (fail == b) for a in (0, 1) for b in (0, 1)]

    def fit(w):
        keep = w > 0
        s = np.sqrt(w[keep])
        used = np.concatenate([[True, True, True], X[keep, 3:].sum(0) > 0])
        return float(np.linalg.lstsq(X[keep][:, used] * s[:, None], z[keep] * s, rcond=None)[0][2])

    rng = np.random.default_rng(20261112)
    draws = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks)).astype(float)[tix]
        if any(w[c].sum() == 0 for c in cells):
            continue
        draws.append(fit(w))
    out = {"gamma": fit(np.ones(len(replies))), "interval95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    first = json.load(open(os.path.join(HERE, "results", "text_rival.json"), encoding="utf-8"))["arms"]["tfidf_words_written"]
    out["first_computation"] = {"gamma": first["gamma"], "interval95": first["interval95"]}
    out["largest_difference"] = float(max(abs(out["gamma"] - first["gamma"]), *np.abs(np.array(out["interval95"]) - np.array(first["interval95"]))))
    with open(os.path.join(HERE, "results", "text_rival_audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
