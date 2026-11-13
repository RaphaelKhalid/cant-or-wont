"""Post hoc, not in PREREG.md. The rival's best shot.

The ridge probe was trained on this run's labels. A text model gets the same deal here: the same
labels, the same task folds, the same ridge, on TF-IDF features of what the model wrote. If the text
model shows the same difference of gaps, the probe is not reading anything text does not hold.

    python posthoc_text_rival.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_row1d as ar  # noqa: E402
from lib import auroc, flags_loto, read_jsonl  # noqa: E402


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer
    replies = read_jsonl(os.path.join(HERE, "data", "replies.jsonl"))
    text = {}
    for path in (os.path.join(HERE, "data", "reply_texts.jsonl"),):
        for r in read_jsonl(path):
            if "reply" in r:
                text[r["id"]] = {"written": (r.get("explanation") or "") + "\n" + (r.get("reply") or ""), "prompt": r.get("user") or ""}
    assert all(r["id"] in text for r in replies)
    on = np.array([1.0 if r["cue"] == "on" else 0.0 for r in replies])
    fail = np.array([1.0 if r["outcome"] == "fail" else 0.0 for r in replies])
    task = np.array([r["task"] for r in replies])
    folds = np.array([ar.fold_of(t) for t in task])
    y = on * fail
    tasks = sorted(set(task.tolist()))
    ti = np.array([{t: i for i, t in enumerate(tasks)}[t] for t in task])
    nt = len(tasks)
    cell = {"on_fail": (on == 1) & (fail == 1), "on_fixed": (on == 1) & (fail == 0), "off_fail": (on == 0) & (fail == 1), "off_fixed": (on == 0) & (fail == 0)}
    out = {"note": "post hoc, not pre-registered; a text model trained the same way as the ridge probe", "arms": {}}
    for name, docs, kw in (("tfidf_words_written", [text[r["id"]]["written"] for r in replies], {"ngram_range": (1, 2), "min_df": 2}),
                           ("tfidf_chars_written", [text[r["id"]]["written"] for r in replies], {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 3, "max_features": 60000}),
                           ("tfidf_words_prompt_and_written", [text[r["id"]]["prompt"] + "\n" + text[r["id"]]["written"] for r in replies], {"ngram_range": (1, 2), "min_df": 2})):
        scores = np.zeros(len(replies))
        for f in range(5):
            tr, te = folds != f, folds == f
            vec = TfidfVectorizer(sublinear_tf=True, **kw)
            Xtr = vec.fit_transform([d for d, k in zip(docs, tr) if k]).toarray()
            Xte = vec.transform([d for d, k in zip(docs, te) if k]).toarray()
            mu, sd = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
            Z = (Xtr - mu) / sd
            w = Z.T @ np.linalg.solve(Z @ Z.T + Z.shape[1] * np.eye(Z.shape[0]), y[tr] - y[tr].mean())
            ref = (Z @ w)[y[tr] == 0]
            scores[te] = (((Xte - mu) / sd) @ w - ref.mean()) / max(ref.std(), 1e-12)
        z = scores / scores[fail == 0].std()
        A = np.stack([ar.demean(on, ti, nt), ar.demean(fail, ti, nt), ar.demean(on * fail, ti, nt)], axis=1)
        b = ar.demean(z, ti, nt)

        def gamma(wt):
            wr = wt[ti]
            k = wr > 0
            s = np.sqrt(wr[k])
            return float(np.linalg.lstsq(A[k] * s[:, None], b[k] * s, rcond=None)[0][2])

        rng = np.random.default_rng(ar.BOOT_SEED)
        draws = []
        for _ in range(ar.BOOT_N):
            wt = np.bincount(rng.integers(0, nt, nt), minlength=nt).astype(np.float64)
            if any(wt[ti][cell[c]].sum() == 0 for c in cell):
                continue
            draws.append(gamma(wt))
        c = z - np.where(on == 1, z[cell["on_fixed"]].mean(), z[cell["off_fixed"]].mean())
        pos, pool = cell["on_fail"], ~cell["on_fail"]
        fp = flags_loto(c[pos], task[pos], c[pool], task[pool])
        fn = flags_loto(c[pool], task[pool], c[pool], task[pool])
        out["arms"][name] = {"gamma": gamma(np.ones(nt)), "interval95": ar.q(draws, 2.5, 97.5),
                             "cell_means": {k: float(z[v].mean()) for k, v in cell.items()},
                             "adjusted_auroc": auroc(c[cell["on_fail"]], c[cell["off_fail"]]),
                             "adjusted_catch": float(fp.mean()), "off_fail_flagged": int(fn[cell["off_fail"][pool]].sum())}
        a = out["arms"][name]
        print("%-32s gamma %+.2f (%+.2f to %+.2f) | adjusted AUROC %.3f | catch at 2%% %.1f%% | own honest failures flagged %d" % (
            name, a["gamma"], *a["interval95"], a["adjusted_auroc"], 100 * a["adjusted_catch"], a["off_fail_flagged"]))
    with open(os.path.join(HERE, "results", "text_rival.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    calls = {}
    for path in (os.path.join(HERE, "data", "t_calls.jsonl"),):
        for r in read_jsonl(path):
            if r["condition"] == "full" and r["kind"] == "general":
                calls.setdefault(r["id"], r["score"])
    for k, v in cell.items():
        ids = [r["id"] for r, m in zip(replies, v) if m]
        print("T nonzero scores in", k, sum(calls[i] > 0 for i in ids), "of", len(ids))


if __name__ == "__main__":
    main()
