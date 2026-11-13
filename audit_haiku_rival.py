"""Second computation for the Haiku rival. Task dummy variables in place of demeaning. It shares only the tie rule with haiku_rival.py.

    python audit_haiku_rival.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def jitter(rid):
    """The tie rule of wcs.tasks.stable, loaded from lib.py: a number in [0, 1) from a hash of the seed and the labels."""
    from importlib import util
    spec = util.spec_from_file_location("tasks_mod", os.path.join(HERE, "lib.py"))
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.stable(mod.SEED, "tie", "T_haiku", rid)


def main():
    replies = rows("data/replies.jsonl")
    haiku = {r["id"]: r["score"] for r in rows("data/haiku_calls.jsonl")}
    ridge = {r["id"]: r["score"] for r in rows("results/ridge_scores.jsonl")}
    tasks = sorted({r["task"] for r in replies})
    col = {t: i for i, t in enumerate(tasks)}
    n, k = len(replies), len(tasks)
    X = np.zeros((n, 3 + k))
    fixed = np.zeros(n, dtype=bool)
    tix = np.zeros(n, dtype=int)
    for i, r in enumerate(replies):
        on, fail = float(r["cue"] == "on"), float(r["outcome"] == "fail")
        X[i, :3] = (on, fail, on * fail)
        X[i, 3 + col[r["task"]]] = 1.0
        fixed[i] = fail == 0
        tix[i] = col[r["task"]]
    y = {"haiku": np.array([haiku[r["id"]] + 0.999 * jitter(r["id"]) for r in replies]), "ridge": np.array([ridge[r["id"]] for r in replies])}
    y = {m: v / v[fixed].std() for m, v in y.items()}
    cells = [(X[:, 0] == a) & (X[:, 1] == b) for a in (0, 1) for b in (0, 1)]

    def fit(m, w):
        keep = w > 0
        s = np.sqrt(w[keep])[:, None]
        used = np.concatenate([[True, True, True], X[keep, 3:].sum(0) > 0])
        beta = np.linalg.lstsq(X[keep][:, used] * s, y[m][keep] * s[:, 0], rcond=None)[0]
        return float(beta[2])

    point = {m: fit(m, np.ones(n)) for m in y}
    rng = np.random.default_rng(20261112)
    d = {"haiku": [], "ridge": []}
    for _ in range(2000):
        wt = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)[tix]
        if any(wt[c].sum() == 0 for c in cells):
            continue
        for m in d:
            d[m].append(fit(m, wt))
    diff = np.array(d["ridge"]) - np.array(d["haiku"])
    out = {"gamma": {"T_haiku": point["haiku"], "W_ridge": point["ridge"]},
           "interval95": {"T_haiku": [float(np.percentile(d["haiku"], 2.5)), float(np.percentile(d["haiku"], 97.5))],
                          "W_ridge": [float(np.percentile(d["ridge"], 2.5)), float(np.percentile(d["ridge"], 97.5))]},
           "ridge_minus_haiku": {"value": point["ridge"] - point["haiku"], "interval95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}}
    main_out = json.load(open(os.path.join(HERE, "results", "haiku_rival.json"), encoding="utf-8"))
    worst = 0.0
    for key in ("gamma", "interval95"):
        for m in out[key]:
            worst = max(worst, float(np.max(np.abs(np.array(out[key][m]) - np.array(main_out[key][m])))))
    worst = max(worst, abs(out["ridge_minus_haiku"]["value"] - main_out["ridge_minus_haiku"]["value"]),
                float(np.max(np.abs(np.array(out["ridge_minus_haiku"]["interval95"]) - np.array(main_out["ridge_minus_haiku"]["interval95"])))))
    out["largest_difference_from_main"] = worst
    with open(os.path.join(HERE, "results", "haiku_rival_audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
