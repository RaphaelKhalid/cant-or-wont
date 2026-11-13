"""The row 1d scorer on made-up inputs with a planted difference of gaps. Plumbing only."""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def make(tmp_path, n_tasks=127, seed=0, planted=0.5):
    """Scores are task effect + 1.0 * cue + 0.6 * fail + planted * cue * fail + noise, for every given monitor."""
    rng = np.random.default_rng(seed)
    d = 24
    direction = rng.normal(size=d)
    replies, scores, ids, feats = [], [], [], []
    for t in range(n_tasks):
        te = rng.normal()
        hard = rng.random() < 0.3
        for k in range(22):
            on = k < 6
            fail = rng.random() < ((0.25 if on else 0.02) + (0.25 if hard else 0.0))
            rid = f"r{t:03d}-{k:02d}"
            replies.append({"id": rid, "task": f"task{t:03d}", "cue": "on" if on else "off", "outcome": "fail" if fail else "fixed"})
            x = rng.normal(size=(6, d))
            x[2] += direction * (1.5 if (on and fail) else 0.0)
            ids.append(rid)
            feats.append(x)
            base = te + 1.0 * on + 0.6 * fail + planted * (on and fail)
            for m in ("T_full", "W_axial", "asking"):
                scores.append({"id": rid, "monitor": m, "score": float(base + rng.normal())})
    p = {k: str(tmp_path / v) for k, v in (("replies", "replies.jsonl"), ("scores", "scores.jsonl"), ("pooled", "pooled.npz"), ("out", "out.json"))}
    with open(p["replies"], "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in replies)
    with open(p["scores"], "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in scores)
    order = rng.permutation(len(ids))
    np.savez(p["pooled"], ids=np.array(ids)[order], feats=np.array(feats)[order].astype(np.float16))
    return p


def run(p):
    r = subprocess.run([sys.executable, os.path.join(HERE, "analyze_row1d.py"), "--replies", p["replies"], "--scores", p["scores"],
                        "--pooled", p["pooled"], "--out", p["out"]], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.load(open(p["out"]))


def test_planted_difference_is_recovered(tmp_path):
    r = run(make(tmp_path, planted=0.5))
    assert r["powered"] and sum(r["counts"][c] for c in ("on_fail", "on_fixed", "off_fail", "off_fixed")) == 127 * 22
    for m in ("T_full", "W_axial", "asking"):
        g = r["interaction"][m]
        # The planted 0.5 is in raw units. Scores are divided by the spread of fixed replies, about 1.5.
        assert 0.15 < g["gamma"] < 0.55, g
        assert g["interval975"][0] <= g["interval95"][0] <= g["gamma"] <= g["interval95"][1] <= g["interval975"][1]
    assert r["adjusted_auroc"]["W_ridge"]["auroc"] > 0.9
    assert r["bars"]["ridge"] is True


def test_no_planted_difference_gives_none(tmp_path):
    r = run(make(tmp_path, planted=0.0, seed=3))
    g = r["interaction"]["W_axial"]
    assert g["interval95"][0] < 0 < g["interval95"][1], g
    assert r["bars"]["axial"] is False
