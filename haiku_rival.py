"""Strongest text rival for row 1d. Every rule is in PLAN_HAIKU_RIVAL.md.

    python haiku_rival.py --check      10 replies, prints the projected cost
    python haiku_rival.py --run        all replies, resumes by id, stops at the cap
    python haiku_rival.py --analyze
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_row1d as ar  # noqa: E402
from lib import SEED, auroc, flags_loto, read_jsonl, stable  # noqa: E402

MODEL, EFFORT, MAX_TOKENS = "claude-haiku-5-5", "high", 8000
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.50 / 1e6
CAP = 1.36
CALLS = os.path.join(HERE, "data", "haiku_calls.jsonl")


def texts():
    out = {}
    for path in (os.path.join(HERE, "data", "reply_texts.jsonl"),):
        for r in read_jsonl(path):
            if "reply" in r:
                out[r["id"]] = r
    return out


def score(limit):
    import anthropic
    from tqdm import tqdm
    from lib import load_prompts, parse_score, transcript
    _, m = load_prompts()
    text = texts()
    replies = read_jsonl(os.path.join(HERE, "data", "replies.jsonl"))
    assert all(r["id"] in text for r in replies)
    done = read_jsonl(CALLS) if os.path.exists(CALLS) else []
    seen = {r["id"] for r in done}
    todo = [r for r in replies if r["id"] not in seen]
    if limit:
        todo = todo[:limit]
    # This key is not tied to one workspace, so the API asks for the workspace in a header.
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    client = anthropic.Anthropic(max_retries=5, default_headers={"anthropic-workspace-id": ws} if ws else None)
    lock, spent, stop = threading.Lock(), [sum(r["dollars"] for r in done)], [False]
    bar = tqdm(total=len(todo))

    def work(r):
        if stop[0]:
            return
        try:
            resp = client.messages.create(model=MODEL, max_tokens=MAX_TOKENS, output_config={"effort": EFFORT}, thinking={"type": "adaptive"},
                                          system=m["general"], messages=[{"role": "user", "content": transcript(text[r["id"]], "full", m)}])
        except anthropic.APIError as e:
            with lock:
                bar.write(f"failed {r['id']}: {type(e).__name__} {str(e)[:120]}")
            return
        out = "".join(b.text for b in resp.content if b.type == "text")
        s = parse_score(out) if resp.stop_reason == "end_turn" else None
        row = {"id": r["id"], "raw": out[:200], "score": s if s is not None else 50.0, "parse_fail": s is None, "stop_reason": resp.stop_reason,
               "tokens_in": resp.usage.input_tokens, "tokens_out": resp.usage.output_tokens,
               "dollars": resp.usage.input_tokens * PRICE_IN + resp.usage.output_tokens * PRICE_OUT}
        with lock:
            with open(CALLS, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            spent[0] += row["dollars"]
            bar.update(1)
            bar.set_postfix(dollars=f"{spent[0]:.3f}")
            if spent[0] >= CAP:
                stop[0] = True

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(work, todo))
    bar.close()
    rows = read_jsonl(CALLS)
    per = sum(r["dollars"] for r in rows) / max(len(rows), 1)
    print(json.dumps({"rows_on_disk": len(rows), "of": len(replies), "dollars_spent": round(sum(r["dollars"] for r in rows), 4),
                      "mean_tokens_in": round(float(np.mean([r["tokens_in"] for r in rows]))), "mean_tokens_out": round(float(np.mean([r["tokens_out"] for r in rows]))),
                      "projected_total": round(per * len(replies), 3), "cap": CAP, "parse_fail": sum(r["parse_fail"] for r in rows), "stopped_at_cap": stop[0]}))


def analyze():
    replies = read_jsonl(os.path.join(HERE, "data", "replies.jsonl"))
    calls = {r["id"]: r for r in read_jsonl(CALLS)}
    assert all(r["id"] in calls for r in replies), "some replies have no Haiku score"
    ridge = {r["id"]: r["score"] for r in read_jsonl(os.path.join(HERE, "results", "ridge_scores.jsonl"))}
    on = np.array([1.0 if r["cue"] == "on" else 0.0 for r in replies])
    fail = np.array([1.0 if r["outcome"] == "fail" else 0.0 for r in replies])
    task = np.array([r["task"] for r in replies])
    tasks = sorted(set(task.tolist()))
    ti = np.array([{t: i for i, t in enumerate(tasks)}[t] for t in task])
    nt = len(tasks)
    cell = {"on_fail": (on == 1) & (fail == 1), "on_fixed": (on == 1) & (fail == 0), "off_fail": (on == 0) & (fail == 1), "off_fixed": (on == 0) & (fail == 0)}
    raw = {"T_haiku": np.array([calls[r["id"]]["score"] + 0.999 * stable(SEED, "tie", "T_haiku", r["id"]) for r in replies]),
           "W_ridge": np.array([ridge[r["id"]] for r in replies])}
    A = np.stack([ar.demean(on, ti, nt), ar.demean(fail, ti, nt), ar.demean(on * fail, ti, nt)], axis=1)
    z = {k: v / v[fail == 0].std() for k, v in raw.items()}
    b = {k: ar.demean(v, ti, nt) for k, v in z.items()}

    def gamma(k, wt):
        w = wt[ti]
        keep = w > 0
        s = np.sqrt(w[keep])
        return float(np.linalg.lstsq(A[keep] * s[:, None], b[k][keep] * s, rcond=None)[0][2])

    rng = np.random.default_rng(ar.BOOT_SEED)
    draws = {"T_haiku": [], "W_ridge": []}
    for _ in range(ar.BOOT_N):
        wt = np.bincount(rng.integers(0, nt, nt), minlength=nt).astype(np.float64)
        if any(wt[ti][cell[c]].sum() == 0 for c in cell):
            continue
        for k in draws:
            draws[k].append(gamma(k, wt))
    one = np.ones(nt)
    h = z["T_haiku"]
    c = h - np.where(on == 1, h[cell["on_fixed"]].mean(), h[cell["off_fixed"]].mean())
    pos, pool = cell["on_fail"], ~cell["on_fail"]
    fp = flags_loto(c[pos], task[pos], c[pool], task[pool])
    fn = flags_loto(c[pool], task[pool], c[pool], task[pool])
    diff = np.array(draws["W_ridge"]) - np.array(draws["T_haiku"])
    scores = np.array([calls[r["id"]]["score"] for r in replies])
    out = {"n": len(replies), "parse_fail": int(sum(calls[r["id"]]["parse_fail"] for r in replies)),
           "dollars": round(sum(calls[r["id"]]["dollars"] for r in replies), 4),
           "raw_score_mean_by_cell": {k: float(scores[v].mean()) for k, v in cell.items()},
           "share_scored_zero": float((scores == 0).mean()),
           "cell_means_sd": {k: float(h[v].mean()) for k, v in cell.items()},
           "gamma": {"T_haiku": gamma("T_haiku", one), "W_ridge": gamma("W_ridge", one)},
           "interval95": {k: ar.q(v, 2.5, 97.5) for k, v in draws.items()},
           "ridge_minus_haiku": {"value": gamma("W_ridge", one) - gamma("T_haiku", one), "interval95": ar.q(diff, 2.5, 97.5)},
           "adjusted_auroc": auroc(c[cell["on_fail"]], c[cell["off_fail"]]),
           "adjusted_catch": float(fp.mean()), "off_fail_flagged": int(fn[cell["off_fail"][pool]].sum()), "resamples_kept": len(diff)}
    with open(os.path.join(HERE, "results", "haiku_rival.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.check:
        score(10)
    if a.run:
        score(0)
    if a.analyze:
        analyze()
