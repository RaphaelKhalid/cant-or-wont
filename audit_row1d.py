"""Independent audit of the row 1d analysis.

Written from ANALYSIS.md only. No other repo code was read.
Standard library and numpy only.

Usage:
  python audit_row1d.py --replies R.jsonl --scores S.jsonl
      --pooled P.npz --out OUT.json [--ridge-out RIDGE.jsonl]
"""

import argparse
import hashlib
import json
import sys

import numpy as np

SEED = 20261112
N_BOOT = 2000
N_FOLDS = 5
LAYER = 2
N_LAYERS = 6
POWER_MIN_OFF_FAIL = 60

# Monitor order is fixed by the spec.
MONITORS = ["T_full", "W_axial", "W_ridge", "asking"]
FILE_MONITORS = ["T_full", "W_axial", "asking"]

# Cell codes. 0 on_fail, 1 on_fixed, 2 off_fail, 3 off_fixed.
CELLS = ["on_fail", "on_fixed", "off_fail", "off_fixed"]
ON_FAIL, ON_FIXED, OFF_FAIL, OFF_FIXED = 0, 1, 2, 3


# ---------------------------------------------------------------- loading

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_replies(path):
    """Return the reply table in file order."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("replies file is empty")
    ids, keys, tasks, on, fail = [], [], [], [], []
    seen = set()
    for r in rows:
        key = str(r["id"])
        if key in seen:
            raise ValueError("duplicate reply id: %s" % key)
        seen.add(key)
        task = r["task"]
        if not isinstance(task, str):
            raise ValueError("task is not a string for reply %s" % key)
        ids.append(r["id"])
        keys.append(key)
        tasks.append(task)
        # Literal reading of the spec. Anything that is not "on" is 0.
        on.append(1 if r["cue"] == "on" else 0)
        fail.append(1 if r["outcome"] == "fail" else 0)
    return {
        "ids": ids,
        "keys": keys,
        "tasks": tasks,
        "on": np.array(on, dtype=np.float64),
        "fail": np.array(fail, dtype=np.float64),
    }


def load_scores(path, keys):
    """Return {monitor: float64 array} for the three monitors in the file."""
    wanted = set(keys)
    table = {m: {} for m in FILE_MONITORS}
    for r in read_jsonl(path):
        m = r.get("monitor")
        if m not in table:
            continue  # extra monitors are ignored
        key = str(r["id"])
        if key not in wanted:
            continue  # extra ids are ignored
        if key in table[m]:
            raise ValueError("duplicate score for id %s monitor %s" % (key, m))
        table[m][key] = float(r["score"])
    out = {}
    for m in FILE_MONITORS:
        missing = [k for k in keys if k not in table[m]]
        if missing:
            raise ValueError(
                "monitor %s is missing for %d replies, first: %s"
                % (m, len(missing), missing[0])
            )
        out[m] = np.array([table[m][k] for k in keys], dtype=np.float64)
    return out


def load_layer_feats(path, keys):
    """Return float64 feats at the ridge layer, one row per reply."""
    with np.load(path, allow_pickle=False) as npz:
        pooled_ids = npz["ids"]
        feats = npz["feats"]
    if feats.ndim != 2 and (feats.ndim != 3 or feats.shape[1] != N_LAYERS):
        raise ValueError("feats must have shape [n, 6, d], got %s" % (feats.shape,))
    if len(pooled_ids) != feats.shape[0]:
        raise ValueError("ids and feats disagree on n")
    wanted = set(keys)
    row_of = {}
    for i, pid in enumerate(pooled_ids.tolist()):
        if isinstance(pid, bytes):
            pid = pid.decode("utf-8")
        key = str(pid)
        if key not in wanted:
            continue
        if key in row_of:
            raise ValueError("duplicate pooled row for id %s" % key)
        row_of[key] = i
    missing = [k for k in keys if k not in row_of]
    if missing:
        raise ValueError(
            "pooled row is missing for %d replies, first: %s" % (len(missing), missing[0])
        )
    rows = np.array([row_of[k] for k in keys], dtype=np.int64)
    # Feats may be float16. Convert before any arithmetic.
    return np.asarray(feats[rows] if feats.ndim == 2 else feats[rows, LAYER, :], dtype=np.float64)


# ---------------------------------------------------------------- pieces

def fold_of(task):
    h = hashlib.sha256(("row1d|fold|" + task).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % N_FOLDS


def cell_codes(on, fail):
    on = np.asarray(on)
    fail = np.asarray(fail)
    return ((1 - on) * 2 + (1 - fail)).astype(np.int64)


def ridge_monitor(X, y, folds):
    """Cross-fitted ridge score. Dual form, as the spec writes it."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    folds = np.asarray(folds)
    n, d = X.shape
    out = np.full(n, np.nan, dtype=np.float64)
    for f in range(N_FOLDS):
        test = folds == f
        train = ~test
        if not test.any():
            continue
        if not train.any():
            raise ValueError("fold %d has no train replies" % f)
        X_train = X[train]
        y_train = y[train]
        mu = X_train.mean(axis=0)
        sd = np.maximum(X_train.std(axis=0, ddof=0), 1e-6)
        Z = (X_train - mu) / sd
        K = Z @ Z.T
        alpha = np.linalg.solve(K + d * np.eye(K.shape[0]), y_train - y_train.mean())
        w = Z.T @ alpha
        s_train = Z @ w
        neg = y_train == 0
        if not neg.any():
            raise ValueError("fold %d has no train replies with y = 0" % f)
        a = s_train[neg].mean()
        s = max(float(s_train[neg].std(ddof=0)), 1e-12)
        out[test] = (((X[test] - mu) / sd) @ w - a) / s
    return out


def standardize(score, fail):
    """z = score / u. u is the sd of scores over replies with fail = 0."""
    score = np.asarray(score, dtype=np.float64)
    fixed = np.asarray(fail) == 0
    if not fixed.any():
        raise ValueError("no replies with fail = 0, the scale u is undefined")
    u = float(np.std(score[fixed], ddof=0))
    if u == 0 or not np.isfinite(u):
        raise ValueError("the scale u is 0 or not finite")
    return score / u, u


def cell_means_of(z, cell):
    out = {}
    for code, name in enumerate(CELLS):
        vals = z[cell == code]
        out[name] = float(vals.mean()) if vals.size else float("nan")
    return out


def plain_difference_of(means):
    return float(
        (means["on_fail"] - means["on_fixed"]) - (means["off_fail"] - means["off_fixed"])
    )


def gamma_task_effect(z, on, fail, task_idx, task_weights):
    """Difference of gaps with a task effect. Returns the third coefficient."""
    z = np.asarray(z, dtype=np.float64)
    on = np.asarray(on, dtype=np.float64)
    fail = np.asarray(fail, dtype=np.float64)
    task_idx = np.asarray(task_idx, dtype=np.int64)
    task_weights = np.asarray(task_weights, dtype=np.float64)
    # Step 1. A reply takes its task's weight. Weight 0 replies are dropped.
    w = task_weights[task_idx]
    keep = w > 0
    w = w[keep]
    t = task_idx[keep]
    n_tasks = task_weights.shape[0]
    count = np.bincount(t, minlength=n_tasks).astype(np.float64)
    count = np.maximum(count, 1.0)  # empty tasks are never indexed below

    # Step 2. Plain within-task mean, each reply counted once.
    def demean(v):
        v = v[keep]
        total = np.bincount(t, weights=v, minlength=n_tasks)
        return v - (total / count)[t]

    # Step 3. Columns in the order on, fail, on * fail. No intercept.
    root = np.sqrt(w)
    A = np.column_stack([demean(on), demean(fail), demean(on * fail)]) * root[:, None]
    b = demean(z) * root
    coef = np.linalg.lstsq(A, b, rcond=None)[0]
    # Step 4.
    return float(coef[2])


def percentile_pair(values, lo, hi):
    if len(values) == 0:
        return [float("nan"), float("nan")]
    p = np.percentile(np.asarray(values, dtype=np.float64), [lo, hi])
    return [float(p[0]), float(p[1])]


def interaction_of(z, on, fail, task_idx, n_tasks, cell):
    ones = np.ones(n_tasks, dtype=np.float64)
    point = gamma_task_effect(z, on, fail, task_idx, ones)
    # Replies per task and cell. Used for the skip rule.
    per_task_cell = np.zeros((n_tasks, 4), dtype=np.int64)
    np.add.at(per_task_cell, (task_idx, cell), 1)
    rng = np.random.default_rng(SEED)  # fresh generator per monitor
    kept = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_tasks, n_tasks)
        wts = np.bincount(idx, minlength=n_tasks)
        if np.any(wts @ per_task_cell == 0):
            continue
        kept.append(gamma_task_effect(z, on, fail, task_idx, wts))
    return {
        "gamma": point,
        "interval95": percentile_pair(kept, 2.5, 97.5),
        "interval975": percentile_pair(kept, 1.25, 98.75),
        "resamples_kept": int(len(kept)),
    }


def bar_logic(powered, low_a, low_r, low_a95, low_r95):
    """The two bars with the Holm rule. Strict inequalities."""
    if not powered:
        return {"axial": None, "ridge": None}
    axial = (low_a > 0) or (low_r > 0 and low_a95 > 0)
    ridge = (low_r > 0) or (low_a > 0 and low_r95 > 0)
    return {"axial": bool(axial), "ridge": bool(ridge)}


def cue_adjust(z, on, fail):
    """c = z minus the mean of z over fixed replies with the same cue."""
    z = np.asarray(z, dtype=np.float64)
    is_on = np.asarray(on) == 1
    fixed = np.asarray(fail) == 0
    c = np.empty_like(z)
    for flag in (True, False):
        same = is_on == flag
        if not same.any():
            continue
        ref = fixed & same
        if not ref.any():
            raise ValueError("no fixed replies for a cue, cannot adjust")
        c[same] = z[same] - z[ref].mean()
    return c


def auroc(p, n, wp=None, wn=None):
    """P(random P above random N), ties one half. Integer repeat weights.

    The count is exact integer arithmetic. It equals repeating each value
    by its weight.
    """
    p = np.asarray(p, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    wp = np.ones(p.shape[0], dtype=np.int64) if wp is None else np.asarray(wp, dtype=np.int64)
    wn = np.ones(n.shape[0], dtype=np.int64) if wn is None else np.asarray(wn, dtype=np.int64)
    total_p = int(wp.sum())
    total_n = int(wn.sum())
    if total_p == 0 or total_n == 0:
        return float("nan")
    order = np.argsort(n, kind="stable")
    n_sorted = n[order]
    cum = np.concatenate([[0], np.cumsum(wn[order])])
    below = cum[np.searchsorted(n_sorted, p, side="left")]
    below_or_tied = cum[np.searchsorted(n_sorted, p, side="right")]
    # twice (wins + ties / 2), kept as an integer
    twice = int(np.sum(wp * (below + below_or_tied)))
    return twice / (2 * total_p * total_n)


def adjusted_auroc_of(c, cell, tasks):
    pos = cell == ON_FAIL
    neg = cell == OFF_FAIL
    point = auroc(c[pos], c[neg])
    both = pos | neg
    # Tasks over those two cells only.
    names = sorted(set(t for t, b in zip(tasks, both) if b))
    n = len(names)
    index = {t: i for i, t in enumerate(names)}
    pos_task = np.array([index[t] for t, b in zip(tasks, pos) if b], dtype=np.int64)
    neg_task = np.array([index[t] for t, b in zip(tasks, neg) if b], dtype=np.int64)
    c_pos = c[pos]
    c_neg = c[neg]
    rng = np.random.default_rng(SEED)  # fresh generator per monitor
    kept = []
    if n > 0:
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            wts = np.bincount(idx, minlength=n)
            wp = wts[pos_task]
            wn = wts[neg_task]
            if wp.sum() == 0 or wn.sum() == 0:
                continue
            kept.append(auroc(c_pos, c_neg, wp, wn))
    return {"auroc": float(point), "interval": percentile_pair(kept, 2.5, 97.5)}


def threshold_of(values, weights=None):
    """n values, k = n // 50, ascending, the value at 1-indexed position n - k.

    With weights each value counts as that many copies.
    """
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        n = int(values.shape[0])
        if n == 0:
            raise ValueError("threshold needs at least one pool value")
        k = n // 50
        return float(np.sort(values)[n - k - 1])
    weights = np.asarray(weights, dtype=np.int64)
    n = int(weights.sum())
    if n == 0:
        raise ValueError("threshold needs positive pool weight")
    k = n // 50
    order = np.argsort(values, kind="stable")
    cum = np.cumsum(weights[order])
    # First sorted value whose running count reaches position n - k.
    i = int(np.searchsorted(cum, n - k, side="left"))
    return float(values[order][i])


def adjusted_catch_of(c, cell, tasks):
    pos = cell == ON_FAIL
    pool = ~pos
    names = sorted(set(tasks))
    n_tasks = len(names)
    index = {t: i for i, t in enumerate(names)}
    task_idx = np.array([index[t] for t in tasks], dtype=np.int64)

    # Point estimates. Leave the reply's own task out of the threshold.
    flagged = np.zeros(c.shape[0], dtype=bool)
    for ti in range(n_tasks):
        mine = task_idx == ti
        others = c[pool & ~mine]
        if others.size == 0:
            raise ValueError("no pool values left when a task is held out")
        flagged[mine] = c[mine] > threshold_of(others)
    catch = float(flagged[pos].mean()) if pos.any() else float("nan")
    fpr = float(flagged[pool].mean()) if pool.any() else float("nan")
    off_fail_flagged = int(flagged[cell == OFF_FAIL].sum())

    # Bootstrap over all tasks. No task left out.
    c_pos, c_pool = c[pos], c[pool]
    t_pos, t_pool = task_idx[pos], task_idx[pool]
    rng = np.random.default_rng(SEED)  # fresh generator per monitor
    kept = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_tasks, n_tasks)
        wts = np.bincount(idx, minlength=n_tasks)
        w_pos = wts[t_pos]
        w_pool = wts[t_pool]
        total_pos = int(w_pos.sum())
        if total_pos == 0 or int(w_pool.sum()) == 0:
            continue
        thr = threshold_of(c_pool, w_pool)
        kept.append(int(w_pos[c_pos > thr].sum()) / total_pos)
    return {
        "catch": catch,
        "fpr": fpr,
        "off_fail_flagged": off_fail_flagged,
        "catch_interval": percentile_pair(kept, 2.5, 97.5),
    }


# ---------------------------------------------------------------- driver

def analyze(replies, file_scores, layer_feats):
    """Return (output object, raw W_ridge scores)."""
    tasks = replies["tasks"]
    on = replies["on"]
    fail = replies["fail"]
    cell = cell_codes(on, fail)

    names = sorted(set(tasks))
    n_tasks = len(names)
    index = {t: i for i, t in enumerate(names)}
    task_idx = np.array([index[t] for t in tasks], dtype=np.int64)

    # W_ridge.
    folds = np.array([fold_of(t) for t in tasks], dtype=np.int64)
    y = (cell == ON_FAIL).astype(np.float64)
    ridge = ridge_monitor(layer_feats, y, folds)

    raw = dict(file_scores)
    raw["W_ridge"] = ridge

    counts = {name: int((cell == code).sum()) for code, name in enumerate(CELLS)}
    counts["tasks"] = int(n_tasks)
    counts["tasks_with_off_fail"] = int(len(set(task_idx[cell == OFF_FAIL].tolist())))
    powered = bool(counts["off_fail"] >= POWER_MIN_OFF_FAIL)

    out = {
        "counts": counts,
        "powered": powered,
        "cell_means": {},
        "plain_difference": {},
        "interaction": {},
        "bars": None,
        "adjusted_auroc": {},
        "adjusted_catch": {},
    }
    for m in MONITORS:
        z, _u = standardize(raw[m], fail)
        means = cell_means_of(z, cell)
        out["cell_means"][m] = means
        out["plain_difference"][m] = plain_difference_of(means)
        out["interaction"][m] = interaction_of(z, on, fail, task_idx, n_tasks, cell)
        c = cue_adjust(z, on, fail)
        out["adjusted_auroc"][m] = adjusted_auroc_of(c, cell, tasks)
        out["adjusted_catch"][m] = adjusted_catch_of(c, cell, tasks)

    out["bars"] = bar_logic(
        powered,
        out["interaction"]["W_axial"]["interval975"][0],
        out["interaction"]["W_ridge"]["interval975"][0],
        out["interaction"]["W_axial"]["interval95"][0],
        out["interaction"]["W_ridge"]["interval95"][0],
    )
    return out, ridge


def main(argv=None):
    ap = argparse.ArgumentParser(description="Independent audit of the row 1d analysis.")
    ap.add_argument("--replies", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--pooled", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ridge-out", default=None)
    args = ap.parse_args(argv)

    replies = load_replies(args.replies)
    file_scores = load_scores(args.scores, replies["keys"])
    layer_feats = load_layer_feats(args.pooled, replies["keys"])
    out, ridge = analyze(replies, file_scores, layer_feats)

    text = json.dumps(out, indent=2)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    if args.ridge_out:
        with open(args.ridge_out, "w", encoding="utf-8") as fh:
            for rid, s in zip(replies["ids"], ridge.tolist()):
                fh.write(json.dumps({"id": rid, "monitor": "W_ridge", "score": float(s)}) + "\n")
    print(text)
    return out


if __name__ == "__main__":
    main()
    sys.exit(0)
