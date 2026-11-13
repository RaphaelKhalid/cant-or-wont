# A linear probe can tell won't from can't

Code and data for the Andar Control post at https://andarcontrol.org/cant-or-wont.

A sandbagging model's held-back failures are compared with its own honest failures on the same 127 tasks. The question is whether a monitor scores the two differently once the trigger and bad work are subtracted.

## Result

| Monitor | Difference, in standard deviations | 95% CI |
|---|---|---|
| Linear probe, trained on labels | +1.03 | +0.81 to +1.27 |
| Claude Haiku 5.5, high effort | +0.42 | +0.06 to +0.79 |
| Trusted monitor, Llama 3.1 8B | +0.25 | -0.13 to +0.68 |
| TF-IDF classifier, same labels as the probe | +0.21 | -0.04 to +0.45 |
| Self-report | +0.01 | -0.15 to +0.20 |
| Published axial probe, no refit | -0.05 | -0.36 to +0.28 |

The probe leads Haiku by 0.60 (95% CI 0.23 to 1.01). Full results and limits are in `RESULTS.md`.

## Limits

- One model, with the behavior trained in.
- The probe was trained on labelled examples of this model holding back. Haiku had none.
- No model was trained to evade the probe.

## Reproduce

```
pip install -r requirements.txt
python analyze_row1d.py --out results/results.json --ridge-out results/ridge_scores.jsonl
python audit_row1d.py --replies data/replies.jsonl --scores data/scores.jsonl --pooled data/activations_layer18.npz --out results/results_audit.json --ridge-out results/ridge_scores_audit.jsonl
python haiku_rival.py --analyze
python audit_haiku_rival.py
python posthoc_text_rival.py
python audit_text_rival.py
python -m pytest -q
```

These read the saved data and call no model. `haiku_rival.py --run` calls the Anthropic API and needs a key.

## Files

| File | What it is |
|---|---|
| `PREREG.md` | The plan, frozen before the run |
| `ANALYSIS.md` | Exact definitions of every number |
| `PLAN_HAIKU_RIVAL.md` | The plan for the Haiku comparison, committed before any Haiku score existed |
| `RESULTS.md` | Results, limits and what was added after the run |
| `analyze_row1d.py`, `audit_row1d.py` | Two scripts written separately from `ANALYSIS.md`. They must agree |
| `haiku_rival.py`, `audit_haiku_rival.py` | The Haiku monitor and a second computation of its numbers |
| `posthoc_text_rival.py`, `audit_text_rival.py` | The TF-IDF classifier and a second computation |
| `lib.py` | Shared helpers |
| `data/replies.jsonl` | 1,601 replies: task, framing on or off, failed or fixed |
| `data/reply_texts.jsonl` | The prompt, the model's explanation, its code and which of the 3 bugs it fixed |
| `data/scores.jsonl` | Scores from the trusted monitor, the published probe and self-report |
| `data/activations_layer18.npz` | The model's activations at layer 18, averaged over the tokens it wrote |
| `data/t_calls.jsonl`, `data/haiku_calls.jsonl` | Raw answers from the two text monitors |
| `results/` | Outputs of the commands above |

## Notes on the data

- The model is Qwen3.5-9B with the LoRA adapter from the earlier paper, [White-box monitors for covert underperformance](https://andarcontrol.org/sandbagging). Its code is at https://github.com/RaphaelKhalid/covert-underperformance.
- The generation job is not in this repository. It wraps that paper's code.
- Six layers were recorded. The analysis reads one, so only that layer is included.
- In `data/replies.jsonl`, `source` says which run a reply came from. `row1b` and `row1c` are earlier runs on the same model, and `row1d` is this one.
- The plans were first committed in a private working repository. `PREREG.md` and `PLAN_HAIKU_RIVAL.md` give the commit and time of each.
