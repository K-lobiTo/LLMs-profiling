"""
Paired bootstrap comparison of the efficient tier (Qwen2.5-3B,
Llama-3.1-8B-4bit) against the upper-bound tier (DeepSeek-R1-Distill-14B,
full precision) under the RAG condition, per metric.

Every model answers the same questions, so the comparison is paired over
questions: the same resampled question indices are applied to both models
and the 95% percentile-bootstrap CI of (efficient - upper-bound) is
reported. Binary uses pooled macro-F1 (same definition as
evaluate_results.py). Single deterministic run per model, so the CIs only
reflect question sampling; no multiple-comparison correction.

Reads:  results/metrics/per_question_scores.csv
Writes: results/metrics/tier_comparison.csv

Usage (from anywhere):
    python src/scripts/tier_comparison.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT_DIR = ROOT / "results" / "metrics"

UPPER = "deepseek-r1-14b-f16"
EFFICIENT = ["qwen2.5-3b", "llama3.1-8b-4bit"]
B = 10000
rng = np.random.default_rng(42)

P = pd.read_csv("results/metrics/per_question_scores.csv")
P = P[P.context_strategy == "rag"]


def lab(x):
    return {"si": 0, "no": 1}.get(x, -1) if isinstance(x, str) else -1


def macro_f1(g, p):
    f = []
    for c in (0, 1):
        tp = ((p == c) & (g == c)).sum(); fp = ((p == c) & (g != c)).sum(); fn = ((g == c) & (p != c)).sum()
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return np.mean(f)


def frame(mk, cat):
    g = P[(P.model_key == mk) & (P.category == cat)].drop_duplicates(["codigo", "question"])
    return g.set_index(["codigo", "question"]).sort_index()


rows = []
for cat, metric in (("binary", "macro_f1"), ("short_answer", "token_f1"), ("short_answer", "bertscore_f1"),
                    ("open_ended", "bertscore_f1"), ("open_ended", "meteor")):
    u = frame(UPPER, cat)
    for mk in EFFICIENT:
        e = frame(mk, cat)
        idx = u.index.intersection(e.index)
        u_, e_ = u.loc[idx], e.loc[idx]
        n = len(idx)
        ix = rng.integers(0, n, (B, n))
        if metric == "macro_f1":
            gu, pu = u_.expected_binary_label.map(lab).values, u_.predicted_binary_label.map(lab).values
            ge, pe = e_.expected_binary_label.map(lab).values, e_.predicted_binary_label.map(lab).values
            vu = np.array([macro_f1(gu[i], pu[i]) for i in ix]); ve = np.array([macro_f1(ge[i], pe[i]) for i in ix])
            mu, me = macro_f1(gu, pu), macro_f1(ge, pe)
        else:
            xu, xe = u_[metric].astype(float).values, e_[metric].astype(float).values
            vu, ve = xu[ix].mean(1), xe[ix].mean(1); mu, me = xu.mean(), xe.mean()
        dd = ve - vu
        lo, hi = np.percentile(dd, [2.5, 97.5])
        rows.append(dict(category=cat, metric=metric, model=mk, upper=UPPER, n=n, model_score=me, upper_score=mu,
                         diff=me - mu, d_lo=lo, d_hi=hi, sig=bool(lo > 0 or hi < 0)))
R = pd.DataFrame(rows)
R.to_csv(OUT_DIR / "tier_comparison.csv", index=False)
pd.set_option("display.width", 200)
R["diff [95% CI]"] = R.apply(lambda r: f"{r['diff']:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]", axis=1)
print(R[["category", "metric", "model", "model_score", "upper_score", "diff [95% CI]", "sig"]].round(3).to_string(index=False))
