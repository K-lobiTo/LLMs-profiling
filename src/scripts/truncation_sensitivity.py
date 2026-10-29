"""
Sensitivity of the open-ended RAG vs. full-context comparison to the
300-token output cap.

Many open-ended answers hit max_new_tokens=300 and are cut off. This script
(1) counts how many answers per model/condition reach the cap, using each
model family's tokenizer on the saved answer text, and (2) repeats the
paired RAG - full-context comparison for open-ended BERTScore and METEOR on
only the questions where NEITHER condition's answer reached the cap, with
the same paired percentile bootstrap as bootstrap_analysis.py.

Token counts come from re-tokenizing decoded text, so they are approximate
(an answer is flagged as capped at >= CAP_THRESHOLD tokens, slightly under
300, to absorb re-tokenization drift). DeepSeek-R1-Distill-Qwen uses the
Qwen2.5 tokenizer family, so the Qwen tokenizer is used for it. No model is
run and no metric is recomputed; scores come from per_question_scores.csv.

Reads:  results/metrics/per_question_scores.csv
Writes: results/metrics/truncation_sensitivity.csv

Usage (from anywhere):
    python src/scripts/truncation_sensitivity.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT_DIR = ROOT / "results" / "metrics"

CAP_THRESHOLD = 295
B = 10000
TOKENIZERS = {"qwen": "Qwen/Qwen2.5-3B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"}
METRICS = ["bertscore_f1", "meteor"]

toks = {k: AutoTokenizer.from_pretrained(v) for k, v in TOKENIZERS.items()}


def n_tokens(model_key, text):
    tok = toks["llama" if model_key.startswith("llama") else "qwen"]
    return len(tok(text if isinstance(text, str) else "", add_special_tokens=False)["input_ids"])


P = pd.read_csv("results/metrics/per_question_scores.csv")
P = P[P.category == "open_ended"].copy()
P["capped"] = [n_tokens(m, t) >= CAP_THRESHOLD for m, t in zip(P.model_key, P.predicted_answer)]

rng = np.random.default_rng(42)
rows = []
for mk in sorted(P.model_key.unique()):
    d = {}
    for s in ("rag", "fullcontext"):
        g = P[(P.model_key == mk) & (P.context_strategy == s)].drop_duplicates(["codigo", "question"])
        d[s] = g.set_index(["codigo", "question"]).sort_index()
    idx = d["rag"].index.intersection(d["fullcontext"].index)
    a, b = d["rag"].loc[idx], d["fullcontext"].loc[idx]
    keep = (~a.capped.values) & (~b.capped.values)
    for m in METRICS:
        xa, xb = a[m].astype(float).values, b[m].astype(float).values
        for subset, mask in (("all", np.ones(len(idx), bool)), ("uncapped_both", keep)):
            va, vb = xa[mask], xb[mask]
            n = len(va)
            ix = rng.integers(0, n, (B, n))
            dd = va[ix].mean(1) - vb[ix].mean(1)
            lo, hi = np.percentile(dd, [2.5, 97.5])
            rows.append(dict(
                model=mk, metric=m, subset=subset, n=n,
                capped_rag=int(a.capped.sum()), capped_fullcontext=int(b.capped.sum()),
                rag=va.mean(), fullcontext=vb.mean(), diff=va.mean() - vb.mean(),
                d_lo=lo, d_hi=hi, sig=bool(lo > 0 or hi < 0),
            ))
R = pd.DataFrame(rows)
R.to_csv(OUT_DIR / "truncation_sensitivity.csv", index=False)

pd.set_option("display.width", 250)
f = R.copy()
f["RAG-FC [95% CI]"] = f.apply(lambda r: f"{r['diff']:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]", axis=1)
print(f[["model", "metric", "subset", "n", "capped_rag", "capped_fullcontext", "RAG-FC [95% CI]", "sig"]].to_string(index=False))
