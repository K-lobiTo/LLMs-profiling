"""
Analysis of the LLM-judge scores (llm_judge.py) for short-answer and
open-ended questions.

Computes, from the stored judge output and per-question scores:
  1. Mean judge score per model x condition x category x criterion, and the
     share of fully correct answers (correctitud == 2).
  2. Paired RAG - full-context differences per model x category x criterion,
     with 95% percentile-bootstrap CIs over questions (same method as
     bootstrap_analysis.py), on all questions and on the subset where
     neither condition's answer hit the 300-token output cap.
  3. Spearman correlation between judge correctitud and the lexical /
     embedding metrics (token F1, BERTScore, METEOR).
  4. Judge scores for capped vs. uncapped answers.
  5. Abstention rate: share of answers that decline to answer ("No tengo
     suficiente informacion ..."), detected by a Spanish phrase regex (so
     approximate), per model x condition x category, and the mean judge
     correctitud for abstaining vs. answering responses.

The judge was NOT validated against human labels (no human sample was
labeled), so these scores are a supplementary signal, not a validated
measure. Single deterministic judge pass; CIs reflect question sampling
only; no multiple-comparison correction.

Reads:  results/metrics/llm_judge_scores.csv, results/metrics/per_question_scores.csv
Writes: results/metrics/judge_summary.csv, judge_paired.csv, judge_correlations.csv,
        judge_abstentions.csv

Usage (from anywhere):
    python src/scripts/judge_analysis.py
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT_DIR = ROOT / "results" / "metrics"

CAP_THRESHOLD = 295  # same approximate cap test as truncation_sensitivity.py
B = 10000
CRITERIA = ["correctitud", "completitud", "fidelidad"]
KEY = ["model_key", "context_strategy", "category", "codigo", "question"]
TOKENIZERS = {"qwen": "Qwen/Qwen2.5-3B-Instruct", "llama": "meta-llama/Llama-3.1-8B-Instruct"}

toks = {k: AutoTokenizer.from_pretrained(v) for k, v in TOKENIZERS.items()}


def n_tokens(model_key, text):
    tok = toks["llama" if model_key.startswith("llama") else "qwen"]
    return len(tok(text if isinstance(text, str) else "", add_special_tokens=False)["input_ids"])


J = pd.read_csv("results/metrics/llm_judge_scores.csv")
P = pd.read_csv("results/metrics/per_question_scores.csv").drop_duplicates(KEY)
D = J.merge(P[KEY + ["predicted_answer", "token_f1", "bertscore_f1", "meteor"]], on=KEY, how="left")
assert len(D) == len(J), "merge changed row count"
D["capped"] = [n_tokens(m, t) >= CAP_THRESHOLD for m, t in zip(D.model_key, D.predicted_answer)]
D["fully_correct"] = (D.correctitud == 2).astype(float)

# ---- 1. summary
summary = (D.groupby(["model_key", "context_strategy", "category"])
             .agg(n=("correctitud", "size"), correctitud=("correctitud", "mean"),
                  completitud=("completitud", "mean"), fidelidad=("fidelidad", "mean"),
                  frac_fully_correct=("fully_correct", "mean"), frac_capped=("capped", "mean"))
             .reset_index())
summary.to_csv(OUT_DIR / "judge_summary.csv", index=False)

# ---- 2. paired RAG - full-context
rng = np.random.default_rng(42)
rows = []
for (mk, cat), g in D.groupby(["model_key", "category"]):
    a = g[g.context_strategy == "rag"].drop_duplicates(["codigo", "question"]).set_index(["codigo", "question"]).sort_index()
    b = g[g.context_strategy == "fullcontext"].drop_duplicates(["codigo", "question"]).set_index(["codigo", "question"]).sort_index()
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    keep = (~a.capped.values) & (~b.capped.values)
    for crit in CRITERIA + ["fully_correct"]:
        xa, xb = a[crit].astype(float).values, b[crit].astype(float).values
        for subset, mask in (("all", np.ones(len(idx), bool)), ("uncapped_both", keep)):
            va, vb = xa[mask], xb[mask]
            n = len(va)
            if n < 2:
                continue
            ix = rng.integers(0, n, (B, n))
            dd = va[ix].mean(1) - vb[ix].mean(1)
            lo, hi = np.percentile(dd, [2.5, 97.5])
            rows.append(dict(model=mk, category=cat, criterion=crit, subset=subset, n=n,
                             rag=va.mean(), fullcontext=vb.mean(), diff=va.mean() - vb.mean(),
                             d_lo=lo, d_hi=hi, sig=bool(lo > 0 or hi < 0)))
paired = pd.DataFrame(rows)
paired.to_csv(OUT_DIR / "judge_paired.csv", index=False)

# ---- 3. correlations with existing metrics (Spearman = Pearson on ranks)
corr_rows = []
for cat, metrics in (("short_answer", ["token_f1", "bertscore_f1"]), ("open_ended", ["bertscore_f1", "meteor"])):
    sub = D[D.category == cat]
    for m in metrics:
        r = sub["correctitud"].rank().corr(sub[m].rank())
        corr_rows.append(dict(category=cat, metric=m, n=len(sub), spearman_vs_correctitud=r))
corr = pd.DataFrame(corr_rows)
corr.to_csv(OUT_DIR / "judge_correlations.csv", index=False)

# ---- 4. capped vs uncapped (open-ended)
cap = (D[D.category == "open_ended"].groupby("capped")[CRITERIA].mean().assign(n=D[D.category == "open_ended"].groupby("capped").size()))

# ---- 5. abstentions
ABSTAIN = re.compile(
    r"no (tengo|cuento con|dispongo de) (suficiente )?informaci|no (se )?(menciona|encuentra|especifica|proporciona|indica)"
    r"|no hay (suficiente )?informaci|no (est[a\u00e1]|aparece) (en|disponible)|insuficiente informaci", re.I)
D["abstains"] = D.predicted_answer.fillna("").map(lambda t: bool(ABSTAIN.search(t)))
abst = (D.groupby(["category", "model_key", "context_strategy"])
          .agg(n=("abstains", "size"), frac_abstains=("abstains", "mean"),
               correctitud_if_abstains=("correctitud", lambda s: s[D.loc[s.index, "abstains"]].mean()),
               correctitud_if_answers=("correctitud", lambda s: s[~D.loc[s.index, "abstains"]].mean()))
          .reset_index())
abst.to_csv(OUT_DIR / "judge_abstentions.csv", index=False)

pd.set_option("display.width", 250)
print("=== 1. mean judge scores (0-2) ===")
print(summary.pivot_table(index=["category", "model_key"], columns="context_strategy",
                          values=["correctitud", "completitud", "fidelidad", "frac_capped"]).round(2).to_string())
print("\n=== 2. paired RAG - full-context (all questions), CI excludes 0 -> sig ===")
f = paired[paired.subset == "all"].copy()
f["RAG-FC [95% CI]"] = f.apply(lambda r: f"{r['diff']:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]", axis=1)
print(f[f.criterion.isin(["correctitud", "completitud", "fidelidad"])][["category", "model", "criterion", "n", "rag", "fullcontext", "RAG-FC [95% CI]", "sig"]].round(2).to_string(index=False))
print("\n=== 2b. correctitud, uncapped-both subset ===")
f = paired[(paired.subset == "uncapped_both") & (paired.criterion == "correctitud")].copy()
f["RAG-FC [95% CI]"] = f.apply(lambda r: f"{r['diff']:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]", axis=1)
print(f[["category", "model", "n", "rag", "fullcontext", "RAG-FC [95% CI]", "sig"]].round(2).to_string(index=False))
print("\n=== 3. Spearman(judge correctitud, metric) ===")
print(corr.round(3).to_string(index=False))
print("\n=== 4. open-ended judge scores, capped vs uncapped answers ===")
print(cap.round(2).to_string())
print("\n=== 5. share of answers that abstain (regex-detected), by condition ===")
print(abst.pivot_table(index=["category", "model_key"], columns="context_strategy", values="frac_abstains").round(2).to_string())
print("\nmean judge correctitud: abstaining vs answering responses:", D.groupby("abstains").correctitud.mean().round(2).to_dict())
