"""
Evaluates all completed benchmark runs (both RAG and full-context),
computing accuracy metrics per the paper's Experimental Design and
efficiency metrics (time, estimated cost), then writes comparison
tables to results/metrics/.

This is intentionally separate from run_benchmark.py / full_context_
benchmark.py: those scripts only run inference and save raw predictions;
this script only scores them. Re-running this script (e.g. after fixing
a metric bug) never requires re-running expensive GPU inference.

Usage:
    python evaluate_results.py
    python evaluate_results.py --skip-open-ended   # faster: skip BERTScore/METEOR
"""

import argparse
import sys
import os
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # src/scripts -> src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from evaluation.metrics import (
    binary_classification_f1, extract_binary_label, token_f1,
    compute_bertscore, compute_rouge_l, compute_meteor,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = PROJECT_ROOT / "results" / "raw"
FULLCTX_DIR = PROJECT_ROOT / "results" / "raw_fullcontext"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

# Rough market reference for a comparable GPU (L40S), for the cost-per-
# 1000-queries estimate. This is NOT what Kabré itself charges (research
# clusters are typically free/allocation-based) — it's a stand-in for
# "what would this cost on commercial cloud infrastructure", which is
# the actual comparison point for the paper's cost argument (self-hosted
# open models vs. paying per-token for a commercial API). Rate is
# Spheron's on-demand dedicated L40S rate — see spheron2026l40s in
# bibli.bib.
GPU_HOURLY_RATE_USD = 0.96


def load_all_results():
    """
    Loads every non-validation (*_limitN excluded) CSV from both result
    directories, tagging each with context_strategy.
    """
    frames = []
    for directory, strategy in [(RAG_DIR, "rag"), (FULLCTX_DIR, "fullcontext")]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.csv")):
            if "_limit" in path.stem:
                continue  # skip pipeline-validation runs, not real results
            df = pd.read_csv(path)
            df["context_strategy"] = strategy
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No result CSVs found in {RAG_DIR} or {FULLCTX_DIR}. "
            f"Run the benchmarks first."
        )

    return pd.concat(frames, ignore_index=True)


def compute_per_question_scores(df, skip_open_ended=False):
    """
    Returns a copy of df with per-question scoring columns added, so
    individual wrong answers can be inspected directly rather than only
    seeing aggregate F1/BERTScore numbers. This is the piece that was
    missing before: n_errors/n_unparseable in the aggregate table are
    pipeline-health counters, not correctness counters — this is where
    actual right/wrong per question lives.
    """
    df = df.copy()
    df["predicted_binary_label"] = None
    df["expected_binary_label"] = None
    df["binary_correct"] = None
    df["token_f1"] = None
    df["bertscore_f1"] = None
    df["rouge_l"] = None
    df["meteor"] = None

    valid_mask = df["predicted_answer"].notna() & df["error"].isna()

    binary_mask = valid_mask & (df["category"] == "binary")
    df.loc[binary_mask, "predicted_binary_label"] = df.loc[binary_mask, "predicted_answer"].apply(extract_binary_label)
    df.loc[binary_mask, "expected_binary_label"] = df.loc[binary_mask, "expected_answer"].apply(extract_binary_label)
    df.loc[binary_mask, "binary_correct"] = (
        df.loc[binary_mask, "predicted_binary_label"] == df.loc[binary_mask, "expected_binary_label"]
    )

    short_mask = valid_mask & (df["category"] == "short_answer")
    df.loc[short_mask, "token_f1"] = [
        token_f1(pred, exp)
        for pred, exp in zip(df.loc[short_mask, "predicted_answer"], df.loc[short_mask, "expected_answer"])
    ]

    if not skip_open_ended:
        open_mask = valid_mask & (df["category"] == "open_ended")
        if open_mask.any():
            preds = df.loc[open_mask, "predicted_answer"].tolist()
            refs = df.loc[open_mask, "expected_answer"].tolist()
            bert = compute_bertscore(preds, refs)
            rouge = compute_rouge_l(preds, refs)
            meteor = compute_meteor(preds, refs)
            df.loc[open_mask, "bertscore_f1"] = bert["f1"]
            df.loc[open_mask, "rouge_l"] = rouge
            df.loc[open_mask, "meteor"] = meteor

    return df


def compute_accuracy_metrics(df, skip_open_ended=False):
    """
    Returns a DataFrame: one row per (model_key, context_strategy,
    category), with the appropriate accuracy metric(s) for that category.
    """
    rows = []

    for (model_key, strategy, category), group in df.groupby(
        ["model_key", "context_strategy", "category"]
    ):
        valid = group[group["predicted_answer"].notna() & group["error"].isna()]
        n_errors = len(group) - len(valid)

        row = {
            "model_key": model_key,
            "context_strategy": strategy,
            "category": category,
            "n_questions": len(group),
            "n_errors": n_errors,
        }

        if category == "binary":
            pairs = list(zip(valid["predicted_answer"], valid["expected_answer"]))
            result = binary_classification_f1(pairs)
            row["f1"] = result["macro_f1"]
            row["n_unparseable"] = result["n_unparseable"]

        elif category == "short_answer":
            f1s = [
                token_f1(pred, exp)
                for pred, exp in zip(valid["predicted_answer"], valid["expected_answer"])
            ]
            row["f1"] = sum(f1s) / len(f1s) if f1s else 0.0

        elif category == "open_ended":
            if skip_open_ended or len(valid) == 0:
                row["bertscore_f1"] = None
                row["rouge_l"] = None
                row["meteor"] = None
            else:
                preds = valid["predicted_answer"].tolist()
                refs = valid["expected_answer"].tolist()
                bert = compute_bertscore(preds, refs)
                rouge = compute_rouge_l(preds, refs)
                meteor = compute_meteor(preds, refs)
                row["bertscore_f1"] = sum(bert["f1"]) / len(bert["f1"])
                row["rouge_l"] = sum(rouge) / len(rouge)
                row["meteor"] = sum(meteor) / len(meteor)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_efficiency_metrics(df):
    """
    Returns a DataFrame: one row per (model_key, context_strategy), with
    mean/median generation time and an estimated cost per 1000 queries.
    """
    rows = []
    for (model_key, strategy), group in df.groupby(["model_key", "context_strategy"]):
        valid = group[group["generation_time_s"].notna()]
        mean_time = valid["generation_time_s"].mean()
        median_time = valid["generation_time_s"].median()
        est_cost_per_1000 = (mean_time * 1000 / 3600) * GPU_HOURLY_RATE_USD

        rows.append({
            "model_key": model_key,
            "context_strategy": strategy,
            "mean_generation_time_s": round(mean_time, 3),
            "median_generation_time_s": round(median_time, 3),
            "est_cost_per_1000_queries_usd": round(est_cost_per_1000, 4),
        })

    return pd.DataFrame(rows)


def build_rag_vs_fullcontext_comparison(accuracy_df, efficiency_df):
    """
    Pivots accuracy + efficiency side by side per model, so the RAG vs.
    full-context tradeoff (the paper's central "best results, minimum
    cost" question) is readable in a single table rather than split
    across the two long-format DataFrames above.
    """
    acc_metric_cols = [c for c in ["f1", "bertscore_f1", "rouge_l", "meteor"] if c in accuracy_df.columns]
    acc_wide = accuracy_df.pivot_table(
        index=["model_key", "category"], columns="context_strategy",
        values=acc_metric_cols,
    )
    acc_wide.columns = [f"{metric}_{strategy}" for metric, strategy in acc_wide.columns]
    acc_wide = acc_wide.reset_index()

    eff_metric_cols = [c for c in ["mean_generation_time_s", "est_cost_per_1000_queries_usd"] if c in efficiency_df.columns]
    eff_wide = efficiency_df.pivot_table(
        index="model_key", columns="context_strategy",
        values=eff_metric_cols,
    )
    eff_wide.columns = [f"{metric}_{strategy}" for metric, strategy in eff_wide.columns]
    eff_wide = eff_wide.reset_index()

    return acc_wide, eff_wide


def main(skip_open_ended=False):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading result CSVs...")
    df = load_all_results()
    print(f"  {len(df)} total rows across "
          f"{df['model_key'].nunique()} models x "
          f"{df['context_strategy'].nunique()} context strategies")

    print("\nComputing accuracy metrics...")
    accuracy_df = compute_accuracy_metrics(df, skip_open_ended=skip_open_ended)
    accuracy_path = METRICS_DIR / "accuracy_by_model_category.csv"
    accuracy_df.to_csv(accuracy_path, index=False)
    print(f"  Saved to {accuracy_path}")

    print("\nComputing per-question scores (for inspecting individual wrong answers)...")
    per_question_df = compute_per_question_scores(df, skip_open_ended=skip_open_ended)
    per_question_path = METRICS_DIR / "per_question_scores.csv"
    per_question_df.to_csv(per_question_path, index=False)
    print(f"  Saved to {per_question_path}")

    n_binary_wrong = (per_question_df["binary_correct"] == False).sum()
    n_binary_total = per_question_df["binary_correct"].notna().sum()
    if n_binary_total > 0:
        print(f"  Binary questions: {n_binary_wrong}/{n_binary_total} wrong "
              f"(this is where actual mistakes live, not n_errors/n_unparseable)")

    print("\nComputing efficiency metrics...")
    efficiency_df = compute_efficiency_metrics(df)
    efficiency_path = METRICS_DIR / "efficiency_by_model.csv"
    efficiency_df.to_csv(efficiency_path, index=False)
    print(f"  Saved to {efficiency_path}")

    print("\nBuilding RAG vs. full-context comparison...")
    acc_wide, eff_wide = build_rag_vs_fullcontext_comparison(accuracy_df, efficiency_df)
    acc_wide.to_csv(METRICS_DIR / "comparison_accuracy.csv", index=False)
    eff_wide.to_csv(METRICS_DIR / "comparison_efficiency.csv", index=False)
    print(f"  Saved to {METRICS_DIR / 'comparison_accuracy.csv'}")
    print(f"  Saved to {METRICS_DIR / 'comparison_efficiency.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-open-ended", action="store_true",
                         help="Skip BERTScore/ROUGE/METEOR (faster, useful for a quick check)")
    args = parser.parse_args()
    main(skip_open_ended=args.skip_open_ended)