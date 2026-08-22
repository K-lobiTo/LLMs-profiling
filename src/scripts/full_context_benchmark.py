"""
Full-context baseline: instead of retrieving relevant chunks via RAG,
the ENTIRE 15-document corpus is stuffed into the prompt for every
question. Feasible here because the corpus (~20-25K tokens) comfortably
fits within all 8 benchmarked models' 128K context windows.

This exists to answer a real question for the paper: for a corpus this
small, is RAG's retrieval step actually buying anything over just
including everything? Compare this script's results directly against
run_benchmark.py's for the same model.

Uses the SAME question composition + category instructions as the RAG
benchmark (via qa_prompts.py), so the only variable between the two
experiments is the context strategy itself.

Usage:
    python full_context_benchmark.py --model qwen2.5-3b
    python full_context_benchmark.py --model qwen2.5-3b --limit 2
"""

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # src/scripts -> src
from model_loading import get_generate_fn
from model_config import MODELS
from load_qa_dataset import load_qa_dataset
from qa_prompts import build_instructed_question

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT_DIR = PROJECT_ROOT / "data" / "extracted_text"
DEFAULT_QA_DIR = PROJECT_ROOT / "data" / "qa_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "raw_fullcontext"


def load_full_corpus(text_dir):
    """
    Concatenates every extracted document into one block, each labeled
    with its filename — same source-labeling convention as the RAG
    pipeline's chunk prefixes, for consistency.
    """
    parts = []
    for path in sorted(Path(text_dir).glob("*.txt")):
        doc_title = path.stem
        text = path.read_text(encoding="utf-8")
        parts.append(f"[Documento: {doc_title}]\n{text}")
    if not parts:
        raise FileNotFoundError(f"No .txt files found in {text_dir}")
    return "\n\n".join(parts)


def build_full_context_prompt(question, corpus):
    return (
        "Responde la siguiente pregunta usando únicamente la información "
        "de los documentos proporcionados. Si la respuesta no está en los "
        "documentos, indica que no tienes suficiente información.\n\n"
        f"Documentos:\n{corpus}\n\n"
        f"Pregunta: {question}\n"
        "Respuesta:"
    )


def run_full_context_benchmark(
    model_key,
    qa_dir=DEFAULT_QA_DIR,
    text_dir=DEFAULT_TEXT_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    max_new_tokens=None,
    skip_reasoning=None,
    limit=None,
):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Options: {list(MODELS)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Full corpus: identical for every model, built once ---
    corpus = load_full_corpus(text_dir)
    corpus_word_count = len(corpus.split())
    print(f"Full corpus loaded: {corpus_word_count} words (~{int(corpus_word_count * 1.4)} tokens estimated)")

    # --- QA dataset ---
    qa = load_qa_dataset(qa_dir)
    if limit is not None:
        qa = {category: df.head(limit) for category, df in qa.items()}
        print(f"--limit={limit}: using only the first {limit} question(s) per category")
    total_questions = sum(len(df) for df in qa.values())
    print(f"Loaded QA dataset: {total_questions} questions across {len(qa)} categories")

    # --- Model ---
    is_reasoning_model = MODELS[model_key]["is_reasoning_model"]
    if max_new_tokens is None:
        effective_skip = skip_reasoning
        if effective_skip is None:
            effective_skip = MODELS[model_key].get("skip_reasoning", False)
        max_new_tokens = 1024 if (is_reasoning_model and not effective_skip) else 300

    generate_fn, meta = get_generate_fn(
        model_key, max_new_tokens=max_new_tokens, skip_reasoning=skip_reasoning
    )
    print(f"Model loaded in {meta['load_time_s']:.1f}s, max_new_tokens={max_new_tokens}")

    context_limit = meta.get("context_limit")
    estimated_prompt_tokens = int(corpus_word_count * 1.4) + 200  # + rough instruction/question overhead
    if context_limit is not None and (estimated_prompt_tokens + max_new_tokens) > context_limit:
        print(
            f"\nWARNING: estimated prompt tokens (~{estimated_prompt_tokens}) + "
            f"max_new_tokens ({max_new_tokens}) = ~{estimated_prompt_tokens + max_new_tokens}, "
            f"which exceeds {model_key}'s actual context limit ({context_limit}). "
            f"Results may silently degrade or truncate context. Consider a smaller "
            f"corpus subset, a shorter max_new_tokens, or skip this model for the "
            f"full-context comparison.\n"
        )

    # --- Run every question once, full corpus in every prompt ---
    suffix = f"_limit{limit}" if limit is not None else ""
    output_path = output_dir / f"{model_key}{suffix}.csv"
    fieldnames = [
        "model_key", "category", "question", "composed_question", "expected_answer",
        "predicted_answer", "generation_time_s", "curso", "codigo", "error",
    ]

    n_done = 0
    n_errors = 0
    run_start = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for category, df in qa.items():
            for _, row in df.iterrows():
                question = row["Pregunta"]
                composed_question = build_instructed_question(row, category)
                expected = row["Respuesta"]

                start = time.time()
                predicted, error = None, None
                try:
                    prompt = build_full_context_prompt(composed_question, corpus)
                    predicted = generate_fn(prompt)
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    n_errors += 1
                    traceback.print_exc()
                elapsed = time.time() - start

                writer.writerow({
                    "model_key": model_key,
                    "category": category,
                    "question": question,
                    "composed_question": composed_question,
                    "expected_answer": expected,
                    "predicted_answer": predicted,
                    "generation_time_s": round(elapsed, 2),
                    "curso": row.get("Curso", ""),
                    "codigo": row.get("Código", ""),
                    "error": error,
                })
                f.flush()

                n_done += 1
                if n_done % 10 == 0:
                    print(f"  [{n_done}/{total_questions}] "
                          f"({n_errors} errors so far, {elapsed:.1f}s last question)")

    total_elapsed = time.time() - run_start
    print(f"\nDone: {n_done} questions, {n_errors} errors, "
          f"{total_elapsed:.1f}s total ({total_elapsed/max(n_done,1):.1f}s/question avg)")
    print(f"Results written to {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help=f"One of: {list(MODELS)}")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--skip-reasoning", action="store_true", default=None)
    parser.add_argument("--with-reasoning", action="store_true")
    parser.add_argument("--qa-dir", default=str(DEFAULT_QA_DIR))
    parser.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N questions per category (for quick validation)")
    args = parser.parse_args()

    skip_reasoning = None
    if args.skip_reasoning:
        skip_reasoning = True
    elif args.with_reasoning:
        skip_reasoning = False

    run_full_context_benchmark(
        model_key=args.model,
        qa_dir=args.qa_dir,
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        skip_reasoning=skip_reasoning,
        limit=args.limit,
    )