"""
Benchmark loop: runs one model against the full QA dataset through the
RAG pipeline, recording raw results (not computed metrics — those are
derived later in a separate analysis pass, likely via bootstrap
resampling over these raw per-question results).

Usage:
    python run_benchmark.py --model qwen2.5-3b
    python run_benchmark.py --model deepseek-r1-14b-4bit --with-reasoning
"""

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # src/scripts -> src
from rag_pipeline import RAGIndex, chunk_documents, generate_answer
from model_loading import get_generate_fn
from model_config import MODELS
from load_qa_dataset import load_qa_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/scripts/.. -> src -> project root
DEFAULT_TEXT_DIR = PROJECT_ROOT / "data" / "extracted_text"
DEFAULT_QA_DIR = PROJECT_ROOT / "data" / "qa_dataset"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "index"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "raw"


# ---------------------------------------------------------------------
# RAG index: build once, reuse across models (identical retrieval for
# every model is the whole point of a fair benchmark)
# ---------------------------------------------------------------------

def load_extracted_texts(text_dir):
    doc_texts = {}
    for path in sorted(Path(text_dir).glob("*.txt")):
        doc_texts[path.name] = path.read_text(encoding="utf-8")
    if not doc_texts:
        raise FileNotFoundError(
            f"No .txt files found in {text_dir}. Run the PDF/DOCX extraction "
            f"step first and save output there."
        )
    return doc_texts


def build_or_load_index(text_dir, index_dir, chunk_size=500, overlap=50):
    index_dir = Path(index_dir)
    rag_index = RAGIndex()

    if (index_dir / "index.faiss").exists():
        print(f"Loading existing RAG index from {index_dir}")
        rag_index.load(index_dir)
        print(f"  {len(rag_index.chunks)} chunks loaded")
        return rag_index

    print(f"No existing index at {index_dir}, building from {text_dir}")
    doc_texts = load_extracted_texts(text_dir)
    chunks = chunk_documents(doc_texts, chunk_size=chunk_size, overlap=overlap)
    print(f"  {len(doc_texts)} documents -> {len(chunks)} chunks")

    rag_index.build(chunks)
    rag_index.save(index_dir)
    print(f"  Index saved to {index_dir}")
    return rag_index


# ---------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------

def run_benchmark(
    model_key,
    qa_dir=DEFAULT_QA_DIR,
    text_dir=DEFAULT_TEXT_DIR,
    index_dir=DEFAULT_INDEX_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    top_k=5,
    max_new_tokens=None,
    skip_reasoning=None,
):
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Options: {list(MODELS)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- RAG index: identical for every model ---
    rag_index = build_or_load_index(text_dir, index_dir)

    # --- QA dataset: full 90 questions per category, no subsampling ---
    qa = load_qa_dataset(qa_dir)
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

    # --- Run every question once, through the RAG pipeline ---
    output_path = output_dir / f"{model_key}.csv"
    fieldnames = [
        "model_key", "category", "question", "expected_answer",
        "predicted_answer", "retrieved_sources", "generation_time_s",
        "curso", "codigo", "error",
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
                expected = row["Respuesta"]

                start = time.time()
                predicted, sources, error = None, None, None
                try:
                    rag_result = generate_answer(question, rag_index, top_k=top_k, generate_fn=generate_fn)
                    predicted = rag_result["answer"]
                    sources = ";".join(c["source"] for c in rag_result["retrieved"])
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    n_errors += 1
                    traceback.print_exc()
                elapsed = time.time() - start

                writer.writerow({
                    "model_key": model_key,
                    "category": category,
                    "question": question,
                    "expected_answer": expected,
                    "predicted_answer": predicted,
                    "retrieved_sources": sources,
                    "generation_time_s": round(elapsed, 2),
                    "curso": row.get("Curso", ""),
                    "codigo": row.get("Código", ""),
                    "error": error,
                })
                f.flush()  # write incrementally — don't lose everything on a late crash

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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--skip-reasoning", action="store_true", default=None)
    parser.add_argument("--with-reasoning", action="store_true")
    parser.add_argument("--qa-dir", default=str(DEFAULT_QA_DIR))
    parser.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    skip_reasoning = None
    if args.skip_reasoning:
        skip_reasoning = True
    elif args.with_reasoning:
        skip_reasoning = False

    run_benchmark(
        model_key=args.model,
        qa_dir=args.qa_dir,
        text_dir=args.text_dir,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        skip_reasoning=skip_reasoning,
    )