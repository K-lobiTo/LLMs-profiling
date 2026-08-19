"""
Standalone smoke test for a single model.

Run this BEFORE wiring a model into the full RAG benchmark loop. Catches
gated-repo auth errors, broken chat templates, and OOM issues cheaply,
without burning queue time on a full 90-question run.

Usage:
    python smoke_test_model.py llama3.1-8b-4bit
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_loading import get_generate_fn
from model_config import MODELS

TEST_QUESTIONS = [
    "¿Cuál es la capital de Costa Rica?",
    "Responde únicamente con 'Sí' o 'No': ¿Es el cielo azul?",
]


def smoke_test(model_key):
    if model_key not in MODELS:
        print(f"Unknown model key '{model_key}'. Options: {list(MODELS)}")
        return

    print(f"=== Smoke test: {model_key} ({MODELS[model_key]['repo']}) ===\n")

    # --- 1. Check token is present for gated repos ---
    if "HF_TOKEN" not in os.environ:
        print("WARNING: HF_TOKEN not set in environment. Gated repos "
              "(Llama, some DeepSeek mirrors) will fail to download.\n")

    # --- 2. Load model ---
    try:
        generate_fn, meta = get_generate_fn(model_key, max_new_tokens=100)
    except Exception as e:
        print(f"LOAD FAILED: {type(e).__name__}: {e}")
        return

    print(f"Load time: {meta['load_time_s']:.1f}s")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            print(f"  GPU {i}: {allocated:.2f} GB allocated")

    # --- 3. Generate on a couple of throwaway questions ---
    for q in TEST_QUESTIONS:
        try:
            start = time.time()
            answer = generate_fn(q)
            elapsed = time.time() - start
            print(f"\nQ: {q}")
            print(f"A: {answer}")
            print(f"  (generated in {elapsed:.1f}s)")
        except torch.cuda.OutOfMemoryError:
            print(f"\nQ: {q}\n  OOM during generation. Consider a lower quant level.")
            return
        except Exception as e:
            print(f"\nQ: {q}\n  GENERATION FAILED: {type(e).__name__}: {e}")
            return

    print(f"\n=== {model_key}: smoke test passed ===")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smoke_test_model.py <model_key>")
        print(f"Available: {list(MODELS)}")
        sys.exit(1)

    smoke_test(sys.argv[1])