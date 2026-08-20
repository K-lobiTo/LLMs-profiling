"""
Standalone smoke test for a single model.

Usage:
    python smoke_test_model.py llama3.1-8b-4bit
"""

import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # if run from src/scripts
from model_loading import get_generate_fn
from model_config import MODELS

TEST_QUESTIONS = [
    "¿Cuál es la capital de Costa Rica?",
    "Responde únicamente con 'Sí' o 'No': ¿Es el cielo azul?",
]


def smoke_test(model_key, skip_reasoning=False):
    if model_key not in MODELS:
        print(f"Unknown model key '{model_key}'. Options: {list(MODELS)}")
        return

    print(f"=== Smoke test: {model_key} ({MODELS[model_key]['repo']}) "
          f"{'[skip_reasoning]' if skip_reasoning else ''} ===\n")

    # Reasoning models (DeepSeek-R1) need far more headroom to reach
    # </think> before producing a real answer — 100 tokens isn't enough
    # even for a trivial question. Not needed when skip_reasoning bypasses
    # the think block entirely, but harmless to keep either way.
    max_new_tokens = 1024 if MODELS[model_key]["is_reasoning_model"] and not skip_reasoning else 100

    # --- 1. Check token is present for gated repos ---
    if "HF_TOKEN" not in os.environ:
        print("WARNING: HF_TOKEN not set in environment. Gated repos "
              "(Llama, some DeepSeek mirrors) will fail to download.\n")

    # --- 2. Load model ---
    try:
        generate_fn, meta = get_generate_fn(
            model_key, max_new_tokens=max_new_tokens, skip_reasoning=skip_reasoning
        )
    except Exception as e:
        print(f"LOAD FAILED: {type(e).__name__}")
        traceback.print_exc()
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
            print(f"\nQ: {q}\n  GENERATION FAILED: {type(e).__name__}")
            traceback.print_exc()
            return

    print(f"\n=== {model_key}: smoke test passed ===")


if __name__ == "__main__":
    print(f"DEBUG: sys.argv = {sys.argv}")
    if len(sys.argv) < 2:
        print("Usage: python smoke_test_model.py <model_key> [--skip-reasoning]")
        print(f"Available: {list(MODELS)}")
        sys.exit(1)

    skip_reasoning = "--skip-reasoning" in sys.argv
    smoke_test(sys.argv[1], skip_reasoning=skip_reasoning)