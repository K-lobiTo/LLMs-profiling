"""
Registry of candidate models for the benchmark.

All models are loaded from their native (safetensors) repo and quantized
on-the-fly via bitsandbytes, instead of relying on pre-quantized GGUF
files. This keeps a single loading/generation codepath for every model,
which matters for a fair benchmark and for a sane 10-day timeline.

`quant` options: None (full fp16), "8bit", "4bit"

`skip_reasoning`: only meaningful when is_reasoning_model=True. Defaults
to True here for all DeepSeek-R1 variants — with a RAG-augmented prompt
(retrieved context + question) on top of an already-long reasoning trace,
full reasoning was measured at 20-40s/question even on simple factual
questions. Across 90 questions x repeated batches x 3 quant levels, that
cost is not worth it for this project's timeline, especially since the
reasoning trace didn't change the final answer on any smoke test question
so far. Override per-run via the --skip-reasoning CLI flag if you want to
spot-check full reasoning on a small sample as a separate ablation.
"""

MODELS = {
    # --- Efficient tier ---
    "qwen2.5-3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct",
        "quant": None,  # small enough to run at full precision
        "is_reasoning_model": False,
        "skip_reasoning": False,  # not applicable, non-reasoning model
    },
    "llama3.1-8b-f16": {
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "quant": None,
        "is_reasoning_model": False,
        "skip_reasoning": False,
    },
    "llama3.1-8b-4bit": {
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "quant": "4bit",
        "is_reasoning_model": False,
        "skip_reasoning": False,
    },
    "deepseek-r1-14b-4bit": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quant": "4bit",
        "is_reasoning_model": True,
        "skip_reasoning": True,
    },

    # --- Mid tier ---
    "deepseek-r1-14b-8bit": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quant": "8bit",
        "is_reasoning_model": True,
        "skip_reasoning": True,
    },
    "qwen2.5-32b-4bit": {
        "repo": "Qwen/Qwen2.5-32B-Instruct",
        "quant": "4bit",
        "is_reasoning_model": False,
        "skip_reasoning": False,
    },

    # --- Upper-bound reference tier ---
    "deepseek-r1-14b-f16": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quant": None,
        "is_reasoning_model": True,
        "skip_reasoning": True,
    },
    "qwen2.5-32b-8bit": {
        "repo": "Qwen/Qwen2.5-32B-Instruct",
        "quant": "8bit",
        "is_reasoning_model": False,
        "skip_reasoning": False,
    },
}