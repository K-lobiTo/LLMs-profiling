"""
Registry of candidate models for the benchmark.

All models are loaded from their native (safetensors) repo and quantized
on-the-fly via bitsandbytes, instead of relying on pre-quantized GGUF
files. This keeps a single loading/generation codepath for every model,
which matters for a fair benchmark and for a sane 10-day timeline.

`quant` options: None (full fp16), "8bit", "4bit"
"""

MODELS = {
    # --- Efficient tier ---
    "qwen2.5-3b": {
        "repo": "Qwen/Qwen2.5-3B-Instruct",
        "quant": None,  # small enough to run at full precision
        "is_reasoning_model": False,
    },
    "llama3.1-8b-f16": {
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "quant": None,
        "is_reasoning_model": False,
    },
    "llama3.1-8b-4bit": {
        "repo": "meta-llama/Llama-3.1-8B-Instruct",
        "quant": "4bit",
        "is_reasoning_model": False,
    },
    "deepseek-r1-14b-4bit": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quant": "4bit",
        "is_reasoning_model": True,
    },

    # --- Mid tier ---
    "deepseek-r1-14b-8bit": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quant": "8bit",
        "is_reasoning_model": True,
    },
    "qwen2.5-32b-4bit": {
        "repo": "Qwen/Qwen2.5-32B-Instruct",
        "quant": "4bit",
        "is_reasoning_model": False,
    },

    # --- Upper-bound reference tier ---
    "deepseek-r1-14b-f16": {
        "repo": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "quant": None,
        "is_reasoning_model": True,
    },
    "qwen2.5-32b-8bit": {
        "repo": "Qwen/Qwen2.5-32B-Instruct",
        "quant": "8bit",
        "is_reasoning_model": False,
    },
    "llama3.3-70b-4bit": {
        "repo": "meta-llama/Llama-3.3-70B-Instruct",
        "quant": "4bit",
        "is_reasoning_model": False,
    },
}