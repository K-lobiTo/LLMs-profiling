"""
Generic model loading + generation, shared across all benchmarked models.

Design goal: one codepath for every model in model_config.MODELS, so the
only thing that varies across runs is the model itself (repo + quant),
keeping the RAG pipeline and prompting identical everywhere.
"""

import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.model_config import MODELS


# ---------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------

def load_model(model_key):
    """
    Loads a model + tokenizer from model_config.MODELS by key.

    device_map="auto" spreads the model across all visible GPUs on the
    node automatically (relevant for the larger models on a multi-GPU
    Kabré allocation).
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Options: {list(MODELS)}")

    config = MODELS[model_key]
    repo = config["repo"]
    quant = config["quant"]

    tokenizer = AutoTokenizer.from_pretrained(repo)

    quant_config = None
    if quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        repo,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
    )
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------
# 2. Post-processing (strip DeepSeek-R1's <think> reasoning block)
# ---------------------------------------------------------------------

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_reasoning(raw_output):
    """
    Removes <think>...</think> reasoning blocks from reasoning-model
    output, leaving only the final answer for scoring. Also handles the
    case where the closing tag is missing (generation cut off mid-think).
    """
    cleaned = THINK_BLOCK_RE.sub("", raw_output)
    # If an unclosed <think> remains (truncated generation), drop everything
    # from <think> onward rather than scoring partial reasoning as the answer.
    cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


# ---------------------------------------------------------------------
# 3. Generation
# ---------------------------------------------------------------------

def build_generate_fn(model, tokenizer, model_key, max_new_tokens=512):
    """
    Returns a generate_fn(prompt: str) -> str, compatible with
    rag_pipeline.generate_answer(generate_fn=...).

    Applies the model's chat template, generates, decodes only the new
    tokens, and strips reasoning blocks for reasoning models.
    """
    is_reasoning_model = MODELS[model_key]["is_reasoning_model"]

    def generate_fn(prompt):
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic, for reproducible benchmark runs
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if is_reasoning_model:
            return strip_reasoning(raw_text)
        return raw_text.strip()

    return generate_fn


# ---------------------------------------------------------------------
# 4. Convenience: load + build generate_fn + basic timing, in one call
# ---------------------------------------------------------------------

def get_generate_fn(model_key, max_new_tokens=512, verbose=True):
    """
    Loads a model by key and returns a ready-to-use generate_fn, along
    with load time (useful to log for the efficiency comparison in the
    paper).
    """
    start = time.time()
    model, tokenizer = load_model(model_key)
    load_time = time.time() - start

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{model_key}] loaded in {load_time:.1f}s | {n_params/1e9:.2f}B params")

    generate_fn = build_generate_fn(model, tokenizer, model_key, max_new_tokens=max_new_tokens)
    return generate_fn, {"load_time_s": load_time}


if __name__ == "__main__":
    # Smoke test (requires HF access + GPU; to run on Kabré/Colab
    generate_fn, meta = get_generate_fn("qwen2.5-3b")
    print(meta)
    answer = generate_fn("¿Cuál es la capital de Costa Rica?")
    print("Answer:", answer)