"""
Generic model loading + generation, shared across all benchmarked models.

Design goal: one codepath for every model in model_config.MODELS, so the
only thing that varies across runs is the model itself (repo + quant),
keeping the RAG pipeline and prompting identical everywhere.
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from model_config import MODELS


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

def strip_reasoning(raw_output):
    """
    Returns only the text after </think>.

    Reasoning models' chat templates often inject the opening <think> tag
    as part of the prompt itself (not generated text), so we can't rely
    on seeing <think> in the output — only check for the closing tag.

    If no closing tag is found, generation was cut off mid-reasoning
    (max_new_tokens too low) — flagged explicitly rather than silently
    scoring a half-finished reasoning trace as if it were the answer.
    """
    if "</think>" in raw_output:
        return raw_output.split("</think>", 1)[-1].strip()
    return "[INCOMPLETE_REASONING: generation cut off before </think>; increase max_new_tokens]"


# ---------------------------------------------------------------------
# 3. Generation
# ---------------------------------------------------------------------

def build_generate_fn(model, tokenizer, model_key, max_new_tokens=512, skip_reasoning=False):
    """
    Returns a generate_fn(prompt: str) -> str, compatible with
    rag_pipeline.generate_answer(generate_fn=...).

    Applies the model's chat template, generates, decodes only the new
    tokens, and strips reasoning blocks for reasoning models.

    skip_reasoning: if True (only meaningful for reasoning models), injects
        an already-closed empty <think></think> block right after the
        prompt, before generation starts. Reasoning models like
        DeepSeek-R1-Distill are conditioned to expect this structural
        pattern, so an empty closed block causes them to skip the actual
        reasoning step and jump straight to the final answer — much
        faster, but likely at some accuracy cost on harder questions.
        Consider benchmarking both settings rather than picking one
        permanently.
    """
    is_reasoning_model = MODELS[model_key]["is_reasoning_model"]

    def generate_fn(prompt):
        messages = [{"role": "user", "content": prompt}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        if is_reasoning_model and skip_reasoning:
            # The chat template already opens <think> as part of the
            # prompt (confirmed empirically: generated text never shows
            # a literal <think>), so we only need to close it — adding
            # another opening tag here would duplicate it.
            forced_close = "\n\n</think>\n\n"
            close_ids = tokenizer(forced_close, add_special_tokens=False, return_tensors="pt")
            encoded["input_ids"] = torch.cat([encoded["input_ids"], close_ids["input_ids"]], dim=1)
            encoded["attention_mask"] = torch.cat(
                [encoded["attention_mask"], torch.ones_like(close_ids["input_ids"])], dim=1
            )

        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic, for reproducible benchmark runs
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_len:]
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        if is_reasoning_model and not skip_reasoning:
            return strip_reasoning(raw_text)
        return raw_text.strip()

    return generate_fn


# ---------------------------------------------------------------------
# 4. Convenience: load + build generate_fn + basic timing, in one call
# ---------------------------------------------------------------------

def get_generate_fn(model_key, max_new_tokens=512, skip_reasoning=None, verbose=True):
    """
    Loads a model by key and returns a ready-to-use generate_fn, along
    with load time (useful to log for the efficiency comparison in the
    paper).

    skip_reasoning: see build_generate_fn docstring. No effect on
        non-reasoning models. Defaults to None, which falls back to
        MODELS[model_key]["skip_reasoning"] — so the config's default
        applies automatically unless explicitly overridden (e.g. from a
        CLI flag, for a one-off ablation run).
    """
    if skip_reasoning is None:
        skip_reasoning = MODELS[model_key].get("skip_reasoning", False)

    start = time.time()
    model, tokenizer = load_model(model_key)
    load_time = time.time() - start

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{model_key}] loaded in {load_time:.1f}s | {n_params/1e9:.2f}B params")

    generate_fn = build_generate_fn(
        model, tokenizer, model_key, max_new_tokens=max_new_tokens, skip_reasoning=skip_reasoning
    )

    # Actual usable context length can differ from the marketed figure
    # (e.g. Qwen2.5-3B lists "up to 128K" but this checkpoint's default
    # config caps at 32768 without explicit RoPE-scaling/YaRN setup) —
    # expose it so callers can check before sending a large prompt.
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if verbose:
        print(f"[{model_key}] max_position_embeddings (actual context limit): {context_limit}")

    return generate_fn, {"load_time_s": load_time, "context_limit": context_limit}


if __name__ == "__main__":
    # Smoke test (requires HF access + GPU; run this on Kabré/Colab, not
    # in a sandbox without internet/model access).
    generate_fn, meta = get_generate_fn("qwen2.5-3b")
    print(meta)
    answer = generate_fn("¿Cuál es la capital de Costa Rica?")
    print("Answer:", answer)