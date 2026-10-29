"""
LLM-as-judge scoring of the stored short-answer and open-ended answers.

Complements the lexical / embedding metrics (token F1, BERTScore, METEOR)
with a reference-guided judgment of factual correctness, completeness, and
faithfulness to the source document, in Spanish. Binary questions are not
judged: their exact-match label scoring already measures correctness
directly.

Like evaluate_results.py, this only scores stored predictions
(results/metrics/per_question_scores.csv); it never re-runs benchmark
inference, so no model answer changes.

Design notes:
  - The judge is Mistral-Small-24B-Instruct (see model_config.py), a model
    family different from every benchmarked configuration, to avoid
    self-preference bias.
  - The prompt contains no model name, quantization, or context condition
    (RAG / full-context), and items are processed in a seeded random order,
    so a partial run is balanced across models and conditions.
  - Greedy decoding. The judge sees the question, the verified reference
    answer, the answer to grade, and the full document of the correct
    course. Faithfulness is therefore graded against that course document
    for both conditions, not against the exact chunks a RAG model saw
    (retrieved chunk text is not stored), which keeps it comparable.
  - Scores (0-2 each): correctitud, completitud, fidelidad, plus a short
    justification written before the scores.
  - Resumable: rows already present in the output file are skipped, so a
    job that hits the Slurm time limit can simply be resubmitted.
  - Judge scores are a supplement to, not a replacement for, human
    verification: use --export-human-sample to build a blind sheet for
    hand-labeling a stratified sample and measure judge-human agreement.

Reads:  results/metrics/per_question_scores.csv, data/extracted_text/*.txt
Writes: results/metrics/llm_judge_scores.csv
        (--export-human-sample: results/metrics/judge_human_sample.csv and
         judge_human_sample_key.csv)

Usage:
    python llm_judge.py --dry-run --limit 3          # show prompts, no model
    python llm_judge.py --limit 20                    # smoke test + timing
    python llm_judge.py                               # full run (resumable)
    python llm_judge.py --model mistral-small-24b-judge-4bit
    python llm_judge.py --export-human-sample 60      # no model needed
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # src/scripts -> src

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES = PROJECT_ROOT / "results" / "metrics" / "per_question_scores.csv"
DEFAULT_TEXT_DIR = PROJECT_ROOT / "data" / "extracted_text"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "metrics" / "llm_judge_scores.csv"
HUMAN_SAMPLE = PROJECT_ROOT / "results" / "metrics" / "judge_human_sample.csv"
HUMAN_SAMPLE_KEY = PROJECT_ROOT / "results" / "metrics" / "judge_human_sample_key.csv"

JUDGED_CATEGORIES = ("short_answer", "open_ended")
CATEGORY_LABEL = {"short_answer": "respuesta corta", "open_ended": "pregunta abierta"}
SCORE_KEYS = ("correctitud", "completitud", "fidelidad")
KEY_COLS = ["model_key", "context_strategy", "category", "codigo", "question"]

JUDGE_PROMPT = """Eres un evaluador experto y estricto de respuestas a preguntas sobre programas de cursos de posgrado universitario. Tu tarea es calificar la RESPUESTA A EVALUAR comparándola con la RESPUESTA DE REFERENCIA y con el DOCUMENTO del curso.

Curso: {curso} ({codigo})
Tipo de pregunta: {tipo}

DOCUMENTO DEL CURSO:
\"\"\"
{documento}
\"\"\"

PREGUNTA:
{pregunta}

RESPUESTA DE REFERENCIA (verificada contra el documento):
{referencia}

RESPUESTA A EVALUAR:
{respuesta}

Califica con una escala de 0 a 2 los siguientes criterios:
- correctitud: ¿los hechos clave de la respuesta coinciden con la referencia?
  2 = todos los hechos clave son correctos y no hay contradicciones con la referencia.
  1 = parcialmente correcta: algunos hechos clave son correctos y otros faltan o son incorrectos.
  0 = incorrecta, contradice la referencia, o dice que no tiene información cuando la referencia sí la contiene.
- completitud: ¿cubre los puntos esenciales de la referencia?
  2 = cubre todos los puntos esenciales. 1 = cubre algunos. 0 = no cubre ninguno.
- fidelidad: ¿las afirmaciones de la respuesta están respaldadas por el DOCUMENTO?
  2 = todas las afirmaciones están respaldadas.
  1 = alguna afirmación no está respaldada por el documento, pero ninguna lo contradice.
  0 = alguna afirmación contradice el documento o inventa datos.

Reglas:
- Ignora la longitud, el estilo y el formato. Una respuesta más larga no es mejor, y no debes penalizar la extensión si los hechos son correctos.
- Basa la correctitud en la RESPUESTA DE REFERENCIA; usa el DOCUMENTO para evaluar la fidelidad y para resolver dudas sobre la referencia.
- La respuesta puede estar cortada al final; evalúa únicamente lo que está escrito.

Responde ÚNICAMENTE con un objeto JSON con estas claves, en este orden:
{{"justificacion": "<máximo dos oraciones>", "correctitud": <0, 1 o 2>, "completitud": <0, 1 o 2>, "fidelidad": <0, 1 o 2>}}"""


def load_course_documents(text_dir):
    """Maps course code (e.g. 'MC1001') to the full text of its document."""
    docs = {}
    for path in sorted(Path(text_dir).glob("*.txt")):
        docs[path.name.split("-")[0]] = path.read_text(encoding="utf-8")
    return docs


def build_judge_prompt(row, documents):
    return JUDGE_PROMPT.format(
        curso=row["curso"],
        codigo=row["codigo"],
        tipo=CATEGORY_LABEL[row["category"]],
        documento=documents[row["codigo"]].strip(),
        pregunta=row["question"],
        referencia=row["expected_answer"],
        respuesta=row["predicted_answer"] if isinstance(row["predicted_answer"], str) and row["predicted_answer"].strip() else "(sin respuesta)",
    )


def parse_judge_output(text):
    """
    Returns (scores_dict_or_None, justification, error_or_None). Tolerant of
    text around the JSON object; falls back to per-key regexes if the JSON
    itself is malformed. Scores outside {0, 1, 2} are treated as parse errors.
    """
    parsed = None
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            parsed = None

    scores = {}
    for key in SCORE_KEYS:
        value = parsed.get(key) if isinstance(parsed, dict) else None
        if value is None:
            m = re.search(rf'"{key}"\s*:\s*"?([0-9])', text)
            value = int(m.group(1)) if m else None
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if value not in (0, 1, 2):
            return None, "", f"missing or invalid '{key}'"
        scores[key] = int(value)

    justification = parsed.get("justificacion", "") if isinstance(parsed, dict) else ""
    return scores, str(justification), None


def load_pool(scores_path, categories):
    df = pd.read_csv(scores_path)
    df = df[df["category"].isin(categories)].copy()
    return df.drop_duplicates(KEY_COLS).reset_index(drop=True)


def already_judged(output_path):
    if not Path(output_path).exists():
        return set()
    done = pd.read_csv(output_path, usecols=KEY_COLS)
    return set(map(tuple, done[KEY_COLS].astype(str).values))


def export_human_sample(pool, n, seed):
    """
    Writes a blind sheet (no model or condition shown) for hand-labeling a
    sample stratified over category x context condition, plus a key file
    mapping each id back to its row. Judge-human agreement is computed from
    these two files and the judge output.
    """
    rng = random.Random(seed)
    cells = [(c, s) for c in JUDGED_CATEGORIES for s in ("rag", "fullcontext")]
    per_cell = max(n // len(cells), 1)
    picked = []
    for category, strategy in cells:
        cell = pool[(pool["category"] == category) & (pool["context_strategy"] == strategy)]
        picked.extend(cell.iloc[i] for i in rng.sample(range(len(cell)), min(per_cell, len(cell))))
    rng.shuffle(picked)

    sheet, key = [], []
    for i, row in enumerate(picked, start=1):
        item_id = f"H{i:03d}"
        sheet.append({
            "id": item_id, "curso": row["curso"], "categoria": CATEGORY_LABEL[row["category"]],
            "pregunta": row["question"], "respuesta_referencia": row["expected_answer"],
            "respuesta_a_evaluar": row["predicted_answer"],
            "correctitud": "", "completitud": "", "fidelidad": "",
        })
        key.append({"id": item_id, **{c: row[c] for c in KEY_COLS}})
    pd.DataFrame(sheet).to_csv(HUMAN_SAMPLE, index=False)
    pd.DataFrame(key).to_csv(HUMAN_SAMPLE_KEY, index=False)
    print(f"Wrote {len(sheet)} blind items to {HUMAN_SAMPLE} (fill correctitud/completitud/fidelidad, 0-2)")
    print(f"Key (id -> model/condition) in {HUMAN_SAMPLE_KEY}; do not open it before labeling.")


def run_judge(model_key, scores_path, text_dir, output_path, limit, seed, max_new_tokens, dry_run):
    documents = load_course_documents(text_dir)
    pool = load_pool(scores_path, JUDGED_CATEGORIES)
    done = already_judged(output_path)

    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)  # balanced partial runs; no ordering by model/condition
    todo = [i for i in order if tuple(str(pool.loc[i, c]) for c in KEY_COLS) not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(pool)} judgeable answers, {len(done)} already judged, {len(todo)} to judge now")

    if dry_run:
        for i in todo:
            print("=" * 80)
            print(build_judge_prompt(pool.loc[i], documents))
        return

    from model_loading import get_generate_fn  # lazy: heavy import, not needed for --dry-run / export
    generate_fn, meta = get_generate_fn(model_key, max_new_tokens=max_new_tokens)
    print(f"Judge loaded in {meta['load_time_s']:.1f}s")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = KEY_COLS + list(SCORE_KEYS) + ["justificacion", "parse_error", "judge_model", "judge_time_s", "raw_output"]
    write_header = not output_path.exists()
    n_errors, start_all = 0, time.time()

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for n, i in enumerate(todo, start=1):
            row = pool.loc[i]
            start = time.time()
            raw, error = "", None
            try:
                raw = generate_fn(build_judge_prompt(row, documents))
                scores, justification, error = parse_judge_output(raw)
            except Exception as e:  # keep going; record the failure for this row
                scores, justification, error = None, "", f"{type(e).__name__}: {e}"
            if error:
                n_errors += 1
            writer.writerow({
                **{c: row[c] for c in KEY_COLS},
                **({k: scores[k] for k in SCORE_KEYS} if scores else {k: "" for k in SCORE_KEYS}),
                "justificacion": justification, "parse_error": error or "", "judge_model": model_key,
                "judge_time_s": round(time.time() - start, 2), "raw_output": raw,
            })
            f.flush()
            if n % 20 == 0:
                print(f"  [{n}/{len(todo)}] {n_errors} parse/other errors, "
                      f"{(time.time() - start_all) / n:.1f}s/item avg")

    print(f"Done: {len(todo)} judged, {n_errors} errors, {time.time() - start_all:.0f}s total")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="mistral-small-24b-judge", help="judge key in model_config.MODELS")
    parser.add_argument("--scores", default=str(DEFAULT_SCORES))
    parser.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None, help="judge only this many (not yet judged) answers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="print prompts only; do not load the model")
    parser.add_argument("--export-human-sample", type=int, default=None, metavar="N",
                        help="write a blind N-item sheet for human labeling and exit (no model needed)")
    args = parser.parse_args()

    if args.export_human_sample is not None:
        export_human_sample(load_pool(args.scores, JUDGED_CATEGORIES), args.export_human_sample, args.seed)
        return
    run_judge(args.model, args.scores, args.text_dir, args.output, args.limit, args.seed,
              args.max_new_tokens, args.dry_run)


if __name__ == "__main__":
    main()
