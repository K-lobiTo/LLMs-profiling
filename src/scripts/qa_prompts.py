"""
Shared prompt-construction helpers, used by both run_benchmark.py (RAG)
and full_context_benchmark.py (entire corpus as context), so the only
difference between the two experiments is *how context is supplied*
(retrieved chunks vs. the full 15-document corpus) — not the question
wording or answer-format instructions.
"""

CATEGORY_INSTRUCTIONS = {
    "binary": "Responde únicamente con 'Sí' o 'No'.",
    "short_answer": "Da una respuesta breve y directa, en una sola frase.",
    "open_ended": "Da una respuesta completa y explicativa.",
}


def compose_question(row):
    """
    The raw 'Pregunta' text alone is ambiguous — it doesn't say which
    course or program it's about, so retrieval has no way to find the
    right document, and the model has no way to disambiguate "el curso"
    on its own. Prepend course/program context.
    """
    return f'En el curso {row["Curso"]} de la {row["Maestría"]}, {row["Pregunta"]}'


def build_instructed_question(row, category):
    """
    Composed question + a category-specific answer-format instruction,
    so binary/short-answer questions aren't padded with unnecessary
    explanation that would hurt exact-match-style scoring, while
    open-ended questions are explicitly invited to elaborate.
    """
    composed = compose_question(row)
    instruction = CATEGORY_INSTRUCTIONS.get(category, "")
    if instruction:
        return f"{composed}\n\n{instruction}"
    return composed