"""
Metric implementations for the QA benchmark, matching the Experimental
Design section of the paper:
  - binary        -> classification F1 (Sí/No as a 2-class label problem)
  - short_answer   -> token-overlap F1 (SQuAD-style) + BERTScore
  - open_ended     -> BERTScore, METEOR

Design notes:
  - Binary and short-answer both get called "F1 score" in the paper, but
    they're conceptually different: binary questions have a fixed label
    space (Sí/No), so classification F1 makes sense; short answers don't,
    so token-overlap F1 (same family as SQuAD) is used instead.
  - Short-answer questions also get BERTScore alongside token-overlap
    F1, added as a complementary metric rather than a replacement: F1
    measures exact lexical overlap and is unforgiving of a verbose-but-
    correct answer (predicted tokens beyond the terse gold answer count
    against precision regardless of whether they're wrong or merely
    extra), while BERTScore tolerates paraphrase/elaboration. The
    reverse tradeoff also matters here — BERTScore is comparatively
    insensitive to a single swapped fact (e.g. a wrong percentage or
    name) embedded in an otherwise similar sentence, which is exactly
    the kind of error short-answer questions are meant to catch — so
    it's reported alongside F1, not in place of it.
  - BERTScore uses BETO (dccuchile/bert-base-spanish-wwm-cased), a
    Spanish BERT model, rather than a multilingual or English model,
    since both the source documents and expected answers are in Spanish.
  - ROUGE-L was dropped from open-ended evaluation: it's a token-overlap
    F-measure like short-answer's, and open-ended answers in this
    dataset run roughly 10x longer than the curated reference (the
    "Da una respuesta completa y explicativa" instruction invites
    elaboration), which crushes ROUGE-L's precision term far more than
    METEOR's (whose default recall-weighted formula, alpha=0.9, is
    comparatively tolerant of that same length mismatch). Keeping both
    would have meant reporting two lexical-overlap metrics where one
    (ROUGE-L) is dominated by a verbosity artifact rather than adding
    independent signal beyond METEOR and BERTScore.
  - METEOR was also designed primarily for English (it leans on
    WordNet synonym matching and, in nltk's implementation, an English
    Porter stemmer by default, both far weaker for Spanish). It is
    still computed since the paper's Experimental Design specifies it
    for open-ended questions, but scores should be read as approximate,
    not as precisely calibrated for Spanish — worth a caveat in the
    paper's methodology or limitations section.
"""

import os
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------
# Binary: classification F1
# ---------------------------------------------------------------------

def extract_binary_label(text):
    """
    Extracts a 'sí' / 'no' label from a model's free-text answer. Looks
    at the first few words rather than requiring an exact single-token
    answer, since models sometimes prepend a label before elaborating
    (e.g. "No, el curso no tiene requisitos...").

    Returns "si", "no", or None if no clear label could be extracted
    (counted as a wrong/unparseable answer during aggregation, not
    silently dropped).
    """
    if not isinstance(text, str) or not text.strip():
        return None

    normalized = strip_accents(text.strip().lower())
    normalized = normalized.lstrip(string.punctuation + " ")

    # Check the first word only — "sí, ..." / "no, ..." / "si." / "no"
    first_word = re.split(r"[\s,\.;:!]", normalized, maxsplit=1)[0]

    if first_word in ("si", "sí"):
        return "si"
    if first_word == "no":
        return "no"
    return None


def binary_classification_f1(pairs):
    """
    pairs: list of (predicted_text, expected_text) tuples.

    Returns a dict with precision/recall/F1 (macro-averaged over the two
    classes) plus the count of unparseable predictions, which are scored
    as wrong (neither class) rather than excluded.
    """
    tp = {"si": 0, "no": 0}
    fp = {"si": 0, "no": 0}
    fn = {"si": 0, "no": 0}
    n_unparseable = 0

    for pred_text, expected_text in pairs:
        gold = extract_binary_label(expected_text)
        pred = extract_binary_label(pred_text)

        if gold is None:
            continue  # shouldn't happen — expected answers are curated Sí/No

        if pred is None:
            n_unparseable += 1
            fn[gold] += 1
            continue

        if pred == gold:
            tp[pred] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    per_class_f1 = {}
    for label in ("si", "no"):
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1[label] = {"precision": precision, "recall": recall, "f1": f1}

    macro_f1 = (per_class_f1["si"]["f1"] + per_class_f1["no"]["f1"]) / 2

    return {
        "macro_f1": macro_f1,
        "per_class": per_class_f1,
        "n_unparseable": n_unparseable,
        "n_total": len(pairs),
    }


# ---------------------------------------------------------------------
# Short answer: token-overlap F1 (SQuAD-style)
# ---------------------------------------------------------------------

def strip_accents(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_text(text):
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = strip_accents(text)
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_f1(predicted, expected):
    """
    SQuAD-style token-overlap F1 between two free-text answers. Returns
    0.0 if there is no overlap at all (including when either string is
    empty after normalization).
    """
    pred_tokens = normalize_text(predicted).split()
    gold_tokens = normalize_text(expected).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0

    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------
# BERTScore (short_answer + open_ended) and METEOR (open_ended only)
# ---------------------------------------------------------------------
# Loaded lazily (only when requested via evaluate_results.py's
# skip_open_ended flag) — these pull in heavier dependencies
# (bert_score, nltk) and a Spanish BERT model download, no need to pay
# that cost for a run that only needs binary's classification F1 and
# short_answer's token-overlap F1.

_bertscore_model = None
_nltk_ready = False


def _ensure_nltk_data():
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk

    # NLTK defaults to ~/nltk_data — a completely separate cache path
    # from HF_HOME, unaffected by that env var. On a cluster with a
    # tight home-directory quota this fails outright ("Disk quota
    # exceeded"), so redirect it alongside HF_HOME instead. Falls back
    # to nltk's default only if HF_HOME isn't set either.
    nltk_dir = os.environ.get("NLTK_DATA")
    if not nltk_dir:
        hf_home = os.environ.get("HF_HOME")
        nltk_dir = str(Path(hf_home).parent / "nltk_data") if hf_home else str(Path.home() / "nltk_data")
    os.makedirs(nltk_dir, exist_ok=True)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_dir)

    # wordnet/omw-1.4 live under corpora/, punkt/punkt_tab under
    # tokenizers/ — the original code checked all four under corpora/,
    # which meant punkt/punkt_tab always looked "missing" and re-
    # attempted downloading every run even when already cached.
    resource_paths = {
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }
    for resource, check_path in resource_paths.items():
        try:
            nltk.data.find(check_path)
        except LookupError:
            nltk.download(resource, download_dir=nltk_dir, quiet=True)
    _nltk_ready = True


def compute_bertscore(predictions, references, lang_model="dccuchile/bert-base-spanish-wwm-cased", num_layers=10):
    """
    predictions, references: parallel lists of strings.
    Returns lists of precision/recall/F1 (one per pair), using BETO
    (Spanish BERT) rather than a multilingual model, since the
    documents and expected answers are in Spanish.

    num_layers: bert_score's internal model2layers table only has
    empirically-calibrated (via WMT16 correlation data) layer choices
    for a fixed set of well-known checkpoints; BETO isn't among them, so
    this must be passed explicitly or the library raises KeyError.
    10 (of BETO's 12 layers) follows the common convention of using a
    late-but-not-final layer for semantic similarity (the final layer
    tends to be more pretraining-objective-specific, less semantic) —
    this is a reasonable default, NOT an empirically-tuned choice for
    Spanish/BETO the way bert_score's built-in models are. Worth noting
    as a methodology caveat in the paper rather than presenting
    BERTScore numbers as precisely calibrated.
    """
    from bert_score import score as bertscore_score

    P, R, F1 = bertscore_score(
        predictions, references, model_type=lang_model, num_layers=num_layers,
        lang="es", verbose=False,
    )
    return {
        "precision": P.tolist(),
        "recall": R.tolist(),
        "f1": F1.tolist(),
    }


def compute_meteor(predictions, references):
    """
    Returns a list of METEOR scores, one per pair. Uses nltk's
    implementation — designed primarily for English (WordNet-based
    synonym matching), so treat Spanish scores as approximate.
    """
    _ensure_nltk_data()
    from nltk.translate.meteor_score import meteor_score
    from nltk.tokenize import word_tokenize

    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = word_tokenize(pred.lower()) if isinstance(pred, str) else []
        ref_tokens = word_tokenize(ref.lower()) if isinstance(ref, str) else []
        if not pred_tokens or not ref_tokens:
            scores.append(0.0)
            continue
        scores.append(meteor_score([ref_tokens], pred_tokens))
    return scores