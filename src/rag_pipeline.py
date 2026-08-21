"""
RAG pipeline: chunking + embedding + retrieval.

Generation is intentionally left as a placeholder — plug in
Qwen2.5-3B-Instruct / Llama-3.2-3B / DeepSeek-R1 later via `generate_fn`.
Everything else (chunking, embedding, retrieval) stays identical across
models, so the generator is the only variable in the benchmark.
"""

import os
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# 1. Chunking
# ---------------------------------------------------------------------

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks by word count.

    chunk_size: approx. words per chunk
    overlap: words shared between consecutive chunks (helps avoid
             cutting relevant info at a chunk boundary)
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def chunk_documents(doc_texts, chunk_size=500, overlap=50):
    """
    doc_texts: dict of {filename: raw_text}, e.g. output of your
               pdf_to_text / docx_to_text extraction step.

    Returns: list of dicts, one per chunk:
             {"text": ..., "source": filename, "chunk_id": i}

    Each chunk's embedded text is prefixed with the document name. This
    matters here specifically: all 15 source documents share near-
    identical boilerplate structure ("Asistencia: Obligatoria",
    "Requisitos: Ninguno", etc.), so without a course identifier baked
    into the embedded text, the same generic line from two different
    courses embeds almost indistinguishably — causing retrieval to pull
    the wrong course's chunk. The prefix gives the embedding model a
    consistent, course-specific anchor to differentiate on.
    """
    all_chunks = []
    for filename, text in doc_texts.items():
        doc_title = Path(filename).stem  # drop .txt extension
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            prefixed_text = f"[Documento: {doc_title}]\n{chunk}"
            all_chunks.append({
                "text": prefixed_text,
                "source": filename,
                "chunk_id": i,
            })
    return all_chunks


# ---------------------------------------------------------------------
# 2. Embedding + vector index
# ---------------------------------------------------------------------

class RAGIndex:
    """
    Wraps a SentenceTransformer embedder + a FAISS index over document
    chunks. Default embedding model is multilingual, since your source
    documents (course programs) are in Spanish.
    """

    def __init__(self, embedding_model="paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []  # parallel list: self.chunks[i] <-> vector i in index

    def build(self, chunks):
        """
        chunks: list of dicts from chunk_documents(), each with a "text" key.
        """
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine sim, since normalized
        self.index.add(embeddings)

    def retrieve(self, query, top_k=5, max_per_source=2):
        """
        Returns top_k chunk dicts most relevant to the query, each with
        an added "score" key.

        max_per_source: caps how many chunks from the same source
        document can appear in the results, keeping the correct ranking
        order but skipping extra chunks once a source hits the cap. This
        is a standard retrieval-diversity technique — purely based on
        ranking, uses no ground-truth information about which document
        "should" be relevant — that prevents one verbose/generic
        document from crowding out all top_k slots with its own
        near-duplicate overlapping chunks, at the expense of other
        genuinely relevant documents. Set to None to disable.
        """
        # Over-fetch from FAISS since some results may get filtered by
        # the per-source cap; fetch more candidates than top_k to have
        # enough left after filtering.
        fetch_k = top_k * 4 if max_per_source is not None else top_k

        query_vec = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )
        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        source_counts = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])

            if max_per_source is not None:
                count = source_counts.get(chunk["source"], 0)
                if count >= max_per_source:
                    continue
                source_counts[chunk["source"]] = count + 1

            chunk["score"] = float(score)
            results.append(chunk)

            if len(results) >= top_k:
                break

        return results

    def save(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, str(Path(dir_path) / "index.faiss"))
        np.save(Path(dir_path) / "chunks.npy", np.array(self.chunks, dtype=object))

    def load(self, dir_path):
        self.index = faiss.read_index(str(Path(dir_path) / "index.faiss"))
        self.chunks = np.load(Path(dir_path) / "chunks.npy", allow_pickle=True).tolist()


# ---------------------------------------------------------------------
# 3. Prompt construction
# ---------------------------------------------------------------------

def build_prompt(question, retrieved_chunks):
    """
    Builds a simple context + question prompt. Chat-template formatting
    (per model) is applied later, not here — this is just the raw content.
    """
    context = "\n\n".join(
        f"[Fuente: {c['source']}]\n{c['text']}" for c in retrieved_chunks
    )
    prompt = (
        "Responde la siguiente pregunta usando únicamente la información "
        "del contexto proporcionado. Si la respuesta no está en el "
        "contexto, indica que no tienes suficiente información.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta: {question}\n"
        "Respuesta:"
    )
    return prompt


# ---------------------------------------------------------------------
# 4. Generation placeholder — plug in each model here later
# ---------------------------------------------------------------------

def generate_answer(question, rag_index, top_k=5, generate_fn=None):
    """
    Full RAG call: retrieve relevant chunks, build prompt, generate answer.

    generate_fn: a function (prompt: str) -> str, model-specific.
                 Left as None for now — once models are wired in, pass e.g.
                 generate_fn=qwen_generate / llama_generate / deepseek_generate.
    """
    retrieved = rag_index.retrieve(question, top_k=top_k)
    prompt = build_prompt(question, retrieved)

    if generate_fn is None:
        # Placeholder: no model wired in yet, return the prompt itself
        # so you can inspect what would be sent to the LLM.
        return {"prompt": prompt, "retrieved": retrieved, "answer": None}

    answer = generate_fn(prompt)
    return {"prompt": prompt, "retrieved": retrieved, "answer": answer}


if __name__ == "__main__":
    sample_docs = {
        "curso_cibercrimen.txt": (
            "Programa del curso MC3010 Cibercrimen. "
            "La asistencia es obligatoria. El curso tiene 4 créditos y "
            "no permite suficiencia. La evaluación consiste en pruebas "
            "cortas (20%), investigación (40%) y estudio de casos (40%)."
        ),
    }

    chunks = chunk_documents(sample_docs, chunk_size=30, overlap=5)
    print(f"Generated {len(chunks)} chunks")

    rag = RAGIndex()
    rag.build(chunks)

    result = generate_answer("¿Es la asistencia obligatoria?", rag, top_k=2)
    print("\n--- Prompt that would be sent to the model ---")
    print(result["prompt"])