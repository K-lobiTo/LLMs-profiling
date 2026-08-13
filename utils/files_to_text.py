import fitz  # pymupdf
from docx import Document
import os

def pdf_to_text(path):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def docx_to_text(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(path):
    if path.endswith(".pdf"):
        return pdf_to_text(path)
    elif path.endswith(".docx"):
        return docx_to_text(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

corpus = {f: extract_text(os.path.join("files/", f)) for f in os.listdir("files/")}