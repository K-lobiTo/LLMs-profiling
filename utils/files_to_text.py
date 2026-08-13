import fitz  # pymupdf
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

import os

def pdf_to_text(path):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

def docx_to_text(path):
    doc = Document(path)
    text_parts = []

    def iter_block_items(parent):
        parent_elm = parent.element.body
        for child in parent_elm.iterchildren():
            if child.tag == qn('w:p'):
                yield Paragraph(child, parent)
            elif child.tag == qn('w:tbl'):
                yield Table(child, parent)

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            if block.text.strip():
                text_parts.append(block.text)
        elif isinstance(block, Table):
            for row in block.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                text_parts.append(row_text)

    return "\n".join(text_parts)


def extract_text(path):
    if path.endswith(".pdf"):
        return pdf_to_text(path)
    elif path.endswith(".docx"):
        return docx_to_text(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

corpus = {f: extract_text(os.path.join("files/", f)) for f in os.listdir("files/")}