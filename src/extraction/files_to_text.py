import pymupdf as fitz 
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

def get_list_level(paragraph):
    """Returns the indent level (int) if paragraph is a list item, else None."""
    pPr = paragraph._p.pPr
    if pPr is None:
        return None
    numPr = pPr.find(qn('w:numPr'))
    if numPr is None:
        return None
    ilvl = numPr.find(qn('w:ilvl'))
    return int(ilvl.get(qn('w:val'))) if ilvl is not None else 0

def paragraph_to_text(paragraph):
    text = paragraph.text.strip()
    if not text:
        return None
    level = get_list_level(paragraph)
    if level is not None:
        return ("  " * level) + "- " + text
    return text

def iter_block_items(parent):
    parent_elm = parent.element.body if hasattr(parent, "element") else parent._tc
    for child in parent_elm.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)

def docx_to_text(path):
    doc = Document(path)
    text_parts = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            line = paragraph_to_text(block)
            if line:
                text_parts.append(line)
        elif isinstance(block, Table):
            for row in block.rows:
                for cell in row.cells:
                    for cell_block in iter_block_items(cell):
                        if isinstance(cell_block, Paragraph):
                            line = paragraph_to_text(cell_block)
                            if line:
                                text_parts.append(line)
                        elif isinstance(cell_block, Table):
                            # nested table, rare but handle just in case
                            for nrow in cell_block.rows:
                                for ncell in nrow.cells:
                                    for p in ncell.paragraphs:
                                        line = paragraph_to_text(p)
                                        if line:
                                            text_parts.append(line)

    return "\n".join(text_parts)


def extract_text(path):
    if path.endswith(".pdf"):
        return pdf_to_text(path)
    elif path.endswith(".docx"):
        return docx_to_text(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


if __name__ == "__main__":
    # Ad hoc standalone run, kept for reference — the real batch extraction
    # for this project is handled by extract_all.py instead.
    corpus = {f: extract_text(os.path.join("files/", f)) for f in os.listdir("files/")}