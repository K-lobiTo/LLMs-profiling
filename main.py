from utils.files_to_text import corpus, Document, qn

# print(len(corpus))

docIdx = 3
print(corpus[list(corpus.keys())[docIdx]])

path = "files/" + list(corpus.keys())[docIdx]
doc = Document(path)
# for p in doc.paragraphs:
#     if p.text.strip():
#         print(repr(p.text))
#         print("style:", p.style.name if p.style else None)
#         pPr = p._p.pPr
#         print("direct numPr:", pPr.find(qn('w:numPr')) if pPr is not None else None)
#         print("---")

# print("Total doc.paragraphs:", len(doc.paragraphs))
# print("Total doc.tables:", len(doc.tables))

# for t_idx, table in enumerate(doc.tables):
#     for r_idx, row in enumerate(table.rows):
#         for c_idx, cell in enumerate(row.cells):
#             for p in cell.paragraphs:
#                 if p.text.strip():
#                     print(f"[table {t_idx} row {r_idx} cell {c_idx}]", repr(p.text))
#                     print("  style:", p.style.name if p.style else None)
#                     pPr = p._p.pPr
#                     from docx.oxml.ns import qn
#                     print("  direct numPr:", pPr.find(qn('w:numPr')) if pPr is not None else None)

# print("Paragraphs:", len(doc.paragraphs))
# print("Tables:", len(doc.tables))