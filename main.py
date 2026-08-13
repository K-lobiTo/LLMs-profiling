from utils.files_to_text import corpus, Document
# from docx import Document

print(len(corpus))

docIdx = 0
print(corpus[list(corpus.keys())[docIdx]])

doc = Document("files/" + list(corpus.keys())[docIdx])
print("Paragraphs:", len(doc.paragraphs))
print("Tables:", len(doc.tables))