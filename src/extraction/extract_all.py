"""
Batch-extracts every PDF/DOCX in data/raw_docs into plain .txt files in
data/extracted_text, using files_to_text.extract_text().

Usage:
    python extract_all.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from files_to_text import extract_text

RAW_DOCS_DIR = os.path.join("data", "raw_docs")
EXTRACTED_TEXT_DIR = os.path.join("data", "extracted_text")

SUPPORTED_EXTENSIONS = (".pdf", ".docx")


def extract_all(raw_dir=RAW_DOCS_DIR, out_dir=EXTRACTED_TEXT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(raw_dir) if f.lower().endswith(SUPPORTED_EXTENSIONS))

    if not files:
        print(f"No .pdf/.docx files found in {raw_dir}")
        return

    print(f"Found {len(files)} file(s) in {raw_dir}\n")

    succeeded, failed = [], []

    for filename in files:
        in_path = os.path.join(raw_dir, filename)
        out_filename = os.path.splitext(filename)[0] + ".txt"
        out_path = os.path.join(out_dir, out_filename)

        try:
            text = extract_text(in_path)
        except Exception as e:
            print(f"  FAILED: {filename} -> {type(e).__name__}: {e}")
            failed.append(filename)
            continue

        if not text.strip():
            print(f"  WARNING: {filename} extracted empty text (scanned PDF with no text layer?)")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"  OK: {filename} -> {out_filename} ({len(text)} chars)")
        succeeded.append(filename)

    print(f"\nDone: {len(succeeded)} succeeded, {len(failed)} failed.")
    if failed:
        print("Failed files:", ", ".join(failed))


if __name__ == "__main__":
    extract_all()