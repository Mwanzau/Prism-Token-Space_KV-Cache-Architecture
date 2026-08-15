from pathlib import Path
from typing import Optional
import shutil
import subprocess
import tempfile

from .binary_format import PrismWriter
from .binary_format import PrismReader
from .library import (
    LibraryDocumentParser,
    add_document_alias,
    empty_library,
    load_library,
    narrative_multiplier,
    save_library,
)
from .tokenizer_adapter import get_tokenizer_adapter


def _load_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            PdfReader = None

    if PdfReader is not None:
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise ImportError(
            "PDF extraction requires pypdf, PyPDF2, or pdftotext. Install one of these packages or tools."
        )

    result = subprocess.run(
        [pdftotext, path, "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def extract_text_from_file(path: str) -> str:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()
    if suffix == ".txt":
        return path_obj.read_text(errors="ignore")
    if suffix == ".pdf":
        return _load_pdf_text(path)
    raise ValueError(f"Unsupported document format: {suffix}")


def build_prism_from_document(
    document_path: str,
    gguf_path: str,
    output_path: str,
    chunk_size: int = 128,
    append: bool = True,
    progress_callback: Optional[callable] = None,
) -> str:
    text = extract_text_from_file(document_path)
    if not text.strip():
        raise ValueError("Document contains no extractable text")

    tokenizer = get_tokenizer_adapter(gguf_path)
    output = Path(output_path)
    existing_chunk_count = 0
    if append and output.exists() and output.stat().st_size > 0:
        existing = PrismReader(str(output))
        existing.header.validate_vocab(tokenizer.model_vocab_hash)
        existing_chunk_count = existing.header.chunk_count

    library = load_library(output) if append else None
    if append and existing_chunk_count and library is None:
        raise ValueError(
            "Existing Prism artifact has no library metadata sidecar. Rebuild it with --overwrite "
            "to enable section-scoped retrieval."
        )
    if library is None:
        library = empty_library(tokenizer.model_vocab_hash)
    elif library.get("model_vocab_hash") != tokenizer.model_vocab_hash:
        raise ValueError("Library metadata tokenizer does not match the supplied GGUF/tokenizer.")

    parser = LibraryDocumentParser()
    sections = parser.parse_document(text, Path(document_path).name)
    if not sections:
        raise ValueError("Document contains no indexable sections")

    chunks = []
    next_chunk_id = existing_chunk_count
    for ordinal, section in enumerate(sections, start=1):
        digest = section["section_hash"]
        already_indexed = digest in library["sections"]
        add_document_alias(library, section["doc_id"], section)
        if already_indexed:
            continue

        tokens = tokenizer.encode(section["content"])
        if not tokens:
            continue
        for offset in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[offset : offset + chunk_size]
            chunk_id = next_chunk_id + len(chunks)
            chunks.append(chunk_tokens)
            library["chunks"][str(chunk_id)] = {
                "doc_id": section["doc_id"],
                "section_id": section["section_id"],
                "section_hash": digest,
                "section_title": section["section_title"],
                "chunk_index": offset // chunk_size,
                "narrative_multiplier": narrative_multiplier(tokenizer.decode(chunk_tokens)),
                # Retain readable audit metadata outside the compact binary
                # token stream; retrieval itself never reads this field.
                "text": tokenizer.decode(chunk_tokens),
            }
            library["sections"][digest]["chunk_ids"].append(chunk_id)
        if progress_callback:
            progress_callback(ordinal, len(sections))

    if not chunks and existing_chunk_count == 0:
        raise ValueError("No tokens were produced from document text")
    index_entries = PrismWriter.build_index(chunks)

    output.parent.mkdir(parents=True, exist_ok=True)
    if chunks:
        if append:
            PrismWriter.append_prism(output_path, chunks, tokenizer.model_vocab_hash, index_entries)
        else:
            PrismWriter.write_prism(output_path, chunks, tokenizer.model_vocab_hash, index_entries)
    save_library(output, library)
    return output_path
