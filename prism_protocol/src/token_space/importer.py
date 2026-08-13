from pathlib import Path
from typing import Optional
import shutil
import subprocess
import tempfile

from .binary_format import PrismWriter
from .indexer import PrismIndexer
from .tokenizer_bridge import TokenizerBridge
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
    tokens = tokenizer.encode(text, progress_callback=progress_callback)
    if not tokens:
        raise ValueError("No tokens were produced from document text")

    chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    index_entries = PrismWriter.build_index(chunks)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if append:
        PrismWriter.append_prism(output_path, chunks, tokenizer.model_vocab_hash, index_entries)
    else:
        PrismWriter.write_prism(output_path, chunks, tokenizer.model_vocab_hash, index_entries)
    return output_path
