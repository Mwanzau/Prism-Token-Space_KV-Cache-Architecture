"""Section-aware, content-addressable metadata for Prism artifacts.

The binary Prism index stays intentionally token-only.  This sidecar provides
the library taxonomy required for routing without adding JSON parsing to the
hot retrieval path.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

LIBRARY_VERSION = 1
_MARKDOWN_HEADING = re.compile(r"^#{1,3}\s+(.+?)\s*#*$")
_UPPERCASE_HEADING = re.compile(r"^[A-Z][A-Z0-9 ,:;’'\-]{3,100}$")
_WORD = re.compile(r"\w+")


def library_sidecar_path(prism_path: str | Path) -> Path:
    path = Path(prism_path)
    return path.with_suffix(path.suffix + ".library.json")


def normalize_section_text(text: str) -> str:
    return " ".join(text.lower().split())


def section_hash(text: str) -> str:
    return hashlib.sha256(normalize_section_text(text).encode("utf-8")).hexdigest()


def section_id_for(title: str, content_hash: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "untitled"
    return f"SECTION_{slug.upper()}_{content_hash[:12]}"


class LibraryDocumentParser:
    """Conservative plain-text section parser for Markdown and book-like files."""

    @staticmethod
    def _heading(line: str) -> Optional[str]:
        markdown = _MARKDOWN_HEADING.match(line)
        if markdown:
            return markdown.group(1).strip()
        # All-uppercase headings are common in Gutenberg texts.  Avoid short
        # shouting/dialogue lines by requiring no terminal sentence punctuation.
        if _UPPERCASE_HEADING.match(line) and not line.endswith((".", "!", "?")):
            return line.strip().title()
        return None

    def parse_document(self, raw_text: str, doc_name: str) -> List[Dict[str, str]]:
        sections: List[Dict[str, str]] = []
        title = "Preamble"
        lines: List[str] = []

        def commit() -> None:
            content = "\n".join(lines).strip()
            if not content:
                return
            digest = section_hash(content)
            sections.append({
                "doc_id": doc_name,
                "section_title": title,
                "section_hash": digest,
                "section_id": section_id_for(title, digest),
                "content": content,
            })

        for raw_line in raw_text.splitlines():
            candidate = self._heading(raw_line.strip())
            if candidate:
                commit()
                lines = []
                title = candidate
            else:
                lines.append(raw_line)
        commit()
        return sections


def empty_library(model_vocab_hash: int) -> Dict[str, Any]:
    return {
        "version": LIBRARY_VERSION,
        "model_vocab_hash": model_vocab_hash,
        "documents": {},
        "sections": {},
        "chunks": {},
    }


def load_library(prism_path: str | Path) -> Optional[Dict[str, Any]]:
    path = library_sidecar_path(prism_path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("version") != LIBRARY_VERSION:
        raise ValueError(f"Unsupported Prism library metadata version in {path}")
    return data


def save_library(prism_path: str | Path, library: Dict[str, Any]) -> Path:
    path = library_sidecar_path(prism_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(library, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def add_document_alias(library: Dict[str, Any], document_id: str, section: Dict[str, str]) -> None:
    document = library["documents"].setdefault(document_id, {"section_hashes": []})
    if section["section_hash"] not in document["section_hashes"]:
        document["section_hashes"].append(section["section_hash"])
    canonical = library["sections"].setdefault(section["section_hash"], {
        "section_id": section["section_id"],
        "section_title": section["section_title"],
        "source_documents": [],
        "chunk_ids": [],
    })
    if document_id not in canonical["source_documents"]:
        canonical["source_documents"].append(document_id)


def route_section(
    query: str,
    chunk_scores: Dict[int, float],
    library: Dict[str, Any],
) -> Tuple[Optional[str], Set[int], float]:
    """Choose one section before the second, scoped token-space retrieval pass."""
    query_words = {word.lower() for word in _WORD.findall(query) if len(word) > 2}
    section_scores: Dict[str, float] = {}
    for chunk_id, score in chunk_scores.items():
        metadata = library["chunks"].get(str(chunk_id))
        if metadata:
            digest = metadata["section_hash"]
            section_scores[digest] = section_scores.get(digest, 0.0) + score

    for digest, section in library["sections"].items():
        title_words = {word.lower() for word in _WORD.findall(section["section_title"])}
        # Titles are a routing signal, not a replacement for body evidence.
        title_overlap = len(query_words & title_words)
        section_scores[digest] = section_scores.get(digest, 0.0) + 4.0 * title_overlap

    if not section_scores:
        return None, set(), 0.0
    winning_hash, score = max(section_scores.items(), key=lambda item: (item[1], item[0]))
    chunk_ids = {int(chunk_id) for chunk_id in library["sections"][winning_hash]["chunk_ids"]}
    return winning_hash, chunk_ids, score
