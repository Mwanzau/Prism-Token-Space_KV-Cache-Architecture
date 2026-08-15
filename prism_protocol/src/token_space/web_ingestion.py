"""Application-layer web ingestion with exact and near-duplicate suppression."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import urlparse

from .importer import build_prism_from_text
from .library import load_library, save_library

_PREFERRED_DOMAINS = (
    "who.int", "cdc.gov", "nih.gov", "hiv.gov", "gov", "edu",
    "nhs.uk", "mayoclinic.org", "clevelandclinic.org",
)


class WebKnowledgeDeduplicator:
    """Content fingerprinting independent of network fetching and Prism storage."""

    def __init__(self, similarity_threshold: float = 0.75, library: Optional[Dict[str, Any]] = None):
        self.similarity_threshold = similarity_threshold
        self.exact_to_chunk_ids: Dict[str, List[str]] = defaultdict(list)
        self.shingles: Dict[str, Set[str]] = {}
        if library:
            for chunk_id, metadata in library.get("chunks", {}).items():
                text = metadata.get("text", "")
                if text:
                    digest = self.content_hash(text)
                    self.exact_to_chunk_ids[digest].append(chunk_id)
                    self.shingles[digest] = self.ngrams(text)

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())

    @classmethod
    def content_hash(cls, text: str) -> str:
        return hashlib.sha256(cls.normalize(text).encode("utf-8")).hexdigest()

    @classmethod
    def ngrams(cls, text: str, n: int = 3) -> Set[str]:
        words = cls.normalize(text).split()
        return set(words) if len(words) < n else {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

    @staticmethod
    def jaccard(first: Set[str], second: Set[str]) -> float:
        return len(first & second) / len(first | second) if first and second else 0.0

    def process(self, sources: Iterable[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """Return unique paragraphs and duplicate links for provenance updates."""
        unique: List[Dict[str, str]] = []
        duplicates: List[Dict[str, Any]] = []
        for source in sources:
            url, content = source["source_url"], source["content"]
            paragraphs = [line.strip() for line in content.splitlines() if len(line.strip()) >= 40]
            for paragraph in paragraphs:
                digest = self.content_hash(paragraph)
                if digest in self.exact_to_chunk_ids:
                    duplicates.append({"source_url": url, "hash": digest, "kind": "exact", "chunk_ids": self.exact_to_chunk_ids[digest]})
                    continue
                shingles = self.ngrams(paragraph)
                near = next(
                    ((old_hash, self.jaccard(shingles, old_shingles)) for old_hash, old_shingles in self.shingles.items()
                     if self.jaccard(shingles, old_shingles) >= self.similarity_threshold),
                    None,
                )
                if near:
                    duplicates.append({"source_url": url, "hash": near[0], "kind": "near", "similarity": near[1], "chunk_ids": self.exact_to_chunk_ids[near[0]]})
                    continue
                self.exact_to_chunk_ids[digest].append(f"pending:{len(unique)}")
                self.shingles[digest] = shingles
                unique.append({"source_url": url, "content": paragraph})
        return unique, duplicates


def fetch_clean_url(url: str) -> Dict[str, str]:
    """Fetch main article text; trafilatura deliberately remains outside Prism core."""
    try:
        import trafilatura
    except ImportError as exc:
        raise RuntimeError("Web ingestion requires trafilatura; install project requirements first.") from exc
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded or "", include_links=False, include_formatting=False, output_format="txt")
    if not text:
        raise ValueError(f"No readable article content extracted from {url}")
    return {"source_url": url, "content": text}


def _source_rank(url: str) -> int:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    for index, domain in enumerate(_PREFERRED_DOMAINS):
        if host == domain or host.endswith("." + domain):
            return index
    return len(_PREFERRED_DOMAINS)


def discover_and_fetch(query: str, max_sources: int = 5) -> tuple[List[Dict[str, str]], List[str]]:
    """Search the web, favor high-trust sources, and return extractable articles.

    Failed/blocked sites are deliberately skipped; their URLs are returned for
    diagnostics rather than causing a whole autonomous query to fail.
    """
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError("Autonomous web search requires ddgs; install project requirements first.") from exc
    try:
        results = list(DDGS().text(query, max_results=max_sources * 3))
    except Exception as exc:
        raise RuntimeError(f"Web search failed: {exc}") from exc
    urls = []
    for result in results:
        url = result.get("href") or result.get("url")
        if url and url.startswith(("https://", "http://")) and url not in urls:
            urls.append(url)
    urls.sort(key=_source_rank)

    sources: List[Dict[str, str]] = []
    failures: List[str] = []
    for url in urls:
        if len(sources) >= max_sources:
            break
        try:
            sources.append(fetch_clean_url(url))
        except (RuntimeError, ValueError):
            failures.append(url)
    if not sources:
        raise ValueError("Search returned pages, but none yielded readable article text.")
    return sources, failures


def ingest_web_sources(
    sources: Iterable[Dict[str, str]], gguf_path: str, prism_path: str, *, chunk_size: int = 150,
    similarity_threshold: float = 0.75,
) -> Dict[str, int]:
    """Append only new web paragraphs and link duplicate provenance to canonical chunks."""
    library = load_library(prism_path)
    if library is None:
        raise ValueError("Build a section-aware Prism artifact before web ingestion.")
    deduper = WebKnowledgeDeduplicator(similarity_threshold, library)
    unique, duplicates = deduper.process(sources)
    for duplicate in duplicates:
        for chunk_id in duplicate["chunk_ids"]:
            if chunk_id.startswith("pending:"):
                continue
            metadata = library["chunks"].get(chunk_id, {})
            urls = metadata.setdefault("source_urls", [])
            if duplicate["source_url"] not in urls:
                urls.append(duplicate["source_url"])
    save_library(prism_path, library)

    by_url: Dict[str, List[str]] = defaultdict(list)
    for item in unique:
        by_url[item["source_url"]].append(item["content"])
    for url, paragraphs in by_url.items():
        host = urlparse(url).netloc or "web-source"
        text = f"# Web Article: {host}\n\n" + "\n\n".join(paragraphs)
        build_prism_from_text(
            text, f"web:{url}", gguf_path, prism_path, chunk_size=chunk_size, append=True, source_url=url
        )
    # Resolve both pre-existing and same-batch duplicate hashes after new
    # chunks have received stable IDs.
    final_library = load_library(prism_path)
    if final_library:
        final_hashes: Dict[str, List[str]] = defaultdict(list)
        for chunk_id, metadata in final_library["chunks"].items():
            if metadata.get("text"):
                final_hashes[deduper.content_hash(metadata["text"])].append(chunk_id)
        for duplicate in duplicates:
            for chunk_id in final_hashes.get(duplicate["hash"], []):
                urls = final_library["chunks"][chunk_id].setdefault("source_urls", [])
                if duplicate["source_url"] not in urls:
                    urls.append(duplicate["source_url"])
        save_library(prism_path, final_library)
    return {"unique_paragraphs": len(unique), "duplicate_paragraphs": len(duplicates), "sources": len(by_url)}
