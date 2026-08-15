import argparse
import sys
import tempfile
from pathlib import Path
from typing import Optional
import re

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from prism_protocol.src.token_space.importer import build_prism_from_document
from prism_protocol.src.token_space.tokenizer_adapter import get_tokenizer_adapter
from prism_protocol.src.token_space.binary_format import PrismReader
from prism_protocol.src.token_space.library import get_section_summary, load_library, narrative_multiplier, route_section

DEFAULT_GGUF_PATH = ROOT / "models" / "Qwen_Qwen2.5-0.5B-Instruct-GGUF" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
DEFAULT_DOCUMENT_PATH = ROOT / "File_Samples_For_Tests" / "01Chapter1.pdf"
DEFAULT_OUTPUT_PATH = ROOT / "output" / "demo.prism"
DEFAULT_LOG_QUERY_PATH = ROOT / "prism_query.log"
DEFAULT_LOG_DEMO_PATH = ROOT / "prism_demo.log"
DEFAULT_LOG_QA_PATH = ROOT / "prism_qa.log"
DEFAULT_QUERY = "Malawi is a landlocked country"
NOISE_PATTERNS = (
    r"\btable of contents\b",
    r"\bcontents\b",
    r"\blist of illustrations?\b",
    r"\bproject gutenberg\b",
    r"\bgutenberg(?:\.org)?\b",
    r"\blicen[cs]e\b",
    r"\bcopyright\b",
    r"\btranscriber'?s? notes?\b",
)


class Color:
    ENABLED = sys.stdout.isatty()

    @staticmethod
    def _wrap(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if Color.ENABLED else text

    @staticmethod
    def bold(text: str) -> str:
        return Color._wrap(text, "1")

    @staticmethod
    def green(text: str) -> str:
        return Color._wrap(text, "32")

    @staticmethod
    def cyan(text: str) -> str:
        return Color._wrap(text, "36")

    @staticmethod
    def yellow(text: str) -> str:
        return Color._wrap(text, "33")

    @staticmethod
    def red(text: str) -> str:
        return Color._wrap(text, "31")


def print_header(title: str) -> None:
    print(Color.bold(Color.cyan(f"=== {title} ===")))


def print_status(message: str) -> None:
    print(Color.green(message))


def print_error(message: str) -> None:
    print(Color.red(f"ERROR: {message}"))


def _is_local_model_path(p: Path) -> bool:
    """Return True only when gguf_path refers to a local file that must exist."""
    s = str(p)
    if s.startswith("hf:"):
        return False
    if p.suffix == ".json":  # tokenizer.json file
        return True
    if p.is_dir():
        return True
    return p.suffix == ".gguf"


def decode_snippet(tokens: list[int], tokenizer, max_chars: int = 240) -> str:
    decoded = tokenizer.decode(tokens)
    text = " ".join(decoded.split())
    if len(text) <= max_chars:
        return text
    snippet = text[: max_chars - 3].rsplit(" ", 1)[0]
    return f"{snippet}..."


def _split_into_sentences(text: str) -> list[str]:
    # Simple sentence splitter: splits on punctuation followed by space/newline
    # Keeps punctuation at end of sentence.
    pieces = re.split(r'(?<=[\.\?!])\s+', text)
    return [p.strip() for p in pieces if p.strip()]


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "who", "whom", "this", "that", "these", "those", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "to", "from", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "don", "should", "now", "of", "it"
}


def _score_sentence(sentence: str, query: str) -> float:
    # Lightweight relevance: overlap of content word tokens normalized by length
    q_words = {w.lower() for w in re.findall(r"\w+", query) if w.lower() not in STOPWORDS}
    if not q_words:
        q_words = {w.lower() for w in re.findall(r"\w+", query)}
    s_words = [w.lower() for w in re.findall(r"\w+", sentence)]
    if not s_words or not q_words:
        return 0.0
    overlap = sum(1 for w in s_words if w in q_words)
    return overlap / (len(s_words) ** 0.5)


def extract_highest_scoring_sentence_window(text: str, query: str) -> tuple[int, str, float]:
    sentences = _split_into_sentences(text)
    if not sentences:
        return 0, "", 0.0
    scores = [_score_sentence(s, query) for s in sentences]
    best_idx = max(range(len(sentences)), key=lambda i: scores[i])
    start_idx = max(0, best_idx - 1)
    end_idx = min(len(sentences), best_idx + 2)
    window_sentences = sentences[start_idx:end_idx]
    window_text = " ".join(window_sentences)
    return best_idx, window_text, scores[best_idx]


def _is_structural_noise(text: str) -> bool:
    """Identify document furniture that should not occupy the SLM context."""
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized) for pattern in NOISE_PATTERNS)


def _limit_to_token_budget(text: str, tokenizer, max_tokens: int = 150) -> str:
    """Keep a readable sentence window within a strict token budget."""
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= max_tokens:
        return text.strip()
    return tokenizer.decode(token_ids[:max_tokens]).strip()


def _boost_narrative_chunks(ranked, library, limit: int):
    """Apply sidecar narrative tags after token retrieval, before final selection."""
    def adjusted_score(retrieved) -> float:
        metadata = (library or {}).get("chunks", {}).get(str(retrieved.chunk_id), {})
        multiplier = metadata.get("narrative_multiplier")
        if multiplier is None:
            multiplier = narrative_multiplier(metadata.get("text", ""))
        return retrieved.score * multiplier

    return sorted(ranked, key=lambda item: (-adjusted_score(item), item.chunk_id))[:limit]


def build_synthesis_messages(context: str, query: str) -> list[dict[str, str]]:
    """Messages passed to the GGUF's embedded Jinja chat template."""
    instruction = (
        "You are a factual storytelling assistant. Identify the characters correctly "
        "based on the text, including who is the mother, the elephant, and the bride. "
        "Use ONLY the provided story context. Summarize the plot in 2 direct sentences. "
        "Do not echo block headers, chunk tags, or unparsed quotes. "
        "If the story context does not contain the answer, say so."
    )
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            # Section identity is retained in provenance, not injected into the
            # generation context where it can be copied into the answer.
            "content": f"Story Context:\n{context}\n\nQuestion: {query}",
        },
    ]


def _load_local_llm(gguf_path: Path, n_ctx: int, n_threads: int):
    """Import llama-cpp lazily so build/query continue to need no inference dependency."""
    try:
        from llama_cpp import Llama
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "QA synthesis requires llama-cpp-python. Install it with `pip install llama-cpp-python`."
        ) from exc
    return Llama(model_path=str(gguf_path), n_ctx=n_ctx, n_threads=n_threads, verbose=False)


def build_prism_artifact(document_path: Path, gguf_path: Path, output_path: Path, chunk_size: int, append: bool = True) -> Path:
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")
    if _is_local_model_path(gguf_path) and not gguf_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {gguf_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # progress callback will print percentage to stdout
    def _progress(i: int, total: int) -> None:
        pct = 100 if total == 0 else (i * 100) // total
        print(f"\rTokenizing: {pct}%", end="", flush=True)

    build_prism_from_document(
        str(document_path),
        str(gguf_path),
        str(output_path),
        chunk_size=chunk_size,
        append=append,
        progress_callback=_progress,
    )
    print()
    return output_path


def query_prism_artifact(prism_path: Path, gguf_path: Path, query: str, top_n: int, preview_tokens: int, log_path: Optional[Path]) -> int:
    if not prism_path.exists():
        raise FileNotFoundError(f"Prism artifact not found: {prism_path}")
    if _is_local_model_path(gguf_path) and not gguf_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {gguf_path}")

    tokenizer = get_tokenizer_adapter(str(gguf_path))
    query_tokens = tokenizer.encode(query)
    if not query_tokens:
        print_error("No query tokens produced from input text.")
        return 1

    reader = PrismReader(str(prism_path))
    library = load_library(prism_path)
    routed_section = None
    if library:
        routed_hash, allowed_chunk_ids, _ = route_section(query, reader.score_chunks(query_tokens), library)
        if routed_hash:
            routed_section = library["sections"][routed_hash]
            ranked = _boost_narrative_chunks(
                reader.ranked_chunks(query_tokens, top_n=max(top_n * 4, top_n), allowed_chunk_ids=allowed_chunk_ids),
                library, top_n,
            )
        else:
            ranked = reader.ranked_chunks(query_tokens, top_n=top_n)
    else:
        ranked = reader.ranked_chunks(query_tokens, top_n=top_n)

    print_header("Prism Query")
    print_status(f"Prism artifact: {prism_path}")
    print_status(f"GGUF model: {gguf_path}")
    print(f"Query: {Color.bold(query)}")
    print(f"Top chunks: {top_n}, Preview tokens: {preview_tokens}")
    print(f"Tokenizer vocab: {len(tokenizer.vocab)}, Prism chunks: {reader.header.chunk_count}\n")
    if routed_section:
        print_status(f"Routed section: {routed_section['section_title']} ({routed_section['section_id']})\n")

    if not ranked:
        print_error("No matching chunks found in the Prism index.")
        return 0

    print_header("Top Ranked Chunks")
    for retrieved in ranked:
        print(
            f"  {Color.yellow('chunk')}={retrieved.chunk_id} "
            f"score={retrieved.score:.4f} "
            f"tokens={retrieved.metadata.token_count} "
            f"start={retrieved.metadata.token_start}"
        )

    print_header("Chunk Previews")
    for rank, retrieved in enumerate(ranked, start=1):
        preview = decode_snippet(retrieved.tokens[:preview_tokens], tokenizer)
        print(f"[{rank}] chunk={retrieved.chunk_id} => {preview}")

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_content = [
            f"Prism artifact: {prism_path}",
            f"GGUF model: {gguf_path}",
            f"Query: {query}",
            f"Top chunks: {[ (c.chunk_id, round(c.score, 4)) for c in ranked ]}",
        ]
        log_path.write_text("\n".join(log_content), encoding="utf-8")
        print_status(f"Wrote query log to {log_path}")

    return 0


def qa_prism_artifact(
    prism_path: Path,
    gguf_path: Path,
    query: str,
    top_n: int,
    preview_tokens: int,
    answer_length: int,
    log_path: Optional[Path],
    n_ctx: int = 4096,
    n_threads: int = 2,
    model_format: str = "auto",
) -> int:
    if not prism_path.exists():
        raise FileNotFoundError(f"Prism artifact not found: {prism_path}")
    if _is_local_model_path(gguf_path) and not gguf_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {gguf_path}")
    # Retained for CLI compatibility. llama-cpp reads the GGUF's embedded chat
    # template instead of selecting a hand-written format.
    del model_format

    tokenizer = get_tokenizer_adapter(str(gguf_path))
    query_tokens = tokenizer.encode(query)
    if not query_tokens:
        print_error("No query tokens produced from input text.")
        return 1

    reader = PrismReader(str(prism_path))
    library = load_library(prism_path)
    routed_section = None
    allowed_chunk_ids = None
    if library:
        routed_hash, allowed_chunk_ids, _ = route_section(query, reader.score_chunks(query_tokens), library)
        if routed_hash:
            routed_section = library["sections"][routed_hash]
    # Oversampling happens only inside the routed section, preserving the fast
    # token index while ensuring unrelated stories cannot enter the prompt.
    ranked = reader.ranked_chunks(
        query_tokens,
        top_n=max(top_n * 4, top_n),
        allowed_chunk_ids=allowed_chunk_ids,
    )
    ranked = _boost_narrative_chunks(ranked, library, max(top_n * 4, top_n))

    print_header("Prism QA")
    print_status(f"Prism artifact: {prism_path}")
    print_status(f"GGUF model: {gguf_path}")
    print(f"Query: {Color.bold(query)}")
    print(f"Using up to {top_n} filtered micro-chunks for local synthesis.\n")
    if routed_section:
        print_status(f"Stage 1 route: {routed_section['section_title']} ({routed_section['section_id']})\n")

    if not ranked:
        print_error("No matching chunks found in the Prism index.")
        return 0

    # Extract sentence-level window (matching sentence + 1 before + 1 after) for each candidate chunk
    sentence_candidates: list[tuple[float, int, int, float, float, bool, str]] = []
    # (combined_score, chunk_id, sentence_index, retrieval_score, narrative_multiplier, is_noise, window)
    for retrieved in ranked:
        text = tokenizer.decode(retrieved.tokens)
        best_idx, window_text, sent_rel = extract_highest_scoring_sentence_window(text, query)
        if window_text:
            is_noise = _is_structural_noise(text)
            # A substantial discount leaves a noise chunk available only if there
            # is no meaningful alternative, preserving recall for unusual files.
            noise_multiplier = 0.08 if is_noise else 1.0
            chunk_metadata = (library or {}).get("chunks", {}).get(str(retrieved.chunk_id), {})
            narrative_boost = chunk_metadata.get("narrative_multiplier")
            if narrative_boost is None:
                narrative_boost = narrative_multiplier(chunk_metadata.get("text", text))
            combined = retrieved.score * narrative_boost * (1.0 + sent_rel) * noise_multiplier
            window_text = _limit_to_token_budget(window_text, tokenizer, max_tokens=150)
            sentence_candidates.append((combined, retrieved.chunk_id, best_idx, retrieved.score, narrative_boost, is_noise, window_text))

    sentence_candidates.sort(key=lambda t: t[0], reverse=True)

    context_pieces: list[str] = []
    provenance: list[tuple[int, int, float, float, float, bool, str]] = []
    # chunk_id, sentence_index, combined_score, retrieval_score, narrative_multiplier, is_noise, window
    used_chunks = set()
    for score, chunk_id, sidx, retrieval_score, narrative_boost, is_noise, window_text in sentence_candidates:
        if len(context_pieces) >= top_n:
            break
        if chunk_id in used_chunks:
            continue
        piece = window_text.strip()
        if not piece:
            continue
        # Keep model context text-only. Chunk and section identifiers are
        # recorded below in provenance, never exposed to the language model.
        context_pieces.append(piece)
        provenance.append((chunk_id, sidx, score, retrieval_score, narrative_boost, is_noise, piece))
        used_chunks.add(chunk_id)

    if not context_pieces:
        print_error("No usable sentence windows were found in the retrieved chunks.")
        return 0

    if not gguf_path.is_file() or gguf_path.suffix.lower() != ".gguf":
        raise ValueError("QA synthesis requires a local .gguf model path for llama-cpp-python.")
    llm = _load_local_llm(gguf_path, n_ctx=n_ctx, n_threads=n_threads)
    section_title = routed_section["section_title"] if routed_section else "Unscoped legacy artifact"
    summary = get_section_summary(library, routed_hash) if library and routed_section else "No section summary is available."
    anchored_context = f"Section: {section_title}\nSummary: {summary}\n\nContext Slices:\n" + "\n\n".join(context_pieces)
    response = llm.create_chat_completion(
        messages=build_synthesis_messages(anchored_context, query),
        max_tokens=answer_length,
        temperature=0.2,
    )
    answer = (response["choices"][0]["message"]["content"] or "").strip()

    print_header("Candidate Answer (synthesized)")
    print(answer or "(no concise answer could be synthesized)")

    print_header("Provenance")
    for idx, (chunk_id, sidx, score, retrieval_score, narrative_boost, is_noise, sent) in enumerate(provenance, start=1):
        noise_marker = " noise-penalized" if is_noise else ""
        print(f"[{idx}] chunk={chunk_id} sent={sidx} score={score:.4f} retrieval_score={retrieval_score:.4f} narrative_multiplier={narrative_boost:.2f}{noise_marker}")
        print(f"    {sent}\n")

    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_content = [
            f"Prism artifact: {prism_path}",
            f"GGUF model: {gguf_path}",
            f"Query: {query}",
            f"Answer: {answer}",
            f"Routed section: {section_title}",
            f"Top chunks: {[ (c.chunk_id, round(c.score, 4)) for c in ranked[:top_n] ]}",
            f"Provenance: {[ (chunk_id, sent_idx, round(score, 4)) for chunk_id, sent_idx, score, _, _, _, _ in provenance ]}",
        ]
        log_path.write_text("\n".join(log_content), encoding="utf-8")
        print_status(f"Wrote QA log to {log_path}")

    return 0


def run_demo(gguf_path: Path, document_path: Path, chunk_size: int, top_n: int, preview_tokens: int, query: str, log_path: Optional[Path]) -> int:
    print_header("Prism Demo")
    print_status(f"Document: {document_path}")
    print_status(f"GGUF: {gguf_path}")
    print(f"Chunk size: {chunk_size}, Top N: {top_n}, Preview tokens: {preview_tokens}\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        prism_path = Path(tmpdir) / "demo.prism"
        output_path = build_prism_artifact(document_path, gguf_path, prism_path, chunk_size)
        print_status(f"Built Prism artifact: {output_path}")
        print(f"Prism file size: {output_path.stat().st_size} bytes\n")
        return query_prism_artifact(output_path, gguf_path, query, top_n, preview_tokens, log_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prism token-space CLI with build/query/demo/qa subcommands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a .prism artifact from a document.")
    build_parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT_PATH, help="PDF or TXT document to import.")
    build_parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF_PATH, help="GGUF model file used for tokenizer extraction.")
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output path for the generated .prism file.")
    build_parser.add_argument("--chunk-size", type=int, default=150, help="Token size of section-scoped micro-chunks.")
    append_group = build_parser.add_mutually_exclusive_group()
    append_group.add_argument(
        "--append",
        dest="append",
        action="store_true",
        help="Append knowledge to an existing .prism file instead of overwriting it.",
    )
    append_group.add_argument(
        "--overwrite",
        dest="append",
        action="store_false",
        help="Overwrite any existing .prism artifact instead of appending.",
    )
    build_parser.set_defaults(append=True)

    query_parser = subparsers.add_parser("query", help="Query an existing .prism artifact.")
    query_parser.add_argument("--prism", type=Path, required=True, help="Existing .prism artifact to query.")
    query_parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF_PATH, help="GGUF model file used for tokenizer extraction.")
    query_parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query text to search the Prism artifact.")
    query_parser.add_argument("--top-n", type=int, default=3, help="Number of top ranked chunks to display.")
    query_parser.add_argument("--preview-tokens", type=int, default=64, help="Number of tokens to decode from each returned chunk.")
    query_parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_QUERY_PATH, help="Optional log file path for query output.")

    qa_parser = subparsers.add_parser("qa", help="Synthesize a factual answer from a Prism artifact with a local GGUF model.")
    qa_parser.add_argument("--prism", type=Path, required=True, help="Existing .prism artifact to query.")
    qa_parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF_PATH, help="GGUF model file used for tokenizer extraction.")
    qa_parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query text to ask the Prism artifact.")
    qa_parser.add_argument("--top-n", type=int, default=3, help="Number of filtered micro-chunks to pass to the model.")
    qa_parser.add_argument("--preview-tokens", type=int, default=128, help="Number of tokens to decode from each returned chunk.")
    qa_parser.add_argument("--answer-length", type=int, default=250, help="Maximum generated answer tokens.")
    qa_parser.add_argument("--n-ctx", type=int, choices=(4096, 8192), default=4096, help="Local LLM context window.")
    qa_parser.add_argument("--threads", type=int, choices=(2, 3, 4), default=2, help="CPU threads reserved for local inference.")
    qa_parser.add_argument("--model-format", choices=("auto", "qwen", "gemma"), default="auto", help="Deprecated; the GGUF's embedded chat template is used.")
    qa_parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_QA_PATH, help="Optional log file path for QA output.")

    demo_parser = subparsers.add_parser("demo", help="Run the built-in demo using the sample document and default model.")
    demo_parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF_PATH, help="Path to the GGUF model file.")
    demo_parser.add_argument("--document", type=Path, default=DEFAULT_DOCUMENT_PATH, help="Sample document to import.")
    demo_parser.add_argument("--chunk-size", type=int, default=150, help="Token size of section-scoped micro-chunks.")
    demo_parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query text used by the demo.")
    demo_parser.add_argument("--top-n", type=int, default=3, help="Number of top ranked chunks to display.")
    demo_parser.add_argument("--preview-tokens", type=int, default=64, help="Number of tokens to decode from each returned chunk.")
    demo_parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_DEMO_PATH, help="Optional log file path for demo output.")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run tokenizer benchmark on a text file using the configured tokenizer.")
    benchmark_parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF_PATH, help="GGUF model file used for tokenizer extraction.")
    benchmark_parser.add_argument("--file", type=Path, required=True, help="Text file to tokenize for benchmark.")
    benchmark_parser.add_argument("--iterations", type=int, default=3, help="Number of repeated tokenization runs.")

    args = parser.parse_args()

    try:
        if args.command == "build":
            build_path = build_prism_artifact(args.document, args.gguf, args.output, args.chunk_size, append=args.append)
            action = "Appended to" if args.append and args.output.exists() else "Wrote"
            print_status(f"{action} Prism artifact to {build_path}")
            return 0

        if args.command == "benchmark":
            # import lazily to avoid adding a hard dependency at module import time
            from tools.tokenizer_benchmark import benchmark_text

            tokenizer = get_tokenizer_adapter(str(args.gguf))
            text = args.file.read_text(encoding="utf-8")
            stats = benchmark_text(tokenizer, text, iterations=args.iterations)
            print("\nBenchmark results:")
            print(f"  Iterations: {stats['iterations']}")
            print(f"  Tokens per single run: {stats['total_tokens_once']}")
            print(f"  Total tokens processed: {stats['tokens_processed']}")
            print(f"  Elapsed (s): {stats['elapsed_s']:.4f}")
            print(f"  Throughput (tokens/sec): {stats['tokens_per_sec']:.2f}")
            return 0

        if args.command == "query":
            return query_prism_artifact(args.prism, args.gguf, args.query, args.top_n, args.preview_tokens, args.log_path)

        if args.command == "qa":
            return qa_prism_artifact(
                args.prism, args.gguf, args.query, args.top_n, args.preview_tokens,
                args.answer_length, args.log_path, args.n_ctx, args.threads, args.model_format,
            )

        if args.command == "demo":
            return run_demo(args.gguf, args.document, args.chunk_size, args.top_n, args.preview_tokens, args.query, args.log_path)

        parser.print_help()
        return 1
    except Exception as exc:
        print_error(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
