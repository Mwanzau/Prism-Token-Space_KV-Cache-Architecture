import os
import tempfile
from pathlib import Path

from prism_protocol.src.token_space.binary_format import PrismWriter, PrismReader
from prism_protocol.src.token_space.importer import build_prism_from_document
from prism_protocol.src.token_space.tokenizer_bridge import TokenizerBridge
from prism_protocol.src.token_space.indexer import PrismIndexer


def test_prism_token_space_roundtrip():
    root = Path(__file__).resolve().parent.parent
    gguf_path = root / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    sample_text_path = root / "File_Samples_For_Tests" / "A Tale of the Boer Invasion.txt"

    tokenizer = TokenizerBridge.from_gguf(str(gguf_path))
    sample_text = sample_text_path.read_text(errors="ignore")
    chunk_text = " ".join(sample_text.split()[:128])

    encoded = tokenizer.encode(chunk_text)
    assert encoded, "Encoded text should produce token ids"

    chunks = [encoded]
    index_entries = PrismWriter.build_index(chunks)

    with tempfile.TemporaryDirectory() as tmpdir:
        prism_path = Path(tmpdir) / "sample.prism"
        PrismWriter.write_prism(str(prism_path), chunks, tokenizer.model_vocab_hash, index_entries)

        reader = PrismReader(str(prism_path))
        assert reader.header.model_vocab_hash == tokenizer.model_vocab_hash
        assert reader.header.chunk_count == len(chunks)

        result_tokens = reader.stream_tokens_for_query(encoded[:8])
        assert result_tokens
        assert all(isinstance(t, int) for t in result_tokens)
        assert result_tokens[: len(encoded[:8])] == encoded[:8]


def test_prism_importer_builds_from_sample_text():
    root = Path(__file__).resolve().parent.parent
    gguf_path = root / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    sample_path = root / "File_Samples_For_Tests" / "A Tale of the Boer Invasion.txt"

    with tempfile.TemporaryDirectory() as tmpdir:
        prism_path = Path(tmpdir) / "sample_import.prism"
        build_prism_from_document(str(sample_path), str(gguf_path), str(prism_path), chunk_size=64)

        reader = PrismReader(str(prism_path))
        assert reader.header.chunk_count > 0
        assert reader.header.model_vocab_hash is not None

        query_tokens = [1, 2, 3]
        result_tokens = reader.stream_tokens_for_query(query_tokens)
        assert isinstance(result_tokens, list)


def test_prism_importer_builds_from_sample_pdf():
    root = Path(__file__).resolve().parent.parent
    gguf_path = root / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    pdf_path = root / "File_Samples_For_Tests" / "01Chapter1.pdf"

    with tempfile.TemporaryDirectory() as tmpdir:
        prism_path = Path(tmpdir) / "sample_import_pdf.prism"
        build_prism_from_document(str(pdf_path), str(gguf_path), str(prism_path), chunk_size=64)

        reader = PrismReader(str(prism_path))
        assert reader.header.chunk_count > 0
        assert reader.header.model_vocab_hash is not None

        query_tokens = [1, 2, 3]
        result_tokens = reader.stream_tokens_for_query(query_tokens)
        assert isinstance(result_tokens, list)


if __name__ == "__main__":
    test_prism_token_space_roundtrip()
    print("Prism Token-Space roundtrip test passed.")
