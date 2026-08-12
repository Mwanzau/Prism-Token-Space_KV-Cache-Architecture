import os
import tempfile
from pathlib import Path

from prism_protocol.src.token_space.binary_format import PrismReader
from prism_protocol.src.token_space.importer import build_prism_from_document
from prism_protocol.src.token_space.tokenizer_bridge import TokenizerBridge


def test_append_knowledge_preserves_prior_document():
    root = Path(__file__).resolve().parent.parent
    gguf_path = root / "models" / "Qwen_Qwen2.5-0.5B-Instruct-GGUF" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    doc_a = root / "File_Samples_For_Tests" / "The black barque.txt"
    doc_b = root / "File_Samples_For_Tests" / "The man elephant.txt"

    tokenizer = TokenizerBridge.from_gguf(str(gguf_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        prism_path = Path(tmpdir) / "test.prism"

        # Ingest document A first
        build_prism_from_document(str(doc_a), str(gguf_path), str(prism_path), chunk_size=128, append=True)
        reader = PrismReader(str(prism_path))

        query_a = "What animal is mentioned in the story?"
        tokens_a = tokenizer.encode(query_a)
        ranked_a = reader.ranked_chunks(tokens_a, top_n=2)
        assert ranked_a, "Expected query results for document A"

        # Ingest document B into the same file
        build_prism_from_document(str(doc_b), str(gguf_path), str(prism_path), chunk_size=128, append=True)
        reader = PrismReader(str(prism_path))

        query_b = "What happened in the man elephant story?"
        tokens_b = tokenizer.encode(query_b)
        ranked_b = reader.ranked_chunks(tokens_b, top_n=2)
        assert ranked_b, "Expected query results for document B"

        # Confirm both documents remain queryable
        ranked_a_after = reader.ranked_chunks(tokens_a, top_n=2)
        assert ranked_a_after, "Expected document A still queryable after append"
        assert reader.header.chunk_count >= len(ranked_a) + len(ranked_b)

        # Ensure no corruption: chunk_ids references exist and preview tokens decode cleanly
        for retrieved in ranked_a_after + ranked_b:
            tokens = reader.read_chunk_tokens_by_id(retrieved.chunk_id)
            assert tokens, f"Chunk {retrieved.chunk_id} should contain tokens"
            assert all(isinstance(t, int) for t in tokens)


if __name__ == "__main__":
    test_append_knowledge_preserves_prior_document()
    print("append knowledge test passed")
