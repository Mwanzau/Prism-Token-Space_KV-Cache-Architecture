import math
import pytest
from prism_protocol.src.token_space.indexer import PrismIndexer
from run_rag_demo import extract_highest_scoring_sentence_window, _split_into_sentences


def test_bm25_idf_calculation():
    # IDF(q) = log( (N - n(q) + 0.5) / (n(q) + 0.5) + 1.0 )
    idf_rare = PrismIndexer.calculate_bm25_idf(total_chunks=100, doc_freq=1)
    idf_common = PrismIndexer.calculate_bm25_idf(total_chunks=100, doc_freq=99)
    assert idf_rare > idf_common
    assert idf_rare > 0
    assert idf_common > 0

    # Rare term IDF: (100 - 1 + 0.5)/(1 + 0.5) + 1.0 = 99.5/1.5 + 1 = 67.333 -> ln(67.333) ~ 4.209
    expected_rare = math.log((100 - 1 + 0.5) / (1 + 0.5) + 1.0)
    assert math.isclose(idf_rare, expected_rare, rel_tol=1e-5)


def test_proximity_co_occurrence_boost():
    # Create 2 chunks:
    # Chunk 0 has token 100 at index 5 and token 200 at index 12 (distance = 7 <= 15 -> Boosted 2.5x)
    # Chunk 1 has token 100 at index 5 and token 200 at index 80 (distance = 75 > 50 -> No boost 1.0x)
    chunk0 = [10] * 128
    chunk0[5] = 100
    chunk0[12] = 200

    chunk1 = [10] * 128
    chunk1[5] = 100
    chunk1[80] = 200

    corpus = {0: chunk0, 1: chunk1}

    def get_doc_freq(token_id: int) -> int:
        if token_id in (100, 200):
            return 2
        return 2

    def get_candidate_chunks(query_tks):
        return [0, 1]

    def get_chunk_tokens(chunk_id):
        return corpus[chunk_id]

    query_tokens = [100, 200]
    scores = PrismIndexer.rank_chunks(
        query_tokens=query_tokens,
        total_chunks=2,
        get_doc_freq=get_doc_freq,
        get_candidate_chunks=get_candidate_chunks,
        get_chunk_tokens=get_chunk_tokens,
        k1=1.5,
        b=0.75,
        proximity_window=15,
        proximity_boost=2.5,
    )

    assert 0 in scores and 1 in scores
    # Chunk 0 should have 2.5x the score of Chunk 1 since chunk lengths and term frequencies are identical
    ratio = scores[0] / scores[1]
    assert math.isclose(ratio, 2.5, rel_tol=1e-3)


def test_sentence_window_extraction():
    text = (
        "Parle was a very pretty girl who lived with her family by the river. "
        "It was not a pleasant river at all, for it was brown in color, and flowed through a dark forest. "
        "There were crocodiles in the river, and Parle was afraid."
    )
    query = "What is the color of the river?"

    best_idx, window_text, rel_score = extract_highest_scoring_sentence_window(text, query)
    assert best_idx == 1
    assert "brown in color" in window_text
    # Should include sentence before (idx 0), matching sentence (idx 1), and sentence after (idx 2)
    sentences = _split_into_sentences(text)
    assert len(sentences) == 3
    assert window_text == " ".join(sentences)
