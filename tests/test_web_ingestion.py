from prism_protocol.src.token_space.web_ingestion import WebKnowledgeDeduplicator, _source_rank


def test_web_deduplicator_skips_exact_and_near_duplicate_paragraphs():
    library = {
        "chunks": {
            "7": {"text": "Precision crop spraying reduces pesticide waste by targeting affected crops directly."}
        }
    }
    deduper = WebKnowledgeDeduplicator(similarity_threshold=0.55, library=library)
    unique, duplicates = deduper.process([
        {
            "source_url": "https://site-a.example/article",
            "content": "Precision crop spraying reduces pesticide waste by targeting affected crops directly.\n"
                       "Livestock tracking uses thermal vision to locate stray cattle in remote pastures.",
        },
        {
            "source_url": "https://site-b.example/article",
            "content": "Precision crop spraying reduces pesticide waste by targeting affected crops directly.\n"
                       "Yield prediction analyzes canopy cover to estimate harvest output accurately.",
        },
    ])

    assert len(unique) == 2
    assert len(duplicates) == 2
    assert all(item["kind"] == "exact" for item in duplicates)
    assert all("source_url" in item for item in unique)


def test_official_domains_are_ranked_before_generic_search_results():
    assert _source_rank("https://www.hiv.gov/topic") < _source_rank("https://example.com/topic")
