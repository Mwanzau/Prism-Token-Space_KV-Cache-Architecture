import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prism_protocol.src.token_space.library import PrismMetadataRegistry
from prism_service import app

ROOT = Path(__file__).resolve().parent.parent
GGUF_MODEL = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"


# ──────────────────────────────────────────────
# 1. Test PrismMetadataRegistry
# ──────────────────────────────────────────────

def test_metadata_registry_sha256_deduplication():
    with tempfile.TemporaryDirectory() as tmpdir:
        meta_file = Path(tmpdir) / "test.prism.meta.json"
        registry = PrismMetadataRegistry(sidecar_path=meta_file)

        content = "def calculate_bmi(height, weight): return weight / (height ** 2)"
        assert not registry.is_duplicate(content)

        hash1 = registry.register_chunk("CODE_SEC", content, content_format="code", source_uri="bmi.py")
        assert registry.is_duplicate(content)
        assert hash1 == registry.compute_sha256(content)


def test_metadata_registry_multimodal_formatting():
    code_formatted = PrismMetadataRegistry.format_multimodal_content("print('hello')", "code", language="python")
    assert "[CODE_BLOCK: python]" in code_formatted

    table_formatted = PrismMetadataRegistry.format_multimodal_content("| A | B |\n|---|---|\n| 1 | 2 |", "table")
    assert "[STRUCTURED_TABLE]" in table_formatted


# ──────────────────────────────────────────────
# 2. FastAPI Endpoints Integration Tests
# ──────────────────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"


@pytest.mark.skipif(not GGUF_MODEL.exists(), reason="Qwen GGUF model not present")
def test_ingest_append_and_query_service(client):
    with tempfile.TemporaryDirectory() as tmpdir:
        prism_file = str(Path(tmpdir) / "service_test.prism")

        # 1. Ingest text
        payload = {
            "content": "The river in the African story was brown in color and filled with crocodiles.",
            "format": "text",
            "source_uri": "african_tales.txt",
            "section_id": "SECTION_RIVER",
            "gguf_model_path": str(GGUF_MODEL),
            "prism_binary_file": prism_file,
            "chunk_size": 64,
        }
        res_ingest = client.post("/ingest/append", json=payload)
        assert res_ingest.status_code == 200
        assert res_ingest.json()["status"] == "success"

        # 2. Ingest duplicate -> should be skipped
        res_dup = client.post("/ingest/append", json=payload)
        assert res_dup.status_code == 200
        assert res_dup.json()["status"] == "skipped"

        # 3. Query the ingested prism
        res_query = client.post(
            "/query",
            json={
                "prism_path": prism_file,
                "gguf_path": str(GGUF_MODEL),
                "query": "What color was the river?",
                "top_n": 1,
            },
        )
        assert res_query.status_code == 200
        query_data = res_query.json()
        assert query_data["status"] == "success"
        assert len(query_data["retrieved_chunks"]) == 1
        assert "brown in color" in query_data["retrieved_chunks"][0]["preview_text"].lower()

        # 4. Export pack endpoint
        res_export = client.get(f"/export/pack?prism_path={prism_file}")
        assert res_export.status_code == 200
        assert len(res_export.content) > 0
