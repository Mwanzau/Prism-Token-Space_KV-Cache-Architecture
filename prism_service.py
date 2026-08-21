import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from prism_protocol.src.token_space.binary_format import PrismReader
from prism_protocol.src.token_space.importer import build_prism_from_document
from prism_protocol.src.token_space.library import (
    PrismMetadataRegistry,
    load_library,
    route_section,
)
from prism_protocol.src.token_space.tokenizer_adapter import get_tokenizer_adapter
from run_rag_demo import (
    _boost_narrative_chunks,
    _is_local_model_path,
    _is_structural_noise,
    _limit_to_token_budget,
    _load_local_llm,
    build_synthesis_messages,
    extract_highest_scoring_sentence_window,
)

app = FastAPI(
    title="Prism Token-Space Engine API",
    description="High-performance memory-mapped binary RAG engine with sidecar metadata registry and SLM synthesis.",
    version="2.0.0",
)

DEFAULT_PRISM_BINARY_FILE = "output/master_brain.prism"
DEFAULT_GGUF_MODEL_PATH = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"


class IngestAppendRequest(BaseModel):
    content: str = Field(..., description="Plain text, code block, or table markdown to ingest")
    format: str = Field("text", description="Content modality format: 'text', 'code', or 'table'")
    language: Optional[str] = Field(None, description="Programming language if format is 'code'")
    source_uri: str = Field("user_ingest", description="Source document name, URL, or identifier")
    section_id: str = Field("GENERAL_SECTION", description="Taxonomic section tag")
    gguf_model_path: str = Field(DEFAULT_GGUF_MODEL_PATH, description="Model path or hf: identifier for tokenization")
    prism_binary_file: str = Field(DEFAULT_PRISM_BINARY_FILE, description="Target binary .prism file path")
    chunk_size: int = Field(128, description="Token-space chunk size")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary JSON metadata payload")


class QueryRequest(BaseModel):
    prism_path: str = Field(DEFAULT_PRISM_BINARY_FILE, description="Path to binary .prism artifact")
    gguf_path: str = Field(DEFAULT_GGUF_MODEL_PATH, description="Path to model or hf: tokenizer identifier")
    query: str = Field(..., description="Natural language search query")
    top_n: int = Field(3, description="Number of top chunks to retrieve")
    preview_tokens: int = Field(64, description="Preview token length")


class QARequest(BaseModel):
    prism_path: str = Field(DEFAULT_PRISM_BINARY_FILE, description="Path to binary .prism artifact")
    gguf_path: str = Field(DEFAULT_GGUF_MODEL_PATH, description="Path to model or hf: tokenizer identifier")
    query: str = Field(..., description="Natural language question to answer")
    top_n: int = Field(3, description="Number of context micro-chunks for synthesis")
    preview_tokens: int = Field(64, description="Preview token length for snippets")
    answer_length: int = Field(150, description="Maximum token length for candidate answer")
    n_ctx: int = Field(4096, description="Llama context window size")
    n_threads: int = Field(4, description="CPU threads for generation")


@app.get("/health")
async def health_check():
    """Engine health status, default paths, and sidecar availability."""
    prism_exists = os.path.exists(DEFAULT_PRISM_BINARY_FILE)
    meta_path = f"{DEFAULT_PRISM_BINARY_FILE}.meta.json"
    meta_exists = os.path.exists(meta_path)
    return {
        "status": "healthy",
        "engine": "Prism Token-Space KV-Cache Architecture",
        "version": "2.0.0",
        "default_prism_binary": DEFAULT_PRISM_BINARY_FILE,
        "default_prism_exists": prism_exists,
        "sidecar_metadata_exists": meta_exists,
    }


@app.post("/ingest/append")
async def append_to_prism(payload: IngestAppendRequest):
    """
    Appends new multi-modal data by compiling into the binary .prism file 
    and updating the metadata sidecar registry with SHA-256 deduplication.
    """
    content = payload.content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

    sidecar_path = f"{payload.prism_binary_file}.meta.json"
    registry = PrismMetadataRegistry(sidecar_path=sidecar_path)

    # 1. SHA-256 Deduplication check
    if registry.is_duplicate(content):
        return {
            "status": "skipped",
            "reason": "Duplicate content detected by SHA-256 hash check.",
            "sha256": registry.compute_sha256(content),
        }

    # 2. Multi-modal formatting (code blocks, structured tables)
    formatted_payload = registry.format_multimodal_content(content, payload.format, payload.language)

    # 3. Write temporarily to temporary file buffer and append to binary .prism
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".txt") as tmp:
        tmp.write(formatted_payload)
        tmp_path = tmp.name

    try:
        build_prism_from_document(
            document_path=tmp_path,
            gguf_path=payload.gguf_model_path,
            output_path=payload.prism_binary_file,
            chunk_size=payload.chunk_size,
            append=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Binary Prism compilation failed: {str(exc)}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 4. Register chunk metadata in sidecar registry
    content_hash = registry.register_chunk(
        section_id=payload.section_id,
        content=content,
        content_format=payload.format,
        source_uri=payload.source_uri,
        extra_meta=payload.metadata,
    )

    return {
        "status": "success",
        "sha256": content_hash,
        "prism_binary_file": payload.prism_binary_file,
        "message": "Appended cleanly to binary .prism engine.",
    }


@app.post("/query")
async def query_prism(payload: QueryRequest):
    """
    Performs Stage 1 section routing and Stage 2 BM25 token-space retrieval over .prism binary.
    """
    prism_p = Path(payload.prism_path)
    gguf_p = Path(payload.gguf_path)
    if not prism_p.exists():
        raise HTTPException(status_code=404, detail=f"Prism artifact not found: {payload.prism_path}")
    if _is_local_model_path(gguf_p) and not gguf_p.exists():
        raise HTTPException(status_code=404, detail=f"Model path not found: {payload.gguf_path}")

    try:
        tokenizer = get_tokenizer_adapter(str(gguf_p))
        query_tokens = tokenizer.encode(payload.query)
        if not query_tokens:
            raise HTTPException(status_code=400, detail="Query yielded no tokens.")

        reader = PrismReader(str(prism_p))
        library = load_library(prism_p)

        routed_section = None
        if library:
            routed_hash, allowed_chunk_ids, _ = route_section(payload.query, reader.score_chunks(query_tokens), library)
            if routed_hash:
                routed_section = library["sections"][routed_hash]
                ranked = _boost_narrative_chunks(
                    reader.ranked_chunks(query_tokens, top_n=max(payload.top_n * 4, payload.top_n), allowed_chunk_ids=allowed_chunk_ids),
                    library, payload.top_n,
                )
            else:
                ranked = reader.ranked_chunks(query_tokens, top_n=payload.top_n)
        else:
            ranked = reader.ranked_chunks(query_tokens, top_n=payload.top_n)

        results = []
        for rank, retrieved in enumerate(ranked, start=1):
            full_text = tokenizer.decode(retrieved.tokens)
            _, window_text, score = extract_highest_scoring_sentence_window(full_text, payload.query)
            preview = window_text if (window_text and score > 0) else full_text[:payload.preview_tokens * 4]
            results.append({
                "rank": rank,
                "chunk_id": retrieved.chunk_id,
                "score": round(retrieved.score, 4),
                "token_count": retrieved.metadata.token_count,
                "preview_text": preview,
            })

        return {
            "status": "success",
            "query": payload.query,
            "prism_path": payload.prism_path,
            "routed_section": routed_section,
            "retrieved_chunks": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(exc)}") from exc


@app.post("/qa")
async def qa_prism(payload: QARequest):
    """
    Full end-to-end RAG pipeline: Stage 1 routing, Stage 2 retrieval, sentence window filtering, and SLM answer synthesis.
    """
    prism_p = Path(payload.prism_path)
    gguf_p = Path(payload.gguf_path)
    if not prism_p.exists():
        raise HTTPException(status_code=404, detail=f"Prism artifact not found: {payload.prism_path}")
    if _is_local_model_path(gguf_p) and not gguf_p.exists():
        raise HTTPException(status_code=404, detail=f"Model path not found: {payload.gguf_path}")

    try:
        tokenizer = get_tokenizer_adapter(str(gguf_p))
        query_tokens = tokenizer.encode(payload.query)
        if not query_tokens:
            raise HTTPException(status_code=400, detail="Query yielded no tokens.")

        reader = PrismReader(str(prism_p))
        library = load_library(prism_p)

        routed_hash = None
        allowed_chunk_ids = None
        if library:
            routed_hash, allowed_chunk_ids, _ = route_section(payload.query, reader.score_chunks(query_tokens), library)

        if allowed_chunk_ids:
            ranked = _boost_narrative_chunks(
                reader.ranked_chunks(query_tokens, top_n=max(payload.top_n * 4, payload.top_n), allowed_chunk_ids=allowed_chunk_ids),
                library, payload.top_n,
            )
        else:
            ranked = reader.ranked_chunks(query_tokens, top_n=payload.top_n)

        if not ranked:
            return {
                "status": "no_match",
                "answer": "No relevant text chunks found to answer the question.",
                "provenance": [],
            }

        provenance = []
        filtered_snippets = []
        for retrieved in ranked:
            chunk_text = tokenizer.decode(retrieved.tokens)
            if _is_structural_noise(chunk_text):
                continue
            sent_idx, window_text, sent_score = extract_highest_scoring_sentence_window(chunk_text, payload.query)
            clean_text = window_text if (window_text and sent_score > 0) else chunk_text
            clean_text = _limit_to_token_budget(clean_text, tokenizer, max_tokens=150)
            if clean_text:
                filtered_snippets.append(clean_text)
                provenance.append({
                    "chunk_id": retrieved.chunk_id,
                    "score": round(retrieved.score, 4),
                    "text_snippet": clean_text[:180],
                })

        context = "\n\n".join(filtered_snippets) if filtered_snippets else tokenizer.decode(ranked[0].tokens)

        # Answer Synthesis via llama-cpp if model is local GGUF
        answer = ""
        if _is_local_model_path(gguf_p) and gguf_p.exists():
            llm = _load_local_llm(gguf_p, payload.n_ctx, payload.n_threads)
            messages = build_synthesis_messages(context, payload.query)
            output = llm.create_chat_completion(
                messages=messages,
                max_tokens=payload.answer_length,
                temperature=0.2,
            )
            answer = output["choices"][0]["message"]["content"].strip()
        else:
            answer = context[:300] + "..."

        return {
            "status": "success",
            "query": payload.query,
            "answer": answer,
            "provenance": provenance,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"QA failed: {str(exc)}") from exc


@app.get("/export/pack")
async def export_prism_artifact(prism_path: str = Query(DEFAULT_PRISM_BINARY_FILE, description="Path to .prism binary file")):
    """
    Downloads/exports the binary .prism file artifact for deployment or distribution.
    """
    if not os.path.exists(prism_path):
        raise HTTPException(status_code=404, detail=f"Binary .prism file not found: {prism_path}")

    filename = os.path.basename(prism_path)
    return FileResponse(
        path=prism_path,
        filename=filename,
        media_type="application/octet-stream",
    )
