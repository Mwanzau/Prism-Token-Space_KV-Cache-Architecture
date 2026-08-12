import io
import math
import os
import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prism_protocol.src.token_space.indexer import IndexMerger

BLOCK_SIZE = 4096
MAGIC = b"PRISM_AI"
HEADER_STRUCT = struct.Struct("<8sQIQI")
INDEX_HEADER_STRUCT = struct.Struct("<II")
INDEX_ENTRY_STRUCT = struct.Struct("<IQI")
CHUNK_METADATA_STRUCT = struct.Struct("<IQI")


@dataclass
class PrismBinaryHeader:
    model_vocab_hash: int
    chunk_count: int
    index_offset: int
    flags: int = 0

    def pack(self) -> bytes:
        raw = HEADER_STRUCT.pack(MAGIC, self.model_vocab_hash, self.chunk_count, self.index_offset, self.flags)
        return raw + b"\x00" * (BLOCK_SIZE - len(raw))

    @classmethod
    def unpack(cls, raw: bytes) -> "PrismBinaryHeader":
        magic, model_vocab_hash, chunk_count, index_offset, flags = HEADER_STRUCT.unpack(raw[:HEADER_STRUCT.size])
        if magic != MAGIC:
            raise ValueError("Invalid PRISM_AI file magic")
        return cls(model_vocab_hash=model_vocab_hash, chunk_count=chunk_count, index_offset=index_offset, flags=flags)

    def validate_vocab(self, expected_hash: int) -> None:
        if self.model_vocab_hash != expected_hash:
            raise ValueError(
                f"Vocabulary hash mismatch: file={self.model_vocab_hash} expected={expected_hash}"
            )


@dataclass
class PrismChunkMetadata:
    chunk_id: int
    token_start: int
    token_count: int


@dataclass
class PrismRetrievedChunk:
    chunk_id: int
    score: float
    tokens: List[int]
    metadata: PrismChunkMetadata


class PrismWriter:
    @staticmethod
    def _pad_to_block(data: bytes, block_size: int = BLOCK_SIZE) -> bytes:
        if len(data) % block_size == 0:
            return data
        return data + b"\x00" * (block_size - (len(data) % block_size))

    @staticmethod
    def _write_header(file_obj: io.BufferedWriter, header: PrismBinaryHeader) -> None:
        file_obj.seek(0)
        file_obj.write(header.pack())

    @staticmethod
    def compress_token_stream(tokens: List[int]) -> bytes:
        palette = bytes(range(256))
        raw = struct.pack(f"<{len(tokens)}I", *tokens)
        compressed = zlib.compress(raw, level=1)
        return struct.pack("<I", len(palette)) + palette + compressed

    @staticmethod
    def align_file(file_obj: io.BufferedWriter, alignment: int = BLOCK_SIZE) -> None:
        offset = file_obj.tell()
        padding = (-offset) % alignment
        if padding:
            file_obj.write(b"\x00" * padding)

    @staticmethod
    def write_prism(
        path: str,
        chunks: List[List[int]],
        model_vocab_hash: int,
        index_entries: Dict[int, List[Tuple[int, int]]],
        kv_blocks: Optional[Dict[int, bytes]] = None,
        flags: int = 0,
    ) -> None:
        kv_blocks = kv_blocks or {}
        chunk_relative_offsets: Dict[int, int] = {}

        with open(path, "wb") as f:
            header = PrismBinaryHeader(model_vocab_hash=model_vocab_hash, chunk_count=0, index_offset=0, flags=flags)
            f.write(header.pack())
            f.seek(BLOCK_SIZE)

            for chunk_id, chunk_tokens in enumerate(chunks):
                chunk_relative_offsets[chunk_id] = f.tell()
                compressed_payload = PrismWriter.compress_token_stream(chunk_tokens)
                f.write(struct.pack("<I", len(compressed_payload)))
                f.write(compressed_payload)
                PrismWriter.align_file(f, 4)

            kv_data = io.BytesIO()
            for chunk_id, kv_payload in kv_blocks.items():
                kv_data.write(struct.pack("<IQI", chunk_id, len(kv_payload), 0))
                kv_data.write(kv_payload)
                PrismWriter.align_file(kv_data, 4)
            if kv_data.tell():
                f.write(PrismWriter._pad_to_block(kv_data.getvalue()))

            PrismWriter.align_file(f)
            index_offset = f.tell()

            token_items = sorted(index_entries.items())
            index_data = io.BytesIO()
            entry_count = sum(len(entries) for entries in index_entries.values())
            index_data.write(INDEX_HEADER_STRUCT.pack(entry_count, len(chunks)))
            for token, entries in token_items:
                entries = sorted(entries, key=lambda e: e[0])
                for chunk_id, _ in entries:
                    offset = chunk_relative_offsets[chunk_id]
                    index_data.write(INDEX_ENTRY_STRUCT.pack(token, offset, chunk_id))

            token_start = 0
            for chunk_id, chunk_tokens in enumerate(chunks):
                index_data.write(CHUNK_METADATA_STRUCT.pack(chunk_id, token_start, len(chunk_tokens)))
                token_start += len(chunk_tokens)

            f.write(PrismWriter._pad_to_block(index_data.getvalue()))
            header.chunk_count = len(chunks)
            header.index_offset = index_offset
            PrismWriter._write_header(f, header)

    @staticmethod
    def append_prism(
        path: str,
        chunks: List[List[int]],
        model_vocab_hash: int,
        index_entries: Dict[int, List[Tuple[int, int]]],
        flags: int = 0,
    ) -> None:
        if not os.path.exists(path) or os.path.getsize(path) <= BLOCK_SIZE:
            PrismWriter.write_prism(path, chunks, model_vocab_hash, index_entries, flags=flags)
            return

        with open(path, "rb+") as f:
            raw = f.read(BLOCK_SIZE)
            existing_header = PrismBinaryHeader.unpack(raw)
            existing_header.validate_vocab(model_vocab_hash)
            reader = PrismReader(path)

            f.seek(0, os.SEEK_END)
            chunk_relative_offsets: Dict[int, int] = {}
            next_chunk_id = existing_header.chunk_count

            for chunk_tokens in chunks:
                chunk_id = next_chunk_id
                chunk_relative_offsets[chunk_id] = f.tell()
                compressed_payload = PrismWriter.compress_token_stream(chunk_tokens)
                f.write(struct.pack("<I", len(compressed_payload)))
                f.write(compressed_payload)
                PrismWriter.align_file(f, 4)
                next_chunk_id += 1

            new_chunk_count = next_chunk_id
            PrismWriter.align_file(f)
            new_index_offset = f.tell()

            new_index: Dict[int, List[Tuple[int, int]]] = {}
            for token, entries in index_entries.items():
                new_entries: List[Tuple[int, int]] = []
                for relative_chunk_id, _ in entries:
                    global_chunk_id = relative_chunk_id + existing_header.chunk_count
                    offset = chunk_relative_offsets[global_chunk_id]
                    new_entries.append((global_chunk_id, offset))
                new_index[token] = new_entries
            merged_index = IndexMerger.merge(reader.index, new_index)
            token_items = sorted(merged_index.items())
            index_data = io.BytesIO()
            entry_count = sum(len(entries) for entries in merged_index.values())
            index_data.write(INDEX_HEADER_STRUCT.pack(entry_count, new_chunk_count))
            for token, entries in token_items:
                entries = sorted(entries, key=lambda e: (e[0], e[1]))
                for chunk_id, offset in entries:
                    index_data.write(INDEX_ENTRY_STRUCT.pack(token, offset, chunk_id))

            token_start = 0
            for chunk_id in range(new_chunk_count):
                if chunk_id < existing_header.chunk_count:
                    token_count = reader.chunk_metadata[chunk_id].token_count
                else:
                    token_count = len(chunks[chunk_id - existing_header.chunk_count])
                index_data.write(CHUNK_METADATA_STRUCT.pack(chunk_id, token_start, token_count))
                token_start += token_count

            f.write(PrismWriter._pad_to_block(index_data.getvalue()))
            existing_header.chunk_count = new_chunk_count
            existing_header.index_offset = new_index_offset
            existing_header.flags = flags
            PrismWriter._write_header(f, existing_header)

    @staticmethod
    def _read_chunk_metadata(path: str, header: PrismBinaryHeader, chunk_id: int) -> Optional[PrismChunkMetadata]:
        with open(path, "rb") as f:
            f.seek(header.index_offset)
            header_data = f.read(INDEX_HEADER_STRUCT.size)
            entry_count, chunk_count = INDEX_HEADER_STRUCT.unpack(header_data)
            f.seek(entry_count * INDEX_ENTRY_STRUCT.size, os.SEEK_CUR)
            for _ in range(chunk_count):
                raw = f.read(CHUNK_METADATA_STRUCT.size)
                if len(raw) != CHUNK_METADATA_STRUCT.size:
                    return None
                existing_chunk_id, token_start, token_count = CHUNK_METADATA_STRUCT.unpack(raw)
                if existing_chunk_id == chunk_id:
                    return PrismChunkMetadata(existing_chunk_id, token_start, token_count)
            return None

    @staticmethod
    def _load_existing_index(path: str, header: PrismBinaryHeader) -> Dict[int, List[Tuple[int, int]]]:
        from prism_protocol.src.token_space.binary_format import PrismReader

        reader = PrismReader(path)
        return reader.index

    @staticmethod
    def build_index(chunks: List[List[int]]) -> Dict[int, List[Tuple[int, int]]]:
        token_index: Dict[int, List[Tuple[int, int]]] = {}
        for chunk_id, tokens in enumerate(chunks):
            for token in set(tokens):
                token_index.setdefault(token, []).append((chunk_id, 0))
        return token_index


class PrismReader:
    def __init__(self, path: str):
        self.path = path
        self.header: Optional[PrismBinaryHeader] = None
        self.index: Dict[int, List[Tuple[int, int]]] = {}
        self.chunk_offsets: Dict[int, int] = {}
        self._read_header()
        self._read_index()

    def _read_header(self) -> None:
        with open(self.path, "rb") as f:
            raw = f.read(BLOCK_SIZE)
        self.header = PrismBinaryHeader.unpack(raw)

    def _read_index(self) -> None:
        self.chunk_metadata: Dict[int, PrismChunkMetadata] = {}
        with open(self.path, "rb") as f:
            f.seek(self.header.index_offset)
            header_data = f.read(INDEX_HEADER_STRUCT.size)
            if len(header_data) != INDEX_HEADER_STRUCT.size:
                raise ValueError("Index header incomplete")
            entry_count, chunk_count = INDEX_HEADER_STRUCT.unpack(header_data)

            for _ in range(entry_count):
                raw = f.read(INDEX_ENTRY_STRUCT.size)
                if len(raw) != INDEX_ENTRY_STRUCT.size:
                    raise ValueError("Index entry incomplete")
                token, offset, chunk_id = INDEX_ENTRY_STRUCT.unpack(raw)
                self.index.setdefault(token, []).append((chunk_id, offset))
                self.chunk_offsets.setdefault(chunk_id, offset)

            for _ in range(chunk_count):
                raw = f.read(CHUNK_METADATA_STRUCT.size)
                if len(raw) != CHUNK_METADATA_STRUCT.size:
                    raise ValueError("Chunk metadata incomplete")
                chunk_id, token_start, token_count = CHUNK_METADATA_STRUCT.unpack(raw)
                self.chunk_metadata[chunk_id] = PrismChunkMetadata(chunk_id, token_start, token_count)

    def lookup_chunks_for_token(self, token_id: int) -> List[Tuple[int, int]]:
        return self.index.get(token_id, [])

    def lookup_chunk_for_token(self, token_id: int) -> Optional[Tuple[int, int]]:
        entries = self.lookup_chunks_for_token(token_id)
        return entries[0] if entries else None

    def read_chunk_tokens_by_id(self, chunk_id: int) -> List[int]:
        metadata = self.chunk_metadata.get(chunk_id)
        if metadata is None:
            return []
        offset = self.chunk_offsets.get(chunk_id)
        if offset is None:
            return []
        with open(self.path, "rb") as f:
            f.seek(offset)
            size_data = f.read(4)
            size = struct.unpack("<I", size_data)[0]
            compressed = f.read(size)
        return self._decompress_payload(compressed)

    def chunk_info(self, chunk_id: int) -> Optional[PrismChunkMetadata]:
        return self.chunk_metadata.get(chunk_id)

    def score_chunks(self, query_tokens: List[int]) -> Dict[int, float]:
        if not query_tokens:
            return {}

        total_chunks = max(self.header.chunk_count, 1)
        query_token_counts = Counter(query_tokens)
        raw_scores: Dict[int, float] = {}

        for token_id, count in query_token_counts.items():
            entries = self.lookup_chunks_for_token(token_id)
            if not entries:
                continue
            doc_freq = len(entries)
            idf = math.log((total_chunks + 1) / (doc_freq + 1)) + 1.0
            for chunk_id, _ in entries:
                raw_scores[chunk_id] = raw_scores.get(chunk_id, 0.0) + count * idf

        normalized_scores: Dict[int, float] = {}
        for chunk_id, score in raw_scores.items():
            chunk_len = self.chunk_metadata.get(chunk_id).token_count if self.chunk_metadata.get(chunk_id) else 1
            normalized_scores[chunk_id] = score / math.sqrt(chunk_len)

        return normalized_scores

    def ranked_chunks(self, query_tokens: List[int], top_n: int = 5) -> List[PrismRetrievedChunk]:
        scores = self.score_chunks(query_tokens)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_n]
        result: List[PrismRetrievedChunk] = []
        for chunk_id, score in ranked:
            tokens = self.read_chunk_tokens_by_id(chunk_id)
            metadata = self.chunk_info(chunk_id) or PrismChunkMetadata(chunk_id, 0, len(tokens))
            result.append(PrismRetrievedChunk(chunk_id, score, tokens, metadata))
        return result

    def stream_tokens_for_query(self, query_tokens: List[int]) -> List[int]:
        seen = set()
        tokens: List[int] = []
        for token in query_tokens:
            entry = self.lookup_chunk_for_token(token)
            if not entry:
                continue
            chunk_id, _ = entry
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            tokens.extend(self.read_chunk_tokens_by_id(chunk_id))
        return tokens

    @staticmethod
    def _decompress_payload(compressed: bytes) -> List[int]:
        palette_size = struct.unpack("<I", compressed[:4])[0]
        start = 4 + palette_size
        raw = zlib.decompress(compressed[start:])
        count = len(raw) // 4
        return list(struct.unpack(f"<{count}I", raw))
