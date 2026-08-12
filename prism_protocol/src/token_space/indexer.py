from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class TokenIndexEntry:
    token_id: int
    chunk_id: int
    frequency: int


class PrismIndexer:
    def __init__(self):
        self.token_to_chunks: Dict[int, List[int]] = {}
        self.frequency: Counter[int] = Counter()

    def add_chunk(self, chunk_id: int, token_ids: List[int]) -> None:
        unique_tokens = set(token_ids)
        for token_id in unique_tokens:
            self.token_to_chunks.setdefault(token_id, []).append(chunk_id)
        self.frequency.update(token_ids)

    def build_sparse_index(self, chunks: Iterable[List[int]]) -> Dict[int, TokenIndexEntry]:
        self.token_to_chunks.clear()
        self.frequency.clear()
        for chunk_id, chunk in enumerate(chunks):
            self.add_chunk(chunk_id, chunk)

        return {
            token_id: TokenIndexEntry(token_id, chunk_ids[0], self.frequency[token_id])
            for token_id, chunk_ids in self.token_to_chunks.items()
        }

    def lookup(self, token_id: int) -> List[int]:
        return self.token_to_chunks.get(token_id, [])

    def most_frequent(self, top_n: int = 64) -> List[TokenIndexEntry]:
        return [TokenIndexEntry(token, self.token_to_chunks[token][0], count) for token, count in self.frequency.most_common(top_n)]


class IndexMerger:
    @staticmethod
    def merge(existing_index: Dict[int, List[Tuple[int, int]]], new_index: Dict[int, List[Tuple[int, int]]]) -> Dict[int, List[Tuple[int, int]]]:
        merged: Dict[int, List[Tuple[int, int]]] = {}
        for token, entries in existing_index.items():
            merged[token] = list(entries)
        for token, entries in new_index.items():
            merged.setdefault(token, []).extend(entries)
        return merged
