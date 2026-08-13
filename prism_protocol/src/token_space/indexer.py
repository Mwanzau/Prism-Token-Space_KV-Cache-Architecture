import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union


@dataclass
class TokenIndexEntry:
    token_id: int
    chunk_id: int
    frequency: int


class PrismIndexer:
    def __init__(self):
        self.token_to_chunks: Dict[int, List[int]] = {}
        self.frequency: Counter[int] = Counter()
        self.chunks: Dict[int, List[int]] = {}

    def add_chunk(self, chunk_id: int, token_ids: List[int]) -> None:
        unique_tokens = set(token_ids)
        for token_id in unique_tokens:
            self.token_to_chunks.setdefault(token_id, []).append(chunk_id)
        self.frequency.update(token_ids)
        self.chunks[chunk_id] = list(token_ids)

    def build_sparse_index(self, chunks: Iterable[List[int]]) -> Dict[int, TokenIndexEntry]:
        self.token_to_chunks.clear()
        self.frequency.clear()
        self.chunks.clear()
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

    @staticmethod
    def calculate_bm25_idf(total_chunks: int, doc_freq: int) -> float:
        if doc_freq <= 0 or total_chunks <= 0:
            return 0.0
        return math.log((total_chunks - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    @staticmethod
    def rank_chunks(
        query_tokens: List[int],
        total_chunks: int,
        get_doc_freq: Callable[[int], int],
        get_candidate_chunks: Callable[[List[int]], Iterable[int]],
        get_chunk_tokens: Callable[[int], List[int]],
        get_chunk_length: Optional[Callable[[int], int]] = None,
        k1: float = 1.5,
        b: float = 0.75,
        proximity_window: int = 25,
        proximity_boost: float = 2.5,
    ) -> Dict[int, float]:
        if not query_tokens or total_chunks <= 0:
            return {}

        query_counts = Counter(query_tokens)
        unique_query_tokens = list(query_counts.keys())

        # 1. Calculate IDF for each query token
        idf_map: Dict[int, float] = {}
        doc_freq_map: Dict[int, int] = {}
        for qt in unique_query_tokens:
            df = get_doc_freq(qt)
            doc_freq_map[qt] = df
            idf_map[qt] = PrismIndexer.calculate_bm25_idf(total_chunks, df)

        candidate_chunk_ids = set(get_candidate_chunks(unique_query_tokens))
        if not candidate_chunk_ids:
            return {}

        lengths: Dict[int, int] = {}
        for cid in candidate_chunk_ids:
            if get_chunk_length:
                lengths[cid] = get_chunk_length(cid)
            else:
                tokens = get_chunk_tokens(cid)
                lengths[cid] = len(tokens)

        total_len = sum(lengths.values())
        avg_doc_len = (total_len / len(lengths)) if lengths else 1.0
        if avg_doc_len <= 0:
            avg_doc_len = 1.0

        max_allowed_df = max(1, int(0.5 * total_chunks))
        proximity_query_tokens = [qt for qt in unique_query_tokens if doc_freq_map.get(qt, 0) <= max_allowed_df]
        if not proximity_query_tokens:
            proximity_query_tokens = unique_query_tokens

        scores: Dict[int, float] = {}

        for chunk_id in candidate_chunk_ids:
            chunk_tokens = get_chunk_tokens(chunk_id)
            doc_len = len(chunk_tokens)
            if doc_len == 0:
                continue

            chunk_token_counts = Counter(chunk_tokens)

            score = 0.0
            matched_q_tokens = []
            for qt in unique_query_tokens:
                f = chunk_token_counts.get(qt, 0)
                if f > 0:
                    matched_q_tokens.append(qt)
                    idf = idf_map[qt]
                    num = f * (k1 + 1.0)
                    denom = f + k1 * (1.0 - b + b * (doc_len / avg_doc_len))
                    score += idf * (num / denom)

            if score <= 0.0:
                scores[chunk_id] = 0.0
                continue

            boost = 1.0
            content_matched = [qt for qt in matched_q_tokens if qt in proximity_query_tokens]
            if len(content_matched) >= 2:
                pos_map: Dict[int, List[int]] = {}
                for idx, token_id in enumerate(chunk_tokens):
                    if token_id in content_matched:
                        pos_map.setdefault(token_id, []).append(idx)

                has_proximity = False
                matched_term_ids = list(pos_map.keys())
                for i in range(len(matched_term_ids)):
                    for j in range(i + 1, len(matched_term_ids)):
                        t1 = matched_term_ids[i]
                        t2 = matched_term_ids[j]
                        for p1 in pos_map[t1]:
                            for p2 in pos_map[t2]:
                                dist = abs(p1 - p2)
                                if dist <= proximity_window:
                                    has_proximity = True
                                    break
                            if has_proximity:
                                break
                        if has_proximity:
                            break
                    if has_proximity:
                        break

                if has_proximity:
                    boost = proximity_boost

            scores[chunk_id] = score * boost

        return scores


class IndexMerger:
    @staticmethod
    def merge(existing_index: Dict[int, List[Tuple[int, int]]], new_index: Dict[int, List[Tuple[int, int]]]) -> Dict[int, List[Tuple[int, int]]]:
        merged: Dict[int, List[Tuple[int, int]]] = {}
        for token, entries in existing_index.items():
            merged[token] = list(entries)
        for token, entries in new_index.items():
            merged.setdefault(token, []).extend(entries)
        return merged

