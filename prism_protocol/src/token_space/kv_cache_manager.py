from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class KVCacheEntry:
    chunk_id: int
    key_bytes: bytes
    value_bytes: bytes


class KVCacheManager:
    def __init__(self):
        self.entries: Dict[int, KVCacheEntry] = {}

    def add_entry(self, chunk_id: int, key_bytes: bytes, value_bytes: bytes) -> None:
        self.entries[chunk_id] = KVCacheEntry(chunk_id, key_bytes, value_bytes)

    def export(self) -> Dict[int, bytes]:
        return {
            chunk_id: key.key_bytes + key.value_bytes
            for chunk_id, key in self.entries.items()
        }

    def import_from(self, raw: Dict[int, bytes]) -> None:
        for chunk_id, payload in raw.items():
            half = len(payload) // 2
            self.entries[chunk_id] = KVCacheEntry(chunk_id, payload[:half], payload[half:])

    def get_entry(self, chunk_id: int) -> Optional[KVCacheEntry]:
        return self.entries.get(chunk_id)
