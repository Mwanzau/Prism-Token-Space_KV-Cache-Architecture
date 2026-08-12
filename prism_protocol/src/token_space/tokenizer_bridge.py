import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


TOKENIZER_SEPARATOR = re.compile(rb"[\x00-\x0f]\x00{7}")


def hash_vocab(vocab: Dict[str, int]) -> int:
    ordered = sorted(vocab.items(), key=lambda item: item[1])
    joined = b"\n".join(f"{token}:{idx}".encode() for token, idx in ordered)
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _normalize_token_bytes(token_bytes: bytes) -> str:
    token_bytes = token_bytes.split(b"\x00", 1)[0]
    token = token_bytes.decode("utf-8", errors="ignore")
    token = token.strip()
    return token


def _decode_token_block(block: bytes) -> List[str]:
    raw_parts = TOKENIZER_SEPARATOR.split(block)
    tokens: List[str] = []
    for part in raw_parts:
        token = _normalize_token_bytes(part)
        if token:
            tokens.append(token)
    return tokens


def _load_gguf_tokens(gguf_path: str) -> List[str]:
    data = Path(gguf_path).read_bytes()
    if not data.startswith(b"GGUF"):
        raise ValueError("Not a valid GGUF file")

    def parse_entry(key_name: bytes) -> Optional[List[str]]:
        idx = data.find(key_name)
        if idx == -1:
            return None

        pos = idx + len(key_name)
        if len(data) < pos + 16:
            return None

        value_type = int.from_bytes(data[pos:pos + 4], "little")
        pointer = int.from_bytes(data[pos + 8:pos + 16], "little")

        if pointer and 0 < pointer < len(data):
            block = data[pointer:pointer + 2_000_000]
            tokens = _decode_token_block(block)
            if tokens:
                return tokens

        inline_block = data[pos + 16:pos + 2_000_000]
        return _decode_token_block(inline_block)

    token_keys = [b"tokenizer.ggml.tokens", b"vocab"]
    for token_key in token_keys:
        tokens = parse_entry(token_key)
        if tokens:
            # preserve the first appearance order while removing duplicates
            seen = set()
            unique_tokens: List[str] = []
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    unique_tokens.append(token)
            return unique_tokens

    raise ValueError("Could not find tokenizer token list in GGUF file")


@dataclass
class TokenizerBridge:
    vocab: Dict[str, int]
    reverse_vocab: Dict[int, str]
    model_vocab_hash: int
    sorted_tokens: List[str]
    oov_base: int

    def __post_init__(self):
        self.sorted_tokens = sorted(self.vocab.keys(), key=len, reverse=True)
        self.oov_base = max(self.vocab.values(), default=0) + 1
        self.token_trie = self._build_token_trie()

    @classmethod
    def from_tokens(cls, tokens: List[str]) -> "TokenizerBridge":
        unique_tokens = []
        seen = set()
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        vocab = {token: idx + 1 for idx, token in enumerate(unique_tokens)}
        reverse_vocab = {idx: token for token, idx in vocab.items()}
        return cls(
            vocab=vocab,
            reverse_vocab=reverse_vocab,
            model_vocab_hash=hash_vocab(vocab),
            sorted_tokens=[],
            oov_base=0,
        )

    def _build_token_trie(self) -> Dict[str, object]:
        root: Dict[str, object] = {}
        for token, token_id in self.vocab.items():
            node = root
            for ch in token:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            node["__id__"] = token_id
        return root

    @classmethod
    def from_gguf(cls, gguf_path: str) -> "TokenizerBridge":
        tokens = _load_gguf_tokens(gguf_path)
        if not tokens:
            raise ValueError("GGUF tokenizer yielded no tokens")
        return cls.from_tokens(tokens)

    def encode(self, text: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[int]:
        normalized = text.replace("\n", " ").replace(" ", "Ġ")
        output: List[int] = []
        total = len(normalized)
        i = 0
        last_pct = -1

        while i < total:
            node = self.token_trie
            j = i
            last_match_id = None
            last_match_len = 0

            while j < total and normalized[j] in node:
                node = node[normalized[j]]
                j += 1
                if "__id__" in node:
                    last_match_id = node["__id__"]
                    last_match_len = j - i

            if last_match_id is not None:
                output.append(last_match_id)
                i += last_match_len
            else:
                ch = normalized[i]
                output.append(self.oov_base + ord(ch))
                i += 1

            if progress_callback is not None:
                pct = min(100, max(0, i * 100 // total if total else 100))
                if pct != last_pct:
                    last_pct = pct
                    progress_callback(i, total)

        return output

    def decode(self, token_ids: List[int]) -> str:
        parts: List[str] = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                parts.append(self.reverse_vocab[token_id])
            elif token_id >= self.oov_base:
                # OOV encoded as oov_base + ord(ch)
                try:
                    parts.append(chr(token_id - self.oov_base))
                except Exception:
                    parts.append("[UNK]")
            else:
                parts.append("[UNK]")

        # Reconstruct text: tokens often include the Ġ marker for spaces.
        text = "".join(parts)
        text = text.replace("Ġ", " ")
        # collapse multiple spaces
        return " ".join(text.split())

    def _encode_oov(self, token: str) -> List[int]:
        return [self.oov_base + ord(ch) for ch in token]

    def match_vocab_hash(self, hash_value: int) -> bool:
        return self.model_vocab_hash == hash_value
