import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

# BPE space marker (GPT-2 / Qwen / Llama style)
_BPE_SPACE = "Ġ"
# SentencePiece space marker (Gemma / Llama-SPM / Mistral style)
_SPM_SPACE = "▁"

# Matches SentencePiece byte-fallback tokens like <0x4E>
_BYTE_FALLBACK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

TOKENIZER_SEPARATOR = re.compile(rb"[\x00-\x0f]\x00{7}")


def hash_vocab(vocab: Dict[str, int]) -> int:
    ordered = sorted(vocab.items(), key=lambda item: item[1])
    joined = b"\n".join(f"{token}:{idx}".encode() for token, idx in ordered)
    digest = hashlib.sha256(joined).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _decode_byte_fallback(token: str) -> Optional[str]:
    """If token is a SentencePiece byte-fallback like <0x4E>, return the character."""
    m = _BYTE_FALLBACK_RE.match(token)
    if m:
        try:
            return bytes([int(m.group(1), 16)]).decode("utf-8", errors="replace")
        except Exception:
            return None
    return None


def _normalize_token_bytes(token_bytes: bytes) -> str:
    """Strip null bytes and decode raw GGUF token bytes to a Python string."""
    token_bytes = token_bytes.split(b"\x00", 1)[0]
    token = token_bytes.decode("utf-8", errors="ignore")
    return token.strip()


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
            # preserve first-appearance order while deduplicating
            seen: set = set()
            unique_tokens: List[str] = []
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    unique_tokens.append(token)
            return unique_tokens

    raise ValueError("Could not find tokenizer token list in GGUF file")


def _detect_space_marker(vocab_keys: List[str]) -> str:
    """Auto-detect whether the vocabulary uses BPE (Ġ) or SPM (▁) space markers."""
    bpe_count = sum(1 for t in vocab_keys if _BPE_SPACE in t)
    spm_count = sum(1 for t in vocab_keys if _SPM_SPACE in t)
    if spm_count > bpe_count:
        return _SPM_SPACE
    return _BPE_SPACE


def _expand_token(token: str, space_marker: str) -> str:
    """Expand a vocab token into its canonical text form.

    * Byte-fallback tokens (<0x4E>) → raw character.
    * SPM space marker (▁) → space.
    * BPE space marker (Ġ) → space.
    """
    # Byte-fallback e.g. <0x4E>
    ch = _decode_byte_fallback(token)
    if ch is not None:
        return ch
    # Replace space markers with actual spaces
    return token.replace(space_marker, " ")


@dataclass
class TokenizerBridge:
    vocab: Dict[str, int]
    reverse_vocab: Dict[int, str]
    model_vocab_hash: int
    sorted_tokens: List[str]
    oov_base: int
    space_marker: str = field(default=_BPE_SPACE)

    def __post_init__(self):
        self.sorted_tokens = sorted(self.vocab.keys(), key=len, reverse=True)
        self.oov_base = max(self.vocab.values(), default=0) + 1
        # Build a trie over EXPANDED token text for fast longest-match encoding
        self._surface_to_id: Dict[str, int] = {}
        for token, token_id in self.vocab.items():
            surface = _expand_token(token, self.space_marker)
            # keep first mapping (lower id wins on collision)
            if surface and surface not in self._surface_to_id:
                self._surface_to_id[surface] = token_id
        self.token_trie = self._build_token_trie()

    @classmethod
    def from_tokens(cls, tokens: List[str]) -> "TokenizerBridge":
        unique_tokens = []
        seen: set = set()
        for token in tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)
        vocab = {token: idx + 1 for idx, token in enumerate(unique_tokens)}
        reverse_vocab = {idx: token for token, idx in vocab.items()}
        space_marker = _detect_space_marker(unique_tokens)
        return cls(
            vocab=vocab,
            reverse_vocab=reverse_vocab,
            model_vocab_hash=hash_vocab(vocab),
            sorted_tokens=[],
            oov_base=0,
            space_marker=space_marker,
        )

    def _build_token_trie(self) -> Dict[str, object]:
        """Build a character trie over the *surface* text representations of tokens."""
        root: Dict[str, object] = {}
        for surface, token_id in self._surface_to_id.items():
            node = root
            for ch in surface:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            # only set id if not already taken by a longer match that happens to share surface
            if "__id__" not in node:
                node["__id__"] = token_id
        return root

    @classmethod
    def from_gguf(cls, gguf_path: str) -> "TokenizerBridge":
        tokens = _load_gguf_tokens(gguf_path)
        if not tokens:
            raise ValueError("GGUF tokenizer yielded no tokens")
        return cls.from_tokens(tokens)

    def encode(self, text: str, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[int]:
        """Encode plain text using longest-match trie over surface-form tokens."""
        # Normalise line breaks → space
        normalized = text.replace("\n", " ").replace("\r", " ")
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
                # OOV: encode character by character using oov_base + codepoint
                output.append(self.oov_base + ord(normalized[i]))
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
                raw_token = self.reverse_vocab[token_id]
                parts.append(_expand_token(raw_token, self.space_marker))
            elif token_id >= self.oov_base:
                try:
                    parts.append(chr(token_id - self.oov_base))
                except Exception:
                    parts.append("[UNK]")
            else:
                parts.append("[UNK]")

        text = "".join(parts)
        # collapse multiple spaces
        return " ".join(text.split())

    def _encode_oov(self, token: str) -> List[int]:
        return [self.oov_base + ord(ch) for ch in token]

    def match_vocab_hash(self, hash_value: int) -> bool:
        return self.model_vocab_hash == hash_value

