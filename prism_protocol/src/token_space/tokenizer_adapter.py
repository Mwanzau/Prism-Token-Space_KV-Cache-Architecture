from typing import Protocol, runtime_checkable
from pathlib import Path

from prism_protocol.src.token_space.tokenizer_bridge import TokenizerBridge


@runtime_checkable
class TokenizerLike(Protocol):
    def encode(self, text: str, progress_callback=None):
        ...

    def decode(self, tokens):
        ...


class GGUFTokenizerAdapter:
    def __init__(self, gguf_path: str):
        self._bridge = TokenizerBridge.from_gguf(gguf_path)
        self.vocab = self._bridge.vocab

    def encode(self, text: str, progress_callback=None):
        if progress_callback:
            return self._bridge.encode(text, progress_callback=progress_callback)
        return self._bridge.encode(text)

    def decode(self, tokens):
        return self._bridge.decode(tokens)


class TransformersTokenizerAdapter:
    def __init__(self, identifier: str):
        try:
            from transformers import PreTrainedTokenizerFast
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError("transformers is required for HF tokenizer support") from exc
        # allow local directories or HF ids
        self._tok = PreTrainedTokenizerFast.from_pretrained(identifier)
        try:
            self.vocab = self._tok.get_vocab()
        except Exception:
            self.vocab = {}

    def encode(self, text: str, progress_callback=None):
        # transformers tokenizer returns list[int]
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        return self._tok.decode(tokens, clean_up_tokenization_spaces=True)


class TokenizersFileAdapter:
    def __init__(self, file_path: str):
        try:
            from tokenizers import Tokenizer as HFTokenizer
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError("tokenizers library is required to load tokenizer.json directly") from exc
        self._tok = HFTokenizer.from_file(file_path)

    def encode(self, text: str, progress_callback=None):
        out = self._tok.encode(text)
        return out.ids

    def decode(self, tokens):
        return self._tok.decode(tokens)


def get_tokenizer_adapter(gguf_path: str) -> TokenizerLike:
    """Return an adapter for the provided tokenizer resource.

    Supported inputs:
    - strings starting with "hf:<id_or_dir>" to load a HuggingFace tokenizer via `transformers`.
    - paths to a `tokenizer.json` file (loaded via `tokenizers` library).
    - otherwise falls back to the GGUF `TokenizerBridge` adapter.
    """
    path = str(gguf_path)
    # HF id prefix
    if path.startswith("hf:"):
        identifier = path.split(":", 1)[1]
        return TransformersTokenizerAdapter(identifier)

    p = Path(path)
    # tokenizer.json file
    if p.is_file() and p.name.endswith("tokenizer.json"):
        return TokenizersFileAdapter(path)

    # directory containing tokenizer.json
    if p.is_dir() and (p / "tokenizer.json").exists():
        return TokenizersFileAdapter(str(p / "tokenizer.json"))

    # otherwise, fallback to GGUF adapter
    return GGUFTokenizerAdapter(path)
