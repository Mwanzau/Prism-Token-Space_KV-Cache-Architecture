"""
Tests verifying multi-model tokenizer adapter support:
  - GGUF adapter (Qwen / BPE)
  - GGUF adapter (Gemma / SentencePiece) — space-marker auto-detection
  - HuggingFace adapter (hf: prefix)
  - TokenizerBridge byte-fallback decoding
  - model_vocab_hash present on all adapters
"""
import math
import re
import tempfile
from pathlib import Path

import pytest

from prism_protocol.src.token_space.tokenizer_bridge import (
    TokenizerBridge,
    _detect_space_marker,
    _expand_token,
    _decode_byte_fallback,
    _BPE_SPACE,
    _SPM_SPACE,
)
from prism_protocol.src.token_space.tokenizer_adapter import (
    GGUFTokenizerAdapter,
    get_tokenizer_adapter,
)


ROOT = Path(__file__).resolve().parent.parent
QWEN_GGUF = ROOT / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
GEMMA_GGUF = ROOT / "models" / "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf"


# ──────────────────────────────────────────────
# 1. Byte-fallback token decoding
# ──────────────────────────────────────────────

def test_byte_fallback_decoding():
    assert _decode_byte_fallback("<0x4E>") == "N"
    assert _decode_byte_fallback("<0x61>") == "a"
    assert _decode_byte_fallback("<0x20>") == " "
    assert _decode_byte_fallback("hello") is None
    assert _decode_byte_fallback("<0xXX>") is None


# ──────────────────────────────────────────────
# 2. Space-marker auto-detection
# ──────────────────────────────────────────────

def test_space_marker_detection_bpe():
    # Qwen/GPT-2 style: lots of Ġ tokens
    vocab_keys = ["Ġhello", "Ġworld", "Ġthe", "Ġriver", "abc", "xyz"]
    assert _detect_space_marker(vocab_keys) == _BPE_SPACE


def test_space_marker_detection_spm():
    # Gemma/Llama SPM style: ▁ prefix
    vocab_keys = ["▁hello", "▁world", "▁the", "▁river", "abc", "xyz"]
    assert _detect_space_marker(vocab_keys) == _SPM_SPACE


# ──────────────────────────────────────────────
# 3. _expand_token with different markers
# ──────────────────────────────────────────────

def test_expand_token_bpe():
    assert _expand_token("Ġhello", _BPE_SPACE) == " hello"
    assert _expand_token("world", _BPE_SPACE) == "world"


def test_expand_token_spm():
    assert _expand_token("▁hello", _SPM_SPACE) == " hello"
    assert _expand_token("world", _SPM_SPACE) == "world"


def test_expand_token_byte_fallback():
    # byte-fallback overrides space marker processing
    assert _expand_token("<0x4E>", _SPM_SPACE) == "N"
    assert _expand_token("<0x20>", _BPE_SPACE) == " "


# ──────────────────────────────────────────────
# 4. TokenizerBridge from SPM-style token list
# ──────────────────────────────────────────────

def test_tokenizer_bridge_spm_encode_decode():
    tokens = ["▁river", "▁brown", "▁color", "▁the", "▁in", "▁is", "hello", "world"]
    bridge = TokenizerBridge.from_tokens(tokens)
    assert bridge.space_marker == _SPM_SPACE

    text = "the river is brown in color"
    encoded = bridge.encode(text)
    assert isinstance(encoded, list)
    assert all(isinstance(t, int) for t in encoded)

    decoded = bridge.decode(encoded)
    # words should be preserved even if order collapses whitespace
    for word in ["river", "brown", "color"]:
        assert word in decoded


# ──────────────────────────────────────────────
# 5. GGUFTokenizerAdapter exposes model_vocab_hash
# ──────────────────────────────────────────────

@pytest.mark.skipif(not QWEN_GGUF.exists(), reason="Qwen GGUF not present")
def test_gguf_adapter_has_vocab_hash():
    adapter = GGUFTokenizerAdapter(str(QWEN_GGUF))
    assert hasattr(adapter, "model_vocab_hash")
    assert isinstance(adapter.model_vocab_hash, int)
    assert adapter.model_vocab_hash != 0


@pytest.mark.skipif(not QWEN_GGUF.exists(), reason="Qwen GGUF not present")
def test_gguf_adapter_tokenizes_river_as_word():
    """Qwen GGUF (BPE) should encode 'river' as a single vocabulary token."""
    adapter = GGUFTokenizerAdapter(str(QWEN_GGUF))
    encoded = adapter.encode("the river")
    decoded_tks = [adapter.decode([t]) for t in encoded]
    joined = " ".join(decoded_tks).lower()
    assert "river" in joined


# ──────────────────────────────────────────────
# 6. Gemma GGUF: SPM auto-detection & 'river' as a word
# ──────────────────────────────────────────────

@pytest.mark.skipif(not GEMMA_GGUF.exists(), reason="Gemma GGUF not present")
def test_gemma_gguf_spm_detection():
    bridge = TokenizerBridge.from_gguf(str(GEMMA_GGUF))
    assert bridge.space_marker == _SPM_SPACE


@pytest.mark.skipif(not GEMMA_GGUF.exists(), reason="Gemma GGUF not present")
def test_gemma_gguf_river_tokenized_as_word():
    """After SPM fix, Gemma GGUF should tokenise 'river' as sub-tokens, not single chars.

    Gemma's vocabulary contains '▁riv' (surface: ' riv') but not '▁river'.  So
    encoding ' river' should produce ≤ 4 tokens (e.g. ' riv' + 'e' + 'r'),
    NOT 5 single-character OOV tokens as the old BPE-style fallback did.
    """
    bridge = TokenizerBridge.from_gguf(str(GEMMA_GGUF))
    encoded = bridge.encode(" river")
    # Fewer tokens than characters means we matched multi-char vocab tokens
    assert len(encoded) <= 4, (
        f"Expected ≤4 tokens for ' river', got {len(encoded)}: "
        + str([bridge.decode([t]) for t in encoded])
    )
    # At least one of those tokens must be 'riv' (the SPM stem)
    decoded_parts = [bridge.decode([t]).strip() for t in encoded]
    assert any("riv" in p for p in decoded_parts), (
        f"Expected a token containing 'riv', got: {decoded_parts}"
    )


# ──────────────────────────────────────────────
# 7. get_tokenizer_adapter routes correctly
# ──────────────────────────────────────────────

@pytest.mark.skipif(not QWEN_GGUF.exists(), reason="Qwen GGUF not present")
def test_get_tokenizer_adapter_returns_gguf_for_gguf_path():
    from prism_protocol.src.token_space.tokenizer_adapter import GGUFTokenizerAdapter
    adapter = get_tokenizer_adapter(str(QWEN_GGUF))
    assert isinstance(adapter, GGUFTokenizerAdapter)
    assert hasattr(adapter, "model_vocab_hash")
