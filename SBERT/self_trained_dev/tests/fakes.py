"""Deterministic fake encoder so engine / session tests run without any model files."""
import hashlib

import numpy as np

from lexical.text import clean_text


class FakeEncoder:
    """Mean of per-word hash vectors; words in `unknown` have no vector (like an OOV word)."""

    def __init__(self, dim: int = 32, unknown=("qwzxv",)):
        self.dim = dim
        self.unknown = set(unknown)

    def identity(self) -> str:
        return f"fake|{self.dim}"

    def _word(self, word: str) -> np.ndarray:
        seed = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).standard_normal(self.dim)

    def encode(self, sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False, **kw):
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = [w for w in clean_text(text).split() if w not in self.unknown]
            if words:
                v = np.mean([self._word(w) for w in words], axis=0)
                out[i] = v / np.linalg.norm(v)
        return out[0] if single else out

    def is_known(self, text: str) -> bool:
        return any(w not in self.unknown for w in clean_text(text).split())
