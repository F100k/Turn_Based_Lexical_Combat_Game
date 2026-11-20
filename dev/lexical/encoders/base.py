"""Encoder interface expected by LexicalCombatEngine (same call shape as SentenceTransformer.encode)."""
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    """encode() returns L2-normalised vectors: (n, dim) for a list, (dim,) for one string.

    Encoders may also define is_known(text) -> bool; the engine then reports unknown words
    instead of scoring a zero vector.
    """

    def encode(self, sentences, convert_to_numpy: bool = True, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, **kwargs) -> np.ndarray: ...

    def identity(self) -> str: ...
