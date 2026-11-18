"""Encoders: objects with encode(texts) -> L2-normalised vectors."""
from .base import Encoder
from .word_vectors import PhraseMerger, WordVectorEncoder

__all__ = ["Encoder", "PhraseMerger", "WordVectorEncoder"]
