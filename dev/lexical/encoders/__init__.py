"""Encoders: objects with encode(texts) -> L2-normalised vectors."""
from .base import Encoder
from .sentence_transformer import SentenceTransformerEncoder

__all__ = ["Encoder", "SentenceTransformerEncoder"]
