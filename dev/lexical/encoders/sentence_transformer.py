"""SentenceTransformerEncoder: pretrained MiniLM (or a fine-tuned checkpoint) behind the engine's encode() call.

sentence-transformers / torch are imported only here, so other backends run without them.
MiniLM has a subword tokenizer, so every input gets a vector: no is_known() (same as the notebook).
"""
import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEncoder:
    """Wraps SentenceTransformer.encode with an LRU cache of phrase vectors."""

    def __init__(self, model=None, model_name_or_path: str = DEFAULT_MODEL, cache_folder=None,
                 device: str = "cpu", torch_threads: int = None, cache_size: int = 50_000):
        if model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            if torch_threads:
                torch.set_num_threads(int(torch_threads))
            model = SentenceTransformer(str(model_name_or_path), device=device,
                                        cache_folder=None if cache_folder is None else str(cache_folder))
        self.model = model
        self.name = str(model_name_or_path)
        self.dim = int(model.get_sentence_embedding_dimension())
        self._cache, self._cache_size = OrderedDict(), cache_size
        self._lock = threading.Lock()   # the LRU cache is shared between request threads

    @classmethod
    def from_dir(cls, folder, cache_folder=None, device: str = "cpu", torch_threads: int = None):
        """Load the model named in manifest.json (a local fine-tuned 'model/' folder wins)."""
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        local = folder / "model"
        source = local if local.is_dir() else manifest.get("model_id", DEFAULT_MODEL)
        cache_folder = cache_folder or os.environ.get("LEXICAL_ST_CACHE_DIR") or folder / "hf_cache"
        torch_threads = torch_threads or os.environ.get("LEXICAL_TORCH_THREADS")
        encoder = cls(model_name_or_path=source, cache_folder=cache_folder, device=device,
                      torch_threads=torch_threads)
        if manifest.get("dim") and encoder.dim != manifest["dim"]:
            raise ValueError(f"Model dim {encoder.dim} != exported dim {manifest['dim']}")
        return encoder

    def identity(self) -> str:
        return f"sentence_transformer|{self.name}|{self.dim}"

    def encode(self, sentences, convert_to_numpy: bool = True, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, batch_size: int = 64, **kwargs):
        """L2-normalised vectors, shape (n, dim) (or (dim,) for one string); uncached texts in one batch."""
        single = isinstance(sentences, str)
        texts = [sentences] if single else [str(t) for t in sentences]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        missing = []
        with self._lock:
            for i, text in enumerate(texts):
                hit = self._cache.get((text, normalize_embeddings))
                if hit is None:
                    missing.append(i)
                else:
                    self._cache.move_to_end((text, normalize_embeddings))
                    out[i] = hit
        if missing:
            new = self.model.encode([texts[i] for i in missing], convert_to_numpy=True,
                                    normalize_embeddings=normalize_embeddings, show_progress_bar=False,
                                    batch_size=batch_size)
            new = np.asarray(new, dtype=np.float32)
            with self._lock:
                for i, vector in zip(missing, new):
                    out[i] = vector
                    self._cache[(texts[i], normalize_embeddings)] = vector
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return out[0] if single else out
