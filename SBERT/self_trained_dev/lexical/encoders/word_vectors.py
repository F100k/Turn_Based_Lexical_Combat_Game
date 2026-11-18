"""WordVectorEncoder: phrase vectors from exported word vectors (numpy only, no gensim).

Re-implements the notebook's SelfTrainedEncoder (Section 5.1):
clean_text -> whole phrase as one token if known -> bigram / trigram phrase merges (same algorithm as
gensim FrozenPhrases) -> split unknown merged tokens -> mean or SIF composition -> L2 normalisation.
"""
import gzip
import json
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..text import clean_text


class PhraseMerger:
    """Greedy left-to-right phrase merging, identical to gensim FrozenPhrases[tokens]."""

    def __init__(self, phrasegrams: dict, threshold: float, delimiter: str = "_", connector_words=()):
        self.phrasegrams = phrasegrams
        self.threshold = float(threshold)
        self.delimiter = delimiter
        self.connector_words = frozenset(connector_words)

    def _score(self, word_a: str, word_b: str, in_between: list):
        phrase = self.delimiter.join([word_a] + in_between + [word_b])
        score = self.phrasegrams.get(phrase, float("-inf"))
        return phrase if score > self.threshold else None

    def __getitem__(self, tokens: list) -> list:
        out, start, in_between = [], None, []
        for word in tokens:
            if word not in self.connector_words:
                if start:
                    phrase = self._score(start, word, in_between)
                    if phrase is not None:
                        out.append(phrase)
                        start, in_between = None, []
                    else:
                        out.append(start)
                        out.extend(in_between)
                        start, in_between = word, []
                else:
                    start, in_between = word, []
            elif start:
                in_between.append(word)
            else:
                out.append(word)
        if start:
            out.append(start)
            out.extend(in_between)
        return out


class WordVectorEncoder:
    """Word vectors -> phrase vectors with the SentenceTransformer.encode() call shape."""

    def __init__(self, words, vectors, counts: dict, total: float, phrasers=(), composition: str = "sif",
                 common_component=None, stopwords=(), sif_a: float = 1e-3, name: str = "word_vectors",
                 cache_size: int = 50_000):
        self.name, self.composition = name, composition
        self.words = list(words)
        self.key_to_index = {w: i for i, w in enumerate(self.words)}
        vectors = np.asarray(vectors, dtype=np.float32)
        self.unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        self.dim = int(self.unit.shape[1])
        self.counts, self.total = counts, float(total) or 1.0
        self.phrasers = tuple(phrasers)
        self.common_component = None if common_component is None else np.asarray(common_component, np.float32)
        self.stopwords = frozenset(stopwords)
        self.sif_a = float(sif_a)
        self._cache, self._cache_size = OrderedDict(), cache_size
        self._lock = threading.Lock()   # the LRU cache is shared between request threads

    @classmethod
    def from_dir(cls, folder, cache_size: int = 50_000) -> "WordVectorEncoder":
        """Load the files written by the notebook's web-export cell."""
        folder = Path(folder)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        words = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        vectors = np.load(folder / "vectors.npy")
        if vectors.shape != (len(words), manifest["dim"]):
            raise ValueError(f"vectors.npy shape {vectors.shape} does not match vocab / manifest dim")
        counts_file = json.loads((folder / "counts.json").read_text(encoding="utf-8"))
        with gzip.open(folder / "phrases.json.gz", "rt", encoding="utf-8") as handle:
            phrase_models = json.load(handle)
        phrasers = [PhraseMerger(p["phrasegrams"], p["threshold"], p["delimiter"], p["connector_words"])
                    for p in phrase_models]
        component_path = folder / "common_component.npy"
        component = np.load(component_path) if component_path.is_file() else None
        stopwords = json.loads((folder / "stopwords.json").read_text(encoding="utf-8"))
        return cls(words, vectors, counts_file["counts"], counts_file["total"], phrasers,
                   manifest["composition"], component, stopwords, manifest["sif_a"],
                   name=manifest.get("model", "word_vectors"), cache_size=cache_size)

    def identity(self) -> str:
        return f"{self.name}|{self.composition}|{len(self.words)}x{self.dim}"

    # ---- tokens and word vectors --------------------------------------------
    def tokens_for(self, cleaned: str) -> list:
        """Whole phrase as one token if known, else phrase merges, else single words."""
        words = cleaned.split()
        if not words:
            return []
        joined = "_".join(words)
        if len(words) > 1 and joined in self.key_to_index:
            return [joined]
        tokens = words
        for phraser in self.phrasers:
            tokens = phraser[tokens]
        out = []
        for t in tokens:
            if t in self.key_to_index or "_" not in t:
                out.append(t)
            else:
                out.extend(t.split("_"))
        return out

    def word_vector(self, token: str):
        i = self.key_to_index.get(token)
        return None if i is None else self.unit[i]

    def weight(self, token: str) -> float:
        """SIF weight a / (a + p(w)); 1.0 for the plain mean."""
        if self.composition != "sif":
            return 1.0
        return self.sif_a / (self.sif_a + self.counts.get(token, 0) / self.total)

    def _compose(self, tokens: list):
        """(vector before normalisation, known tokens, content tokens)."""
        if self.composition == "sif":
            content = [t for t in tokens if t not in self.stopwords] or tokens
        else:
            content = tokens
        vecs, weights = [], []
        for t in content:
            v = self.word_vector(t)
            if v is not None:
                vecs.append(v)
                weights.append(self.weight(t))
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32), 0, len(content)
        v = np.average(np.vstack(vecs), axis=0, weights=np.asarray(weights, dtype=np.float64))
        if self.composition == "sif" and self.common_component is not None:
            v = v - float(v @ self.common_component) * self.common_component
        return v.astype(np.float32), len(vecs), len(content)

    def _vector(self, text: str):
        with self._lock:
            hit = self._cache.get(text)
            if hit is not None:
                self._cache.move_to_end(text)
                return hit
        hit = self._compose(self.tokens_for(clean_text(text)))
        with self._lock:
            self._cache[text] = hit
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return hit

    # ---- SentenceTransformer-compatible API -----------------------------------
    def encode(self, sentences, convert_to_numpy: bool = True, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, batch_size: int = None, **kwargs):
        """L2-normalised vectors, shape (n, dim) (or (dim,) for one string). Unknown text -> zero vector."""
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self._vector(str(text))[0]
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)
        return out[0] if single else out

    def is_known(self, text: str) -> bool:
        """True if at least one word of the text has a vector."""
        return self._vector(str(text))[1] > 0
