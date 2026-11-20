"""Scoring engines for GloVe and SentenceTransformer with shared game data and config."""

import os
from functools import lru_cache
from pathlib import Path
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from lexical import LexicalCombatEngine, load_engine_config, load_game_data
from lexical.encoders.sentence_transformer import SentenceTransformerEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA = load_game_data(ROOT / "shared_data/game_data.json")
CONFIG = load_engine_config(ROOT / "shared_data/engine_config.json")

STAGES = [dict(s, hp=200) for s in DATA.game_stages]
if not any(s.get("target") == "painting" for s in STAGES):
    STAGES.append(dict(stage=4, enemy="Muse Phantom", target="painting", domain="Arts", hp=200))

STOPWORDS = {"a", "an", "the", "in", "on", "at", "by", "for", "with", "and", "or"}


class GloveEncoder:
    """Mean pooling GloVe encoder with L2 normalization."""

    def __init__(self, vector_path: Path = None):
        self.vectors = {}
        path = vector_path or (ROOT / "glove_module/glove.6B.100d.txt")
        if not Path(path).exists():
            alt = ROOT.parent / "branch_evaluation/glove.6B.100d.txt"
            if alt.exists():
                path = alt
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                self.vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
        self.dim = 100
        self.name = "glove"

    def is_known(self, text: str) -> bool:
        return any(w in self.vectors and w not in STOPWORDS for w in text.lower().split())

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        ts = [texts] if single else texts
        rows = []
        for text in ts:
            vs = [self.vectors[w] for w in str(text).lower().split() if w in self.vectors and w not in STOPWORDS]
            v = np.mean(vs, axis=0) if vs else np.zeros(self.dim, dtype=np.float32)
            norm = float(np.linalg.norm(v))
            rows.append(v / max(norm, 1e-12))
        res = np.asarray(rows, dtype=np.float32)
        return res[0] if single else res


@lru_cache(maxsize=4)
def get_engine(backend: str = "sbert") -> LexicalCombatEngine:
    backend = (backend or "sbert").lower().strip()
    if backend == "glove":
        encoder = GloveEncoder()
    elif backend in ("sbert", "sentence_transformer"):
        cache_folder = str(ROOT / "SBERT/pretrained_dev/artifacts/hf_cache")
        encoder = SentenceTransformerEncoder(cache_folder=cache_folder, torch_threads=2)
    elif backend == "self_trained":
        st_dir = ROOT / "SBERT/self_trained_dev/artifacts"
        if not st_dir.exists():
            st_dir = ROOT / "SBERT/pretrained_dev/artifacts"
        encoder = SentenceTransformerEncoder.from_dir(st_dir)
    else:
        raise ValueError(f"Unknown backend: {backend}. Choose glove or sbert.")
    return LexicalCombatEngine(encoder, DATA.engine_data, CONFIG)
