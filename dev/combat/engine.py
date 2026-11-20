"""Lexical Combat Engine: Unified multi-model NLP scoring engine.

Supports:
1. SBERT (Sentence-Transformers MiniLM) from SBERT/pretrained_dev
2. GloVe (Static Word Embeddings) from glove_module
3. Resilient Semantic Fallback (Zero-dependency N-gram & keyword scoring)
"""

import difflib
import importlib
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Repository root and artifact paths
ENGINE_DIR = Path(__file__).resolve().parent
DEV_DIR = ENGINE_DIR.parent
REPO_ROOT = DEV_DIR.parent

SBERT_DIR = REPO_ROOT / "SBERT" / "pretrained_dev"
SBERT_ARTIFACTS_DIR = SBERT_DIR / "artifacts"
GLOVE_DIR = REPO_ROOT / "glove_module"
GLOVE_TXT_PATH = GLOVE_DIR / "glove.6B.100d.txt"
CORPUS_DOCS_PATH = GLOVE_DIR / "corpus" / "domain_docs.csv"

# ── DOMAINS & ANCHORS ────────────────────────────────────────────────────────
DOMAINS = ["Business", "Technology", "Science", "Arts"]

DOMAIN_ANCHORS: Dict[str, List[str]] = {
    "Business": [
        "Business finance accounting financial reporting quarterly report annual report balance sheet income statement cash flow revenue profit budget audit forecast",
        "Business management leadership executive chief executive officer CEO president director manager supervisor board of directors chairman team organization",
        "Business operations strategy planning human resources payroll hiring project management sales marketing procurement supply chain logistics customer service",
    ],
    "Technology": [
        "Technology computing computer science software development programming source code debugging compiler API application database",
        "Algorithms and data structures array linked list stack queue tree graph hash table heap sorting searching recursion time complexity pseudocode",
        "Artificial intelligence AI machine learning deep learning neural network model training inference natural language processing computer vision",
    ],
    "Science": [
        "Science scientific research physics chemistry biology experiments hypotheses theories measurements natural phenomena",
        "Chemistry studies molecules atoms elements compounds chemical reactions chemical bonds acids bases and the periodic table",
        "Physics includes mechanics gravity electricity magnetism quantum mechanics particles electrons forces energy and motion",
    ],
    "Arts": [
        "Arts visual art painting sculpture drawing illustration design creativity aesthetics artist gallery museum",
        "Painting is a visual art form that uses pigment color composition canvas brushwork and artistic expression",
        "Music includes melody harmony rhythm songs composers orchestras symphonies opera concerts and instruments such as guitar and piano",
    ],
}

# Thematic bosses and stage progression (calibrated with SBERT game_data & GloVe pacing)
BOSS_STAGES = [
    {
        "stage": 1,
        "enemy": "Corporate Titan",
        "domain": "Business",
        "target": "executive",
        "hp": 180,
        "description": "Monolithic CEO guarding the corporate citadel.",
        "taunts": [
            "Your hostile takeover is denied!",
            "Profits plummet before my executive order!",
            "A sub-par quarterly performance! Face liquidation!",
            "My balance sheet has no room for your deficit!",
        ],
    },
    {
        "stage": 2,
        "enemy": "Cyber Sentinel",
        "domain": "Technology",
        "target": "algorithm",
        "hp": 200,
        "description": "Autonomous security core protecting the mainframe.",
        "taunts": [
            "Syntax error detected in your logic!",
            "Buffer overflow! Your firewalls are breached!",
            "Subroutine compromised. Initiating system purge!",
            "Infinite loop detected. Your process has been terminated!",
        ],
    },
    {
        "stage": 3,
        "enemy": "Alchemist Golem",
        "domain": "Science",
        "target": "catalyst",
        "hp": 200,
        "description": "Construct fueled by reactive primordial elements.",
        "taunts": [
            "An unstable reaction consumes you!",
            "My transmutation dissolves your words!",
            "Activation energy insufficient! Your hypothesis fails!",
            "The entropy of this battle accelerates!",
        ],
    },
    {
        "stage": 4,
        "enemy": "Muse Phantom",
        "domain": "Arts",
        "target": "painting",
        "hp": 200,
        "description": "Ethereal spirit weaving discord and tragic harmony.",
        "taunts": [
            "Dissonance shatters your prose!",
            "A masterpiece of tragedy unfolds before you!",
            "Your composition lacks rhythm and soul!",
            "The gallery rejected your amateur creation!",
        ],
    },
]

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "and", "or",
    "is", "it", "to", "of", "from", "as", "that", "this",
}

GENERIC_TERMS = {
    "business", "technology", "science", "arts", "art", "work", "thing",
    "make", "system", "process", "good", "word", "concept", "element",
}


def clean_text(text: str) -> str:
    """Normalize text: lowercase, letters and spaces only."""
    return re.sub(r"[^a-z\s]+", " ", str(text).lower()).strip()


# ── VOCABULARY & CORPUS LOADER ──────────────────────────────────────────────
_CORPUS_VOCAB: Optional[set] = None


def get_corpus_vocab() -> set:
    """Load vocabulary from corpus.txt and game_data.json for spell checking."""
    global _CORPUS_VOCAB
    if _CORPUS_VOCAB is not None:
        return _CORPUS_VOCAB

    vocab = set()
    # 1. From game_data.json
    game_data_path = SBERT_ARTIFACTS_DIR / "game_data.json"
    if game_data_path.is_file():
        try:
            with open(game_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for pool in data.get("pool_vocab", {}).values():
                    for phrase in pool:
                        vocab.update(clean_text(phrase).split())
                for examples in data.get("examples", {}).values():
                    for ex in examples:
                        vocab.update(clean_text(ex).split())
                vocab.update(data.get("noise_vocab", []))
        except Exception as e:
            logger.warning("Failed loading vocab from game_data.json: %s", e)

    # 2. From domain corpus docs (domain_docs.csv)
    if CORPUS_DOCS_PATH.is_file():
        try:
            import csv
            with open(CORPUS_DOCS_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for field in ("term", "title", "text"):
                        val = row.get(field, "")
                        vocab.update(clean_text(val).split())
        except Exception as e:
            logger.warning("Failed loading vocab from domain_docs.csv: %s", e)

    _CORPUS_VOCAB = vocab
    return _CORPUS_VOCAB


def smart_spell_check(text: str) -> Tuple[str, Optional[str]]:
    """Detect and correct slight typos using difflib against known vocabulary."""
    cleaned = clean_text(text)
    words = cleaned.split()
    if not words:
        return text, None

    vocab = get_corpus_vocab()
    if not vocab:
        return text, None

    corrected_words = []
    corrections = []

    for w in words:
        if w in vocab or len(w) <= 3:
            corrected_words.append(w)
        else:
            matches = difflib.get_close_matches(w, vocab, n=1, cutoff=0.78)
            if matches:
                match = matches[0]
                corrections.append(f"'{w}' -> '{match}'")
                corrected_words.append(match)
            else:
                corrected_words.append(w)

    corrected_text = " ".join(corrected_words)
    note = f"Typo auto-corrected: {', '.join(corrections)}" if corrections else None
    return corrected_text, note


# ── GLOVE EMBEDDINGS BACKEND ─────────────────────────────────────────────────
class GloVeBackend:
    """GloVe static word embedding engine with Mean Pooling and Soft Domain Gate."""

    def __init__(self, glove_path: Path = GLOVE_TXT_PATH):
        self.glove_path = glove_path
        self.embeddings: Optional[Dict[str, Any]] = None
        self.domain_vectors: Dict[str, Any] = {}
        self.is_loaded = False

    def load(self) -> bool:
        if self.is_loaded:
            return True
        if not self.glove_path.is_file() or self.glove_path.stat().st_size == 0:
            return False

        try:
            import numpy as np
            if str(GLOVE_DIR) not in sys.path:
                sys.path.insert(0, str(GLOVE_DIR))
            import glove_loader

            logger.info("Loading GloVe vectors from %s...", self.glove_path)
            self.embeddings = glove_loader.load_glove_model(str(self.glove_path))
            self.loader = glove_loader
            self.is_loaded = True

            # Precompute domain anchor vectors from glove_loader
            for dom in glove_loader.DOMAIN_ANCHORS.keys():
                d_vec = glove_loader.get_domain_vector(dom, self.embeddings)
                if d_vec is not None:
                    self.domain_vectors[dom] = d_vec

            logger.info("GloVe loaded successfully with %d words.", len(self.embeddings))
            return True
        except Exception as e:
            logger.warning("GloVe loading failed: %s", e)
            return False

    def get_phrase_vector(self, phrase: str):
        if not self.is_loaded or self.embeddings is None:
            return None
        if hasattr(self, "loader") and self.loader:
            return self.loader.get_phrase_vector(phrase, self.embeddings)
        import numpy as np
        words = [w for w in clean_text(phrase).split() if w not in STOPWORDS]
        vectors = [self.embeddings[w] for w in words if w in self.embeddings]
        if not vectors:
            return None
        phrase_vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(phrase_vec)
        if norm > 0:
            phrase_vec = phrase_vec / norm
        return phrase_vec

    def evaluate(
        self,
        phrase: str,
        target_word: str,
        required_domain: str,
        used_guesses: Optional[set] = None,
        enemy_max_hp: int = 100,
    ) -> Dict[str, Any]:
        import numpy as np

        phrase_vec = self.get_phrase_vector(phrase)
        target_vec = self.get_phrase_vector(target_word)

        if phrase_vec is None or target_vec is None:
            # Word was entirely OOV in GloVe
            return {
                "valid": False,
                "similarity": 0.0,
                "domain_allowed": False,
                "domain_match_type": "UNKNOWN / OOV",
                "multiplier": 0.0,
                "tier": "BLOCKED / MISS",
                "damage": 0,
                "model_used": "GloVe (Static Embeddings)",
            }

        # Use glove_loader calibrated gate & damage
        if hasattr(self, "loader") and self.loader:
            gate_res = self.loader.soft_domain_gate(phrase, required_domain, self.embeddings)
            dmg_res = self.loader.calculate_dynamic_damage(
                phrase, target_word, self.embeddings, max_hp=enemy_max_hp
            )
            domain_allowed = gate_res["allowed"]
            multiplier = gate_res["multiplier"]
            match_type = gate_res["status"]
            similarity = dmg_res["similarity"]
            tier = dmg_res["tier"] if domain_allowed else "BLOCKED / MISS"
            base_damage = dmg_res["damage"] if domain_allowed else 0
        else:
            similarity = float(np.dot(phrase_vec, target_vec))
            domain_allowed = True
            multiplier = 1.0
            match_type = "IN-DOMAIN"
            tier = "SOLID HIT"
            base_damage = 20

        actual_damage = int(round(base_damage * multiplier))
        if base_damage > 0 and domain_allowed:
            actual_damage = max(actual_damage, 1)

        domain_scores = {}
        for dom, dom_vec in self.domain_vectors.items():
            domain_scores[dom] = float(np.dot(phrase_vec, dom_vec))

        return {
            "valid": True,
            "similarity": round(similarity, 4),
            "relevance": round(max(0.0, similarity), 3),
            "domain_scores": {k: round(v, 3) for k, v in domain_scores.items()},
            "domain_allowed": domain_allowed,
            "domain_match_type": match_type,
            "multiplier": multiplier,
            "tier": tier,
            "damage": actual_damage,
            "model_used": "GloVe (Static Embeddings)",
        }

    def hint(self, target_word: str, required_domain: str, used_guesses: set) -> str:
        words = get_corpus_vocab(required_domain)
        cleaned_target = clean_text(target_word)
        for w in words:
            cw = clean_text(w)
            if cw and cw != cleaned_target and cw not in used_guesses and len(cw) > 3:
                return w
        if required_domain in DOMAIN_ANCHORS:
            for sentence in DOMAIN_ANCHORS[required_domain]:
                for w in sentence.split():
                    cw = clean_text(w)
                    if cw and cw != cleaned_target and cw not in used_guesses and cw not in STOPWORDS and len(cw) > 3:
                        return w
        return "Think of words closely associated with the target domain."


# ── SBERT BACKEND (Sentence-Transformers MiniLM) ──────────────────────────────
class SBERTBackend:
    """Pretrained Contextual Sentence Transformer engine from SBERT/pretrained_dev."""

    def __init__(self, artifacts_dir: Path = SBERT_ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.game = None
        self.is_loaded = False

    def load(self) -> bool:
        if self.is_loaded:
            return True
        if not self.artifacts_dir.is_dir():
            return False

        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            # Load the pretrained implementation in an isolated module namespace.
            load_game = _load_lexical_package(SBERT_DIR)

            logger.info("Loading SBERT game engine from %s...", self.artifacts_dir)
            self.game = load_game("pretrained", self.artifacts_dir)
            self.is_loaded = True
            logger.info("SBERT game engine loaded successfully.")
            return True
        except Exception as e:
            logger.info("SBERT game engine not loaded: %s", e)
            return False

    def evaluate(
        self,
        phrase: str,
        target_word: str,
        required_domain: str,
        used_guesses: set,
        enemy_max_hp: int = 100,
    ) -> Dict[str, Any]:
        if not self.is_loaded or self.game is None:
            raise RuntimeError("SBERT backend is not loaded.")

        r = self.game.engine.evaluate_attack(
            phrase, target_word, required_domain, used_guesses, enemy_max_hp
        )
        if not r.get("valid"):
            return {
                "valid": False,
                "reason": r.get("reason", "Invalid attack"),
                "similarity": 0.0,
                "domain_allowed": False,
                "domain_match_type": "INVALID",
                "multiplier": 0.0,
                "tier": "NO EFFECT",
                "damage": 0,
                "model_used": "SBERT (all-MiniLM-L6-v2)",
            }

        return {
            "valid": True,
            "similarity": round(r["similarity"], 4),
            "relevance": round(r["relevance"], 3),
            "domain_scores": {k: round(v, 3) for k, v in r["domain_scores"].items()},
            "domain_allowed": r["domain_allowed"],
            "domain_match_type": r["domain_match_type"],
            "multiplier": r["multiplier"],
            "is_generic": r["is_generic"],
            "is_echo": r["is_echo"],
            "tier": r["tier"],
            "damage": r["damage"],
            "hit_percent": r["hit_percent"],
            "model_used": "SBERT (all-MiniLM-L6-v2)",
        }

    def hint(self, target_word: str, required_domain: str, used_guesses: set) -> str:
        if not self.is_loaded or self.game is None:
            return ""
        try:
            return self.game.engine.hint(target_word, required_domain, used_guesses)
        except Exception:
            return ""


# ── RESILIENT SEMANTIC FALLBACK BACKEND ───────────────────────────────────────
class FallbackSemanticBackend:
    """Zero-dependency, pure-Python semantic similarity and domain gate engine.

    Guarantees the game runs out of the box without requiring PyTorch or
    huge 350MB GloVe vector files.
    """

    def __init__(self):
        self.domain_anchors_tokens = {
            dom: self._tokenize_and_ngrams(" ".join(sentences))
            for dom, sentences in DOMAIN_ANCHORS.items()
        }
        self.game_data = self._load_game_data()

    def _load_game_data(self) -> dict:
        path = SBERT_ARTIFACTS_DIR / "game_data.json"
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _tokenize_and_ngrams(self, text: str) -> Dict[str, float]:
        """Extract word tokens and character 3-grams with sublinear term frequency."""
        cleaned = clean_text(text)
        tokens: Dict[str, float] = {}

        # Word tokens
        words = [w for w in cleaned.split() if w not in STOPWORDS]
        for w in words:
            tokens[w] = tokens.get(w, 0.0) + 1.0

        # Character trigrams for morphological and spelling proximity
        text_padded = f" {cleaned} "
        for i in range(max(len(text_padded) - 2, 1)):
            ng = text_padded[i : i + 3]
            tokens[ng] = tokens.get(ng, 0.0) + 0.35

        return tokens

    def _cosine(self, v1: Dict[str, float], v2: Dict[str, float]) -> float:
        common = set(v1.keys()) & set(v2.keys())
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        norm1 = math.sqrt(sum(v * v for v in v1.values()))
        norm2 = math.sqrt(sum(v * v for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def evaluate(
        self,
        phrase: str,
        target_word: str,
        required_domain: str,
        used_guesses: set,
        enemy_max_hp: int = 100,
    ) -> Dict[str, Any]:
        cleaned_phrase = clean_text(phrase)
        cleaned_target = clean_text(target_word)

        if not cleaned_phrase:
            return {
                "valid": False,
                "similarity": 0.0,
                "domain_allowed": False,
                "domain_match_type": "EMPTY",
                "multiplier": 0.0,
                "tier": "NO EFFECT",
                "damage": 0,
                "model_used": "Semantic Lexical (Fallback)",
            }

        v_phrase = self._tokenize_and_ngrams(cleaned_phrase)
        v_target = self._tokenize_and_ngrams(cleaned_target)

        # 1. Domain Gating against anchor texts
        domain_scores = {}
        for dom, dom_tokens in self.domain_anchors_tokens.items():
            domain_scores[dom] = self._cosine(v_phrase, dom_tokens)

        # Enhance domain score if phrase matches domain term pools
        pool_vocab = self.game_data.get("pool_vocab", {}).get(required_domain, [])
        for term in pool_vocab:
            if clean_text(term) in cleaned_phrase or cleaned_phrase in clean_text(term):
                domain_scores[required_domain] = max(
                    domain_scores.get(required_domain, 0.0), 0.38
                )

        best_dom = max(domain_scores, key=domain_scores.get)
        best_score = domain_scores[best_dom]
        req_score = domain_scores.get(required_domain, 0.0)

        # Gate decision
        threshold = 0.16
        if req_score >= threshold and (best_dom == required_domain or (best_score - req_score) < 0.08):
            if best_dom == required_domain and req_score >= 0.22:
                domain_allowed = True
                multiplier = 1.25
                match_type = f"PRIMARY ({required_domain}) - Bonus!"
            else:
                domain_allowed = True
                multiplier = 1.0
                match_type = f"IN-DOMAIN ({required_domain})"
        else:
            if req_score >= 0.12:
                domain_allowed = True
                multiplier = 0.75
                match_type = f"SECONDARY (Matched {best_dom})"
            else:
                domain_allowed = False
                multiplier = 0.0
                match_type = "BLOCKED (Out of Domain)"

        # 2. Semantic Similarity to Target
        base_sim = self._cosine(v_phrase, v_target)

        # Check for related pool terms to give higher semantic credit
        if domain_allowed:
            for term in pool_vocab:
                c_term = clean_text(term)
                if c_term in cleaned_phrase or cleaned_phrase in c_term:
                    term_sim = self._cosine(v_target, self._tokenize_and_ngrams(c_term))
                    base_sim = max(base_sim, term_sim * 0.85, 0.35)

        similarity = float(base_sim)

        # 3. Dynamic damage cutoffs: (0.05, 0.15, 0.30, 0.50)
        if similarity < 0.05 or not domain_allowed:
            tier = "BLOCKED / MISS"
            base_damage = 0
        elif similarity < 0.15:
            tier = "WEAK GRAZE"
            base_damage = int(enemy_max_hp * 0.10)
        elif similarity < 0.30:
            tier = "SOLID HIT"
            base_damage = int(enemy_max_hp * 0.20)
        elif similarity < 0.50:
            tier = "STRONG HIT"
            base_damage = int(enemy_max_hp * 0.35)
        else:
            tier = "CRITICAL STRIKE!"
            base_damage = int(enemy_max_hp * 0.50)

        # Echo and generic penalties
        is_generic = cleaned_phrase in GENERIC_TERMS
        generic_mult = 0.6 if is_generic else 1.0

        is_echo = any(self._cosine(v_phrase, self._tokenize_and_ngrams(u)) > 0.75 for u in used_guesses)
        echo_mult = 0.5 if is_echo else 1.0

        actual_damage = int(round(base_damage * multiplier * generic_mult * echo_mult))
        if base_damage > 0 and domain_allowed:
            actual_damage = max(actual_damage, 1)

        return {
            "valid": True,
            "similarity": round(similarity, 4),
            "relevance": round(max(0.0, similarity), 3),
            "domain_scores": {k: round(v, 3) for k, v in domain_scores.items()},
            "domain_allowed": domain_allowed,
            "domain_match_type": match_type,
            "multiplier": multiplier,
            "is_generic": is_generic,
            "is_echo": is_echo,
            "tier": tier,
            "damage": actual_damage,
            "model_used": "Semantic Lexical (Zero-Dependency)",
        }

    def hint(self, target_word: str, required_domain: str, used_guesses: set) -> str:
        pool_vocab = self.game_data.get("pool_vocab", {}).get(required_domain, [])
        for term in pool_vocab:
            if clean_text(term) not in used_guesses and clean_text(term) != clean_text(target_word):
                return term
        return "Think of words closely associated with the target domain."


# ── UNIFIED ENGINE MANAGER & FACADE ──────────────────────────────────────────
_SBERT_BACKEND = SBERTBackend()
_GLOVE_BACKEND = GloVeBackend()
_FALLBACK_BACKEND = FallbackSemanticBackend()


def initialize_engines():
    """Attempt loading all available NLP backends on application startup."""
    _SBERT_BACKEND.load()
    _GLOVE_BACKEND.load()


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Return availability status and description for each supported model."""
    if not _SBERT_BACKEND.is_loaded:
        _SBERT_BACKEND.load()
    if not _GLOVE_BACKEND.is_loaded:
        _GLOVE_BACKEND.load()

    return {
        "auto": {
            "name": "Auto (Recommended)",
            "available": True,
            "description": "Selects the best available model (SBERT > GloVe > Semantic Fallback).",
        },
        "sbert": {
            "name": "SBERT (Contextual MiniLM)",
            "available": _SBERT_BACKEND.is_loaded,
            "description": "Dense sentence transformer embedding using all-MiniLM-L6-v2.",
        },
        "glove": {
            "name": "GloVe (Static Word Vectors)",
            "available": _GLOVE_BACKEND.is_loaded,
            "description": "Static word embeddings from glove.6B.100d with Mean Pooling.",
        },
        "fallback": {
            "name": "Semantic Fallback (TF-IDF/N-gram)",
            "available": True,
            "description": "High-speed zero-dependency lexical engine.",
        },
    }


def score_lexical_attack(
    attack_text: str,
    target_topic: str = "Business",
    target_word: str = "executive",
    model_name: str = "auto",
    turn_number: int = 1,
    used_guesses: Optional[List[str]] = None,
    enemy_max_hp: int = 180,
) -> Dict[str, Any]:
    """Score a player's attack phrase using the selected NLP model."""
    used = set(clean_text(u) for u in (used_guesses or []))

    # Smart typo check
    corrected_text, typo_note = smart_spell_check(attack_text)
    eval_text = corrected_text or attack_text

    # Model selection resolution with on-demand loading
    selected_model = model_name.lower().strip()
    if selected_model == "sbert":
        if not _SBERT_BACKEND.is_loaded:
            _SBERT_BACKEND.load()
        backend = _SBERT_BACKEND if _SBERT_BACKEND.is_loaded else _FALLBACK_BACKEND
    elif selected_model == "glove":
        if not _GLOVE_BACKEND.is_loaded:
            _GLOVE_BACKEND.load()
        backend = _GLOVE_BACKEND if _GLOVE_BACKEND.is_loaded else _FALLBACK_BACKEND
    elif selected_model == "fallback":
        backend = _FALLBACK_BACKEND
    else:  # auto
        if not _SBERT_BACKEND.is_loaded:
            _SBERT_BACKEND.load()
        if not _GLOVE_BACKEND.is_loaded:
            _GLOVE_BACKEND.load()
        if _SBERT_BACKEND.is_loaded:
            backend = _SBERT_BACKEND
        elif _GLOVE_BACKEND.is_loaded:
            backend = _GLOVE_BACKEND
        else:
            backend = _FALLBACK_BACKEND

    # Evaluate attack
    try:
        eval_result = backend.evaluate(
            phrase=eval_text,
            target_word=target_word,
            required_domain=target_topic,
            used_guesses=used,
            enemy_max_hp=enemy_max_hp,
        )
    except Exception as exc:
        logger.warning("Primary backend evaluation failed, using fallback: %s", exc)
        eval_result = _FALLBACK_BACKEND.evaluate(
            phrase=eval_text,
            target_word=target_word,
            required_domain=target_topic,
            used_guesses=used,
            enemy_max_hp=enemy_max_hp,
        )

    words = [w for w in eval_text.split() if w]
    unique_words = set(w.lower() for w in words)

    return {
        "damage": eval_result.get("damage", 0),
        "actual_damage": eval_result.get("damage", 0),
        "similarity": eval_result.get("similarity", 0.0),
        "relevance": eval_result.get("relevance", 0.0),
        "tier": eval_result.get("tier", "BLOCKED / MISS"),
        "domain_allowed": eval_result.get("domain_allowed", False),
        "domain_match_type": eval_result.get("domain_match_type", "UNKNOWN"),
        "multiplier": eval_result.get("multiplier", 1.0),
        "model_used": eval_result.get("model_used", "Semantic Engine"),
        "reason": eval_result.get("reason"),
        "target_topic": target_topic,
        "target_word": target_word,
        "word": eval_text,
        "raw_input": attack_text,
        "word_count": len(words),
        "unique_word_count": len(unique_words),
        "topic_bonus": 5 if eval_result.get("multiplier", 1.0) > 1.0 else 0,
        "is_generic": eval_result.get("is_generic", False),
        "is_echo": eval_result.get("is_echo", False),
        "typo_note": typo_note,
        "projection": {
            "topic_id": 0,
            "topic_weights": [1.0],
            "confidence": eval_result.get("similarity", 0.5),
        },
    }


def generate_enemy_counterattack(
    turn_number: int = 1,
    enemy_name: str = "Corporate Titan",
    domain: str = "Business",
) -> Dict[str, Any]:
    """Generate enemy counterattack with thematic boss taunts."""
    boss_info = next((b for b in BOSS_STAGES if b["enemy"] == enemy_name), None)
    if not boss_info:
        boss_info = next((b for b in BOSS_STAGES if b["domain"] == domain), BOSS_STAGES[0])

    taunts = boss_info.get("taunts", ["Your vocabulary falters before my lexicon!"])
    taunt = taunts[(turn_number - 1) % len(taunts)]
    damage = 8 + (turn_number % 5) * 2

    return {
        "enemy_name": boss_info["enemy"],
        "enemy_text": taunt,
        "damage": damage,
    }


def generate_hint(
    target_word: str,
    domain: str = "Business",
    used_guesses: Optional[List[str]] = None,
    model_name: str = "auto",
) -> Dict[str, str]:
    """Provide a semantic hint related to the target and domain."""
    used = set(clean_text(u) for u in (used_guesses or []))

    hint_text = ""
    if model_name in ("auto", "sbert") and _SBERT_BACKEND.is_loaded:
        hint_text = _SBERT_BACKEND.hint(target_word, domain, used)
    elif model_name == "glove" and _GLOVE_BACKEND.is_loaded:
        hint_text = _GLOVE_BACKEND.hint(target_word, domain, used)

    if not hint_text:
        hint_text = _FALLBACK_BACKEND.hint(target_word, domain, used)

    return {
        "hint": hint_text,
        "target_word": target_word,
        "domain": domain,
    }
