"""LexicalCombatEngine: domain gate, target relevance and damage (notebook Section 6.1, pandas-free).

The engine only calls ``model.encode(texts, ...)``; an optional ``model.is_known(text)`` enables the
unknown-word check. It is read-only after construction except for the per-target cache.
"""
import time

import numpy as np

from .config import EngineConfig, EngineData
from .text import clean_text, domain_term_pool, duplicate_key, is_generic_term, validate_player_input, word_keys

GATE_LABEL_OF_KIND = {"primary": "in_domain", "secondary": "ambiguous",
                      "weak_secondary": "ambiguous", "out_of_domain": "out_of_domain"}


def topk_mean(similarities, k: int):
    """Mean of the k largest values along the last axis."""
    sims = np.asarray(similarities, dtype=float)
    k = max(1, min(int(k), sims.shape[-1]))
    return np.sort(sims, axis=-1)[..., -k:].mean(axis=-1)


def normalized_relevance(similarity, noise, ceiling):
    """r = (cos - noise) / (ceiling - noise), clipped to [0, 1]."""
    return np.clip((np.asarray(similarity, dtype=float) - noise) / (ceiling - noise), 0.0, 1.0)


def gate_kind_arrays(required, lead, primary_min, secondary_min, secondary_gap, weak_gap) -> np.ndarray:
    """Vectorised gate. lead = required score - best other domain score."""
    required = np.asarray(required, dtype=float)
    lead = np.asarray(lead, dtype=float)
    is_primary = (required >= primary_min) & (lead >= 0)
    passes = required >= secondary_min
    is_secondary = ~is_primary & passes & (lead >= -secondary_gap)
    is_weak = ~is_primary & ~is_secondary & passes & (lead >= -weak_gap)
    return np.select([is_primary, is_secondary, is_weak],
                     ["primary", "secondary", "weak_secondary"], "out_of_domain")


def timed_call(function, *args, **kwargs):
    """Run a function; return (result, elapsed milliseconds)."""
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


class LexicalCombatEngine:
    """Owns the model, anchor embeddings, backgrounds, pools, target cache and span floor."""

    def __init__(self, model, data: EngineData, config: EngineConfig = EngineConfig()):
        self.data = data
        self._config = config
        self.refresh(model)

    # ---- configuration -------------------------------------------------
    @property
    def config(self) -> EngineConfig:
        return self._config

    @config.setter
    def config(self, new_config: EngineConfig) -> None:
        """Apply new parameters; the span floor depends on them, so recompute it."""
        self._config = new_config
        self.span_floor = self.resolve_span_floor()

    # ---- encoding and engine state --------------------------------------
    def encode(self, texts) -> np.ndarray:
        """L2-normalised sentence vectors."""
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)

    def refresh(self, model=None) -> None:
        """(Re)encode anchors, noise words and term pools with the given model."""
        if model is not None:
            self.model = model
        d = self.data
        texts, owners = [], []
        for domain, descriptions in d.anchors.items():
            for description in descriptions:
                texts.append(description)
                owners.append(domain)
        vectors = self.encode(texts)
        self.anchor_vectors = {dom: vectors[[i for i, o in enumerate(owners) if o == dom]] for dom in d.domains}
        known = getattr(self.model, "is_known", None)
        self.noise_terms = [w for w in d.noise_vocab if known is None or known(w)]
        self.dropped_terms = {"noise": [w for w in d.noise_vocab if w not in self.noise_terms]}
        self.noise_vectors = self.encode(self.noise_terms)
        self.background_by_k = {
            k: {dom: float(topk_mean(self.noise_vectors @ self.anchor_vectors[dom].T, k).mean())
                for dom in d.domains}
            for k in (1, 2, 3, 4, 5)
        }
        self.pools = {}
        for dom in d.domains:
            pool = domain_term_pool(dom, d.examples, d.pool_vocab, d.generic_terms)
            self.dropped_terms[dom] = [t for t in pool if known is not None and not known(t)]
            pool = [t for t in pool if known is None or known(t)]
            self.pools[dom] = (pool, self.encode(pool))
        self.targets = {}
        pair_sims = self.noise_vectors @ self.noise_vectors.T
        upper = pair_sims[np.triu_indices(len(self.noise_terms), k=1)]
        self.noise_pair_mean = float(upper.mean())     # anisotropy check
        self.noise_pair_p90 = float(np.percentile(upper, 90))
        self.span_floor = self.resolve_span_floor()

    @property
    def background(self) -> dict:
        return self.background_by_k[self.config.anchor_topk]

    # ---- domain gate ------------------------------------------------------
    def domain_scores(self, vectors, k: int = None):
        """Debiased (n, 4) and raw (n, 4) domain scores at top-k."""
        k = self.config.anchor_topk if k is None else int(k)
        vectors = np.atleast_2d(vectors)
        raw = np.column_stack([topk_mean(vectors @ self.anchor_vectors[dom].T, k) for dom in self.data.domains])
        background = np.array([self.background_by_k[k][dom] for dom in self.data.domains])
        return raw - background, raw

    def domain_scores_from_vector(self, vector, k: int = None):
        """(debiased scores, raw scores, display probabilities, nearest anchors) for one vector."""
        k = self.config.anchor_topk if k is None else int(k)
        raw, adjusted, nearest = {}, {}, {}
        for dom in self.data.domains:
            sims = self.anchor_vectors[dom] @ vector
            raw[dom] = float(topk_mean(sims, k))
            adjusted[dom] = raw[dom] - self.background_by_k[k][dom]
            nearest[dom] = self.data.anchors[dom][int(np.argmax(sims))]
        values = np.array([adjusted[dom] for dom in self.data.domains])
        weights = np.exp((values - values.max()) / self.config.display_softmax_temperature)
        probabilities = {dom: float(p) for dom, p in zip(self.data.domains, weights / weights.sum())}
        return adjusted, raw, probabilities, nearest

    def gate_multipliers(self) -> dict:
        c = self.config
        return {"primary": c.domain_primary_bonus, "secondary": c.domain_secondary_multiplier,
                "weak_secondary": c.domain_weak_secondary_multiplier, "out_of_domain": 0.0}

    def gate_kinds(self, required, lead) -> np.ndarray:
        """Gate kinds with the current thresholds."""
        c = self.config
        return gate_kind_arrays(required, lead, c.gate_primary_min, c.gate_secondary_min,
                                c.gate_secondary_max_gap, c.gate_weak_max_gap)

    def gate(self, domain_scores: dict, required_domain: str) -> tuple:
        """Soft gate on debiased scores. Returns (allowed, multiplier, kind)."""
        if required_domain not in domain_scores:
            raise ValueError(f"Unknown domain: {required_domain}")
        required = float(domain_scores[required_domain])
        lead = required - max(v for dom, v in domain_scores.items() if dom != required_domain)
        kind = str(self.gate_kinds([required], [lead])[0])
        return kind != "out_of_domain", self.gate_multipliers()[kind], kind

    # ---- target relevance ---------------------------------------------------
    def target_calibration(self, target: str, domain: str) -> dict:
        """Cached target vector plus its cosines to noise words and the domain pool."""
        key = (clean_text(target), domain)
        if key not in self.targets:
            target_clean = clean_text(target)
            vector = self.encode([target_clean])[0]
            pool, pool_vectors = self.pools[domain]
            target_keys = word_keys(target_clean)
            keep = [i for i, term in enumerate(pool) if not (word_keys(term) & target_keys)]
            self.targets[key] = {
                "vector": vector,
                "noise_sims": self.noise_vectors @ vector,
                "pool_sims": pool_vectors[keep] @ vector,
                "pool_terms": [pool[i] for i in keep],
            }
        return self.targets[key]

    def raw_bounds(self, cal: dict, noise_percentile=None, ceiling_topk=None) -> tuple:
        """(noise_floor, ceiling) before the span floor."""
        noise_percentile = self.config.noise_percentile if noise_percentile is None else noise_percentile
        ceiling_topk = self.config.ceiling_topk if ceiling_topk is None else ceiling_topk
        return (float(np.percentile(cal["noise_sims"], noise_percentile)),
                float(topk_mean(cal["pool_sims"], ceiling_topk)))

    def median_calibration_span(self, noise_percentile=None, ceiling_topk=None) -> float:
        """Median raw span (ceiling - noise) over the calibration targets."""
        spans = []
        for target, domain in self.data.calibration_targets.items():
            noise, ceiling = self.raw_bounds(self.target_calibration(target, domain), noise_percentile, ceiling_topk)
            spans.append(ceiling - noise)
        return float(np.median(spans))

    def resolve_span_floor(self) -> float:
        if self.config.min_span_mode == "median":
            return max(self.median_calibration_span(), 0.05)
        return self.config.min_calibration_span

    def bounds(self, cal: dict, noise_percentile=None, ceiling_topk=None, span_floor=None) -> tuple:
        """(noise_floor, ceiling) with ceiling >= noise + span floor."""
        noise, ceiling = self.raw_bounds(cal, noise_percentile, ceiling_topk)
        span_floor = self.span_floor if span_floor is None else span_floor
        return noise, max(ceiling, noise + span_floor)

    relevance = staticmethod(normalized_relevance)

    def graze_floor_mask(self, similarity, cal: dict, kind):
        """True where a primary-gate guess beats the median noise word (gets chip damage)."""
        if not self.config.primary_graze_floor:
            return np.zeros(np.shape(similarity), dtype=bool)
        floor = float(np.percentile(cal["noise_sims"], self.config.primary_graze_noise_pct))
        return (np.asarray(kind) == "primary") & (np.asarray(similarity, dtype=float) >= floor)

    # ---- damage ---------------------------------------------------------------
    def hit_percent(self, relevance, multiplier, allowed, generic_multiplier=1.0, echo_multiplier=1.0,
                    relevance_min=None, gamma=None, floor_mask=None):
        """Damage in percent of the boss's starting HP (single source of truth).

        r < r_min or r <= 0 -> 0; base = GRAZE + (MAX - GRAZE) * ((r - r_min) / (1 - r_min)) ** gamma;
        hit = base * domain * generic * echo; chip floor for allowed hits (and floor_mask rows); cap at MAX.
        """
        c = self.config
        relevance_min = c.relevance_min if relevance_min is None else relevance_min
        gamma = c.damage_gamma if gamma is None else gamma
        r = np.asarray(relevance, dtype=float)
        scaled = np.clip((r - relevance_min) / (1.0 - relevance_min), 0.0, 1.0)
        base = np.where((r < relevance_min) | (r <= 0.0), 0.0,
                        c.graze_min_percent + (c.max_hit_percent - c.graze_min_percent) * scaled ** gamma)
        allowed = np.asarray(allowed, dtype=bool)
        hit = base * np.asarray(multiplier, float) * np.asarray(generic_multiplier, float) \
            * np.asarray(echo_multiplier, float)
        chip_ok = allowed & (base > 0)
        if floor_mask is not None:
            chip_ok = chip_ok | (allowed & np.asarray(floor_mask, dtype=bool))
        hit = np.where(chip_ok, np.maximum(hit, c.chip_percent), 0.0)
        return np.minimum(hit, c.max_hit_percent)

    def base_damage(self, relevance) -> float:
        """Base damage percent for a relevance with domain multiplier 1."""
        return float(self.hit_percent(relevance, 1.0, True))

    def tier(self, damage_percent: float, allowed: bool) -> str:
        if not allowed:
            return "BLOCKED — OUT OF DOMAIN"
        if damage_percent <= 0:
            return "NO EFFECT"
        for cut, name in self.config.tier_cuts:
            if damage_percent >= cut:
                return name
        return "WEAK GRAZE"

    def is_generic(self, cleaned: str) -> bool:
        return is_generic_term(cleaned, self.data.generic_terms)

    def validate(self, guess: str, target: str, used) -> tuple:
        return validate_player_input(guess, target, used, self.config.block_target_word_in_guess)

    def unknown_reason(self, cleaned: str):
        """Message when the encoder has no vector for any word of the guess, else None."""
        known = getattr(self.model, "is_known", None)
        if known is not None and not known(cleaned):
            return (f"Unknown word: the model does not know '{cleaned}' — "
                    "this turn is not counted, try another word.")
        return None

    def echo_threshold(self) -> float:
        """Raw cosine above which a guess counts as a semantic repeat (model-relative)."""
        floor = self.noise_pair_p90
        return floor + self.config.echo_relative * (1.0 - floor)

    def echo_similarity(self, guess_vector, previous_vectors, cal: dict, required_domain: str) -> float:
        """Max cosine to earlier guesses that would have dealt damage."""
        best = 0.0
        noise, ceiling = self.bounds(cal)
        for vector in previous_vectors:
            adjusted, _, _, _ = self.domain_scores_from_vector(vector)
            allowed, _, _ = self.gate(adjusted, required_domain)
            relevance = float(normalized_relevance(vector @ cal["vector"], noise, ceiling))
            if allowed and relevance >= self.config.relevance_min:
                best = max(best, float(vector @ guess_vector))
        return best

    # ---- attacks, battles, hints ----------------------------------------------
    def evaluate_attack(self, guess: str, target: str, required_domain: str, used_guesses, enemy_max_hp: int) -> dict:
        """Validate, score and compute damage for one guess."""
        c = self.config
        is_valid, cleaned, reason = self.validate(guess, target, used_guesses)
        if not is_valid:
            return {"valid": False, "reason": reason}
        unknown = self.unknown_reason(cleaned)
        if unknown:
            return {"valid": False, "reason": unknown, "unknown": True, "word": cleaned}
        cal = self.target_calibration(target, required_domain)
        vectors, latency_ms = timed_call(self.encode, [cleaned] + sorted(used_guesses))
        guess_vector = vectors[0]

        similarity = float(guess_vector @ cal["vector"])
        noise, ceiling = self.bounds(cal)
        relevance = float(normalized_relevance(similarity, noise, ceiling))

        adjusted, raw, probabilities, nearest = self.domain_scores_from_vector(guess_vector)
        allowed, multiplier, match_type = self.gate(adjusted, required_domain)
        lead = adjusted[required_domain] - max(v for dom, v in adjusted.items() if dom != required_domain)

        generic = self.is_generic(cleaned)
        generic_multiplier = c.generic_penalty if generic else 1.0
        echo_similarity = self.echo_similarity(guess_vector, vectors[1:], cal, required_domain)
        echo = bool(echo_similarity >= self.echo_threshold())
        echo_multiplier = c.echo_penalty if echo else 1.0

        base_percent = self.base_damage(relevance) if allowed else 0.0
        floor_mask = bool(self.graze_floor_mask(similarity, cal, match_type))
        hit_percent = float(self.hit_percent(relevance, multiplier, allowed, generic_multiplier,
                                             echo_multiplier, floor_mask=floor_mask))
        damage = int(round(enemy_max_hp * hit_percent / 100))
        if hit_percent > 0:
            damage = max(damage, 1)
        return {
            "valid": True, "word": cleaned, "similarity": similarity,
            "noise_floor": noise, "ceiling": ceiling, "relevance": relevance,
            "domain_scores": adjusted, "domain_raw_scores": raw,
            "domain_background": dict(self.background), "domain_lead": lead,
            "graze_floor": floor_mask,
            "domain_probs": probabilities, "nearest_anchors": nearest,
            "domain_allowed": bool(allowed), "domain_match_type": match_type, "multiplier": multiplier,
            "is_generic": generic, "generic_multiplier": generic_multiplier,
            "echo_similarity": echo_similarity, "is_echo": echo, "echo_multiplier": echo_multiplier,
            "base_percent": base_percent, "damage_percent": round(hit_percent, 1),
            "hit_percent": hit_percent, "damage": damage,
            "tier": self.tier(hit_percent, allowed), "latency_ms": latency_ms,
        }

    @staticmethod
    def resolve_damage(result: dict, enemy_max_hp: int, current_hp: int) -> dict:
        """Add overkill bookkeeping (the per-hit cap is already applied)."""
        if not result["valid"]:
            return result
        calculated = result["damage"] if result["domain_allowed"] else 0
        actual = max(0, min(calculated, current_hp))
        result.update({"calculated_damage": calculated, "actual_damage": actual,
                       "overkill": calculated - actual, "battle_tier": result["tier"]})
        return result

    def simulate_battle(self, target: str, domain: str, initial_hp: int, guesses, on_turn=None) -> list:
        """Non-interactive battle; one dict per guess. on_turn(result, hp, domain) renders a turn."""
        hp, used, rows = initial_hp, set(), []
        for phrase in guesses:
            if hp <= 0:
                break
            result = self.evaluate_attack(phrase, target, domain, used, initial_hp)
            if not result["valid"]:
                rows.append({"Guess": clean_text(phrase), "Tier": "INVALID", "Note": result["reason"]})
                continue
            result = self.resolve_damage(result, initial_hp, hp)
            used.add(result["word"])
            hp_before, hp = hp, hp - result["actual_damage"]
            if on_turn is not None:
                on_turn(result, hp, domain)
            rows.append({
                "Guess": result["word"], "Cosine": round(result["similarity"], 3),
                "Relevance r": round(result["relevance"], 2), "Gate": result["domain_match_type"],
                "Generic": result["is_generic"], "Echo": result["is_echo"],
                "Hit %": round(result["hit_percent"], 1), "Tier": result["tier"],
                "Damage": result["actual_damage"], "HP before": hp_before, "HP left": max(hp, 0),
            })
        return rows

    @staticmethod
    def summarize_battle(rows: list, initial_hp: int) -> dict:
        """Turns, outcome and tier counts for one battle (rows from simulate_battle)."""
        valid = [r for r in rows if r["Tier"] != "INVALID"]
        tiers = [r["Tier"] for r in valid]
        defeated = bool(valid and valid[-1]["HP left"] <= 0)
        return {
            "Valid turns": len(valid), "Defeated": defeated,
            "Turns to defeat": len(valid) if defeated else None,
            "HP left": int(valid[-1]["HP left"]) if valid else initial_hp,
            "Critical": tiers.count("CRITICAL STRIKE"),
            "Strong": tiers.count("STRONG HIT"),
            "Solid": tiers.count("SOLID HIT"),
            "Graze": tiers.count("WEAK GRAZE"),
            "No effect / blocked": tiers.count("NO EFFECT") + tiers.count("BLOCKED — OUT OF DOMAIN"),
        }

    def hint(self, target: str, domain: str, used_guesses) -> str:
        """A useful but not optimal pool term (relevance 0.35-0.75)."""
        cal = self.target_calibration(target, domain)
        noise, ceiling = self.bounds(cal)
        relevance = normalized_relevance(cal["pool_sims"], noise, ceiling)
        used = {duplicate_key(u) for u in used_guesses}
        options = [(abs(r - 0.55), t) for t, r in zip(cal["pool_terms"], relevance)
                   if duplicate_key(t) not in used and 0.35 <= r <= 0.75]
        if not options:
            options = [(abs(r - 0.55), t) for t, r in zip(cal["pool_terms"], relevance)
                       if duplicate_key(t) not in used]
        return min(options)[1] if options else "No unused hint is available."
