"""Target bank screening and random stage drawing (notebook Section 6.5c, pandas-free)."""
import random

import numpy as np

from .engine import LexicalCombatEngine, normalized_relevance

RANDOM_ROUNDS = 4
RANDOM_BOSS_HP = 200
MIN_SPAN_RATIO = 0.6      # raw span must be >= this x span floor
GOOD_R = 0.6
MIN_GOOD_ANSWERS = 4
FALLBACK_TARGETS = 5      # used only if a topic has no eligible target


def pool_gate_kinds(engine: LexicalCombatEngine, domain: str) -> dict:
    """Gate kind of every pool term when played in its own domain."""
    pool, vectors = engine.pools[domain]
    adjusted, _ = engine.domain_scores(vectors, engine.config.anchor_topk)
    col = list(engine.data.domains).index(domain)
    required = adjusted[:, col]
    lead = required - np.delete(adjusted, col, axis=1).max(axis=1)
    return dict(zip(pool, engine.gate_kinds(required, lead)))


def build_target_bank(engine: LexicalCombatEngine, extra_targets: dict) -> list:
    """Screen candidate targets per topic; one dict per candidate with the reason for the decision."""
    rows = []
    known = getattr(engine.model, "is_known", None)
    for domain in engine.data.domains:
        kinds = pool_gate_kinds(engine, domain)
        candidates = list(dict.fromkeys(list(extra_targets.get(domain, [])) + engine.pools[domain][0]))
        vectors = engine.encode(candidates)
        for target, vector in zip(candidates, vectors):
            adjusted, _, _, _ = engine.domain_scores_from_vector(vector)
            _, _, self_kind = engine.gate(adjusted, domain)
            cal = engine.target_calibration(target, domain)
            noise, ceiling_raw = engine.raw_bounds(cal)
            noise, ceiling = engine.bounds(cal)
            r = normalized_relevance(cal["pool_sims"], noise, ceiling)
            good = sum(1 for term, rel in zip(cal["pool_terms"], r)
                       if rel >= GOOD_R and kinds.get(term, "out_of_domain") != "out_of_domain")
            span = ceiling_raw - noise
            if known is not None and not known(target):
                reason = "target unknown to the model"
            elif len(target.split()) > 2 or engine.is_generic(target):
                reason = "generic / too long"
            elif self_kind != "primary":
                reason = f"target itself is {self_kind}"
            elif span < MIN_SPAN_RATIO * engine.span_floor:
                reason = "span too narrow"
            elif good < MIN_GOOD_ANSWERS:
                reason = f"only {good} good answers"
            else:
                reason = "ok"
            rows.append({"domain": domain, "target": target, "self_gate": self_kind,
                         "span": round(span, 3), "good_answers": good, "eligible": reason == "ok",
                         "reason": reason})
    return rows


def eligible_targets(bank: list, domains) -> dict:
    """Eligible targets per domain; falls back to the best-ranked candidates when none pass."""
    out = {}
    for d in domains:
        rows = [r for r in bank if r["domain"] == d]
        ok = [r["target"] for r in rows if r["eligible"]]
        if not ok:
            ranked = sorted(rows, key=lambda r: (r["self_gate"] == "primary", r["good_answers"], r["span"]),
                            reverse=True)
            ok = [r["target"] for r in ranked[:FALLBACK_TARGETS]]
        out[d] = ok
    return out


def random_stages(eligible: dict, boss_names: dict, rounds: int, seed: int, hp: int = RANDOM_BOSS_HP) -> list:
    """Random topic order (all topics before repeating), random eligible target per topic."""
    rng = random.Random(seed)
    topics = [d for d in eligible if eligible[d]]
    stages, order, used = [], [], set()
    for number in range(1, rounds + 1):
        if not order:
            order = rng.sample(topics, len(topics))
        domain = order.pop()
        choices = [t for t in eligible[domain] if t not in used] or eligible[domain]
        target = rng.choice(choices)
        used.add(target)
        stages.append({"stage": number, "enemy": rng.choice(boss_names[domain]), "target": target,
                       "domain": domain, "hp": hp})
    return stages
