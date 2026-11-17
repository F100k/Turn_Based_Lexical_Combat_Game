"""Modular scoring & topic functions migrated from the experiment notebook."""

from typing import Any


def get_lda_projector():
    """Return the preloaded LDA projector artifact from CombatConfig."""
    from combat.apps import CombatConfig

    return CombatConfig.lda_projector


def project_topic(text: str, projector: Any = None) -> dict[str, Any]:
    """Project input text onto topic space using the loaded LDA projector.

    Replace the stub logic below with your notebook's preprocessing and
    projection pipeline.
    """
    if projector is None:
        projector = get_lda_projector()

    cleaned = text.strip()
    if not cleaned:
        return {"topic_id": None, "topic_weights": [], "confidence": 0.0}

    # TODO: Integrate real vectorizer/LDA projection using `projector`
    return {
        "topic_id": 0,
        "topic_weights": [1.0],
        "confidence": 0.5,
    }


def score_lexical_attack(
    attack_text: str,
    target_topic: str = "",
    projector: Any = None,
) -> dict[str, Any]:
    """Evaluate a player's lexical attack and return damage & breakdown.

    Replace the placeholder heuristic with your notebook's scoring functions.
    """
    words = [w for w in attack_text.strip().split() if w]
    unique_words = set(w.lower() for w in words)

    projection = project_topic(attack_text, projector=projector)

    # Placeholder scoring components
    base_damage = min(len(unique_words) * 4, 30)
    topic_bonus = 5 if target_topic and target_topic.lower() in attack_text.lower() else 0
    total_damage = base_damage + topic_bonus

    return {
        "damage": total_damage,
        "word_count": len(words),
        "unique_word_count": len(unique_words),
        "topic_bonus": topic_bonus,
        "projection": projection,
    }


def generate_enemy_counterattack(turn_number: int = 1) -> dict[str, Any]:
    """Return an enemy counterattack payload for the current turn."""
    # TODO: Replace with topic-driven or AI-generated counterattack logic
    damage = 8 + (turn_number % 5) * 2
    return {
        "enemy_text": "Your vocabulary falters before my lexicon!",
        "damage": damage,
    }
