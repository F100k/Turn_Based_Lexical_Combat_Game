"""App configuration for combat; loads NLP models and ML artifacts once at startup."""

import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class CombatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "combat"

    def ready(self):
        """Pre-warm NLP models once when Django starts."""
        try:
            from combat.scoring import get_engine
            get_engine("sbert")
            get_engine("glove")
            logger.info("Combat NLP models initialized successfully.")
        except Exception as exc:
            logger.warning("Failed initializing Combat NLP engines: %s", exc)
