"""App configuration for combat; loads NLP models and ML artifacts once at startup."""

import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class CombatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "combat"

    # Shared in-memory reference for loaded ML artifacts
    lda_projector = None

    def ready(self):
        """Initialize NLP models and ML artifacts once when Django starts."""
        try:
            from combat.engine import initialize_engines
            initialize_engines()
            logger.info("Combat NLP models initialized successfully.")
        except Exception as exc:
            logger.warning("Failed initializing Combat NLP engines: %s", exc)
