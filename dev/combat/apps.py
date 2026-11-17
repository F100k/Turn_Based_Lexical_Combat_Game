"""App configuration for combat; loads ML artifacts once at startup."""

import logging
from django.apps import AppConfig
from django.conf import settings

logger = logging.getLogger(__name__)


class CombatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "combat"

    # Shared in-memory reference for loaded ML artifacts
    lda_projector = None

    def ready(self):
        """Load serialized ML models once when Django starts."""
        if CombatConfig.lda_projector is not None:
            return

        artifact_path = settings.LDA_PROJECTOR_PATH
        if not artifact_path.exists() or artifact_path.stat().st_size == 0:
            logger.warning(
                "ML artifact not found or empty at %s. Running without preloaded model.",
                artifact_path,
            )
            return

        try:
            import joblib

            CombatConfig.lda_projector = joblib.load(artifact_path)
            logger.info("Loaded LDA projector from %s", artifact_path)
        except Exception as exc:
            logger.warning("Failed to load LDA projector from %s: %s", artifact_path, exc)
