"""Database models for Leaderboard and MatchHistory."""

from django.db import models


class MatchHistory(models.Model):
    """Stores completed or in-progress lexical combat matches."""

    STATUS_CHOICES = [
        ("active", "Active"),
        ("won", "Won"),
        ("lost", "Lost"),
    ]

    player_name = models.CharField(max_length=100, default="Challenger")
    target_topic = models.CharField(max_length=120, blank=True, default="General")
    turns_taken = models.PositiveIntegerField(default=0)
    total_damage_dealt = models.PositiveIntegerField(default=0)
    score = models.IntegerField(default=0)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="active")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-score", "-created_at"]
        verbose_name_plural = "Match histories"

    def __str__(self) -> str:
        return f"{self.player_name} ({self.status}) - Score: {self.score}"
