"""URL routing for the combat app."""

from django.urls import path
from combat import views

app_name = "combat"

urlpatterns = [
    path("", views.arena_view, name="arena"),
    path("leaderboard/", views.leaderboard_page, name="leaderboard_page"),
    path("api/attack/", views.attack_endpoint, name="attack"),
    path("api/hint/", views.hint_endpoint, name="hint"),
    path("api/stages/", views.stages_endpoint, name="stages"),
    path("api/leaderboard/", views.leaderboard_endpoint, name="leaderboard"),
]
