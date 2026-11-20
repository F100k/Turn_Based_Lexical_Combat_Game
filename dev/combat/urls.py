"""URL routing for the combat app."""

from django.urls import path
from combat import views

app_name = "combat"

urlpatterns = [
    path("", views.arena_view, name="arena"),
    path("leaderboard/", views.leaderboard_page, name="leaderboard_page"),
    path("api/start/", views.start_endpoint, name="start"),
    path("api/start-battle/", views.start_endpoint, name="start_battle"),
    path("api/attack/", views.attack_endpoint, name="attack"),
    path("api/export/", views.export_endpoint, name="export"),
    path("api/leaderboard/", views.leaderboard_endpoint, name="leaderboard"),
]
