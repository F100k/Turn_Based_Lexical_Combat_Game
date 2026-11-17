"""Views for rendering the game arena UI and handling JSON combat endpoints."""

import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from combat.engine import generate_enemy_counterattack, score_lexical_attack
from combat.models import MatchHistory


@require_GET
def arena_view(request):
    """Render the main turn-based lexical combat arena HTML page."""
    return render(request, "combat/arena.html", {"default_topic": "Science"})


@require_GET
def leaderboard_page(request):
    """Render the leaderboard HTML page."""
    entries = MatchHistory.objects.exclude(status="active")[:20]
    return render(request, "combat/leaderboard.html", {"leaderboard": entries})


@require_POST
def attack_endpoint(request):
    """Handle a player's lexical attack submission and return JSON results."""
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload."}, status=400)

    attack_text = str(payload.get("attack_text", "")).strip()
    target_topic = str(payload.get("target_topic", "")).strip()
    turn_number = int(payload.get("turn_number", 1))
    player_hp = int(payload.get("player_hp", 100))
    enemy_hp = int(payload.get("enemy_hp", 100))
    player_name = str(payload.get("player_name", "Challenger")).strip() or "Challenger"

    if not attack_text:
        return JsonResponse({"error": "Attack text cannot be empty."}, status=400)

    attack_result = score_lexical_attack(attack_text, target_topic=target_topic)
    new_enemy_hp = max(0, enemy_hp - attack_result["damage"])

    enemy_action = None
    new_player_hp = player_hp
    match_status = "active"

    if new_enemy_hp == 0:
        match_status = "won"
    else:
        enemy_action = generate_enemy_counterattack(turn_number=turn_number)
        new_player_hp = max(0, player_hp - enemy_action["damage"])
        if new_player_hp == 0:
            match_status = "lost"

    # Record finished matches to MatchHistory
    if match_status in ("won", "lost"):
        MatchHistory.objects.create(
            player_name=player_name,
            target_topic=target_topic or "General",
            turns_taken=turn_number,
            total_damage_dealt=attack_result["damage"],
            score=new_player_hp * 10 + (100 if match_status == "won" else 0),
            status=match_status,
        )

    return JsonResponse(
        {
            "attack_result": attack_result,
            "enemy_action": enemy_action,
            "player_hp": new_player_hp,
            "enemy_hp": new_enemy_hp,
            "match_status": match_status,
        }
    )


@require_GET
def leaderboard_endpoint(request):
    """Return top recorded matches as JSON."""
    entries = list(
        MatchHistory.objects.exclude(status="active").values(
            "player_name",
            "target_topic",
            "turns_taken",
            "score",
            "status",
            "created_at",
        )[:10]
    )
    return JsonResponse({"leaderboard": entries})
