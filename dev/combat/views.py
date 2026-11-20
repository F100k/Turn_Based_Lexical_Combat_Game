"""Views for rendering the game arena UI and handling JSON combat endpoints."""

import csv
import io
import json
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from combat.models import MatchHistory
from combat.scoring import get_engine, STAGES
from lexical.session import public_result


def _parse_payload(request) -> dict:
    try:
        data = json.loads(request.body.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError()
        return data
    except (ValueError, UnicodeDecodeError):
        raise ValueError("Expected JSON object")


@require_GET
def arena_view(request):
    return render(
        request,
        "combat/arena.html",
        {
            "default_topic": "Technology",
            "default_target": "algorithm",
            "stages": STAGES,
        },
    )


@require_POST
def start_endpoint(request):
    try:
        data = _parse_payload(request)
        backend = str(data.get("backend", "sbert")).lower().strip()
        stage_id = data.get("stage", 1)
        if isinstance(stage_id, str):
            stage_id = int(stage_id) if stage_id.isdigit() else 1
        name = str(data.get("player_name", "Challenger")).strip() or "Challenger"

        if backend not in {"glove", "sbert", "self_trained"}:
            backend = "sbert"
        if not (0 <= stage_id < len(STAGES)):
            stage_id = 0
        if len(name) > 100:
            raise ValueError("Name must be at most 100 characters")

        get_engine(backend)
        state = {
            "backend": backend,
            "stage": stage_id,
            "name": name,
            "hp": 200,
            "player_hp": 100,
            "turns": 0,
            "used": [],
            "total_damage": 0,
            "status": "active",
        }
        request.session["battle"] = state
        return JsonResponse({"state": state, "stage": STAGES[stage_id]})
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except (OSError, ImportError) as e:
        return JsonResponse({"error": f"Model unavailable: {e}"}, status=503)


@require_POST
def attack_endpoint(request):
    try:
        data = _parse_payload(request)
        text = str(data.get("attack_text") or data.get("text") or "").strip()
        if not text:
            return JsonResponse({"error": "Attack text cannot be empty."}, status=400)
        if len(text) > 200:
            raise ValueError("Answer must be at most 200 characters")

        state = request.session.get("battle")
        if not state or state.get("status") != "active":
            backend = str(data.get("backend", "sbert")).lower().strip()
            stage_id = int(data.get("stage", 1))
            if not (0 <= stage_id < len(STAGES)):
                stage_id = 1
            state = {
                "backend": backend,
                "stage": stage_id,
                "name": str(data.get("player_name", "Challenger")).strip() or "Challenger",
                "hp": 200,
                "player_hp": 100,
                "turns": 0,
                "used": [],
                "total_damage": 0,
                "status": "active",
            }
            request.session["battle"] = state

        stage = STAGES[state["stage"]]
        engine = get_engine(state["backend"])
        r = engine.evaluate_attack(text, stage["target"], stage["domain"], set(state["used"]), 200)

        if not r["valid"]:
            return JsonResponse({"error": r.get("reason", "Invalid attack word.")}, status=400)

        r = engine.resolve_damage(r, 200, state["hp"])
        state["hp"] -= r["actual_damage"]
        state["total_damage"] += r["actual_damage"]
        state["turns"] += 1
        state["used"].append(r["word"])

        counter = 0 if state["hp"] == 0 else min(state["player_hp"], 8 + (state["turns"] % 5) * 2)
        state["player_hp"] -= counter
        state["status"] = "won" if state["hp"] == 0 else ("lost" if state["player_hp"] == 0 else "active")

        logs = request.session.get("trial_logs", [])
        logs.append(
            {
                "backend": state["backend"],
                "stage": state["stage"] + 1,
                "domain": stage["domain"],
                "target": stage["target"],
                "answer": r["word"],
                "turn": state["turns"],
                "similarity": r["similarity"],
                "relevance": r["relevance"],
                "gate": r["domain_match_type"],
                "domain_scores": json.dumps(r["domain_scores"]),
                "damage": r["actual_damage"],
                "latency_ms": r["latency_ms"],
            }
        )
        request.session["trial_logs"] = logs[-1000:]
        request.session["battle"] = state

        if state["status"] != "active":
            MatchHistory.objects.create(
                player_name=state["name"],
                target_topic=f"{state['backend']} | {stage['domain']} | {stage['target']}",
                turns_taken=state["turns"],
                total_damage_dealt=state["total_damage"],
                score=state["player_hp"] * 10 + (100 if state["status"] == "won" else 0),
                status=state["status"],
            )

        return JsonResponse(
            {
                "attack_result": public_result(r),
                "player_hp": state["player_hp"],
                "enemy_hp": state["hp"],
                "turn_number": state["turns"],
                "match_status": state["status"],
                "enemy_action": {"damage": counter, "enemy_text": "Counterattack"} if counter else None,
            }
        )
    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)


@require_GET
def leaderboard_page(request):
    entries = MatchHistory.objects.exclude(status="active")[:20]
    return render(request, "combat/leaderboard.html", {"leaderboard": entries})


@require_GET
def leaderboard_endpoint(request):
    entries = list(MatchHistory.objects.exclude(status="active").values()[:20])
    return JsonResponse({"leaderboard": entries})


@require_GET
def export_endpoint(request):
    rows = request.session.get("trial_logs", [])
    out = io.StringIO(newline="")
    if rows:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    response = HttpResponse(out.getvalue(), content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = 'attachment; filename="trial_logs.csv"'
    return response


# Aliases for backward compatibility
start_battle_endpoint = start_endpoint
