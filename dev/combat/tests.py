import json
from django.test import Client, TestCase
from django.urls import reverse
from combat.models import MatchHistory


class CombatAppTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_arena_view_loads(self):
        response = self.client.get(reverse("combat:arena"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Turn-Based Lexical Combat")

    def test_attack_endpoint_success(self):
        payload = {
            "attack_text": "quantum entanglement experiment",
            "target_topic": "Science",
            "turn_number": 1,
            "player_hp": 100,
            "enemy_hp": 100,
            "player_name": "Hero",
        }
        response = self.client.post(
            reverse("combat:attack"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("attack_result", data)
        self.assertIn("enemy_action", data)
        self.assertIn("player_hp", data)
        self.assertIn("enemy_hp", data)
        self.assertIn("match_status", data)

    def test_attack_endpoint_empty_text_error(self):
        response = self.client.post(
            reverse("combat:attack"),
            data=json.dumps({"attack_text": "   "}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_leaderboard_endpoint(self):
        MatchHistory.objects.create(
            player_name="Champion",
            target_topic="Science",
            turns_taken=3,
            total_damage_dealt=50,
            score=500,
            status="won",
        )
        response = self.client.get(reverse("combat:leaderboard"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("leaderboard", data)
        self.assertEqual(len(data["leaderboard"]), 1)
        self.assertEqual(data["leaderboard"][0]["player_name"], "Champion")
