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
        self.assertContains(response, "Corporate Titan")

    def test_attack_endpoint_success(self):
        payload = {
            "attack_text": "board of directors",
            "target_topic": "Business",
            "target_word": "executive",
            "enemy_name": "Corporate Titan",
            "model_name": "auto",
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
        self.assertTrue(data["attack_result"]["domain_allowed"])
        self.assertGreater(data["attack_result"]["damage"], 0)

    def test_attack_out_of_domain_blocked(self):
        payload = {
            "attack_text": "photosynthesis chloroplast chloroplast",
            "target_topic": "Business",
            "target_word": "executive",
            "enemy_name": "Corporate Titan",
            "turn_number": 1,
            "player_hp": 100,
            "enemy_hp": 100,
        }
        response = self.client.post(
            reverse("combat:attack"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["attack_result"]["domain_allowed"])
        self.assertEqual(data["attack_result"]["damage"], 0)
        self.assertIn("BLOCKED", data["attack_result"]["tier"])

    def test_attack_endpoint_empty_text_error(self):
        response = self.client.post(
            reverse("combat:attack"),
            data=json.dumps({"attack_text": "   "}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_hint_endpoint(self):
        response = self.client.post(
            reverse("combat:hint"),
            data=json.dumps({"target_word": "executive", "target_topic": "Business"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("hint", data)
        self.assertTrue(len(data["hint"]) > 0)

    def test_stages_endpoint(self):
        response = self.client.get(reverse("combat:stages"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("stages", data)
        self.assertEqual(len(data["stages"]), 4)
        self.assertEqual(data["stages"][0]["enemy"], "Corporate Titan")

    def test_leaderboard_endpoint(self):
        MatchHistory.objects.create(
            player_name="Champion",
            target_topic="Business (executive) [SBERT]",
            turns_taken=3,
            total_damage_dealt=100,
            score=500,
            status="won",
        )
        response = self.client.get(reverse("combat:leaderboard"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("leaderboard", data)
        self.assertEqual(len(data["leaderboard"]), 1)
        self.assertEqual(data["leaderboard"][0]["player_name"], "Champion")

    def test_attack_endpoint_self_trained_backend(self):
        payload = {
            "attack_text": "board of directors",
            "target_topic": "Business",
            "target_word": "executive",
            "enemy_name": "Corporate Titan",
            "model_name": "self_trained",
            "turn_number": 1,
            "player_hp": 100,
            "enemy_hp": 100,
        }
        response = self.client.post(
            reverse("combat:attack"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("Self-Trained", data["attack_result"]["model_used"])
        self.assertTrue(data["attack_result"]["domain_allowed"])
        self.assertGreater(data["attack_result"]["damage"], 0)

    def test_attack_endpoint_glove_backend(self):
        payload = {
            "attack_text": "director of operations",
            "target_topic": "Business",
            "target_word": "executive",
            "enemy_name": "Corporate Titan",
            "model_name": "glove",
            "turn_number": 1,
            "player_hp": 100,
            "enemy_hp": 100,
        }
        response = self.client.post(
            reverse("combat:attack"),
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("GloVe", data["attack_result"]["model_used"])
        self.assertTrue(data["attack_result"]["domain_allowed"])
        self.assertGreater(data["attack_result"]["damage"], 0)

