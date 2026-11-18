"""Engine, text rules, config loading and game session (no model files needed)."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from tests.fakes import FakeEncoder
from lexical import EngineConfig, GameSession, LexicalCombatEngine, load_engine_config, load_game_data
from lexical.target_bank import build_target_bank, eligible_targets
from lexical.text import clean_text, validate_player_input

GAME = load_game_data()
ENGINE = LexicalCombatEngine(FakeEncoder(), GAME.engine_data, EngineConfig())


class TextRules(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("  Data-Base!! "), "data base")

    def test_validation(self):
        ok, cleaned, _ = validate_player_input("Chief Executive", "algorithm", set())
        self.assertTrue(ok)
        self.assertEqual(cleaned, "chief executive")
        self.assertFalse(validate_player_input("", "algorithm", set())[0])
        self.assertFalse(validate_player_input("one two three four", "algorithm", set())[0])
        self.assertFalse(validate_player_input("algorithms", "algorithm", set())[0])        # target itself
        self.assertFalse(validate_player_input("sorting algorithm", "algorithm", set())[0])  # contains target
        self.assertFalse(validate_player_input("sortings", "algorithm", {"sorting"})[0])     # near duplicate


class EngineBehaviour(unittest.TestCase):
    def test_attack_result(self):
        r = ENGINE.evaluate_attack("company director", "executive", "Business", set(), 200)
        self.assertTrue(r["valid"])
        self.assertIn(r["domain_match_type"], {"primary", "secondary", "weak_secondary", "out_of_domain"})
        self.assertGreaterEqual(r["damage"], 0)
        json.dumps({k: v for k, v in r.items() if k != "nearest_anchors"})   # JSON-safe values

    def test_unknown_word_not_scored(self):
        r = ENGINE.evaluate_attack("qwzxv", "executive", "Business", set(), 200)
        self.assertFalse(r["valid"])
        self.assertTrue(r["unknown"])

    def test_blocked_has_zero_damage(self):
        for guess in ["banana", "sofa", "pizza", "umbrella", "kite"]:
            r = ENGINE.resolve_damage(ENGINE.evaluate_attack(guess, "executive", "Business", set(), 200), 200, 200)
            if r["valid"] and not r["domain_allowed"]:
                self.assertEqual(r["actual_damage"], 0)

    def test_hint_is_unused_pool_term(self):
        hint = ENGINE.hint("executive", "Business", {"ceo"})
        self.assertIn(hint, ENGINE.pools["Business"][0])
        self.assertNotEqual(hint, "ceo")


class ConfigLoading(unittest.TestCase):
    def test_round_trip(self):
        config = EngineConfig(gate_primary_min=0.123, tier_cuts=((40, "CRITICAL STRIKE"), (20, "SOLID HIT")))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine_config.json"
            path.write_text(json.dumps(asdict(config)))
            loaded = load_engine_config(path)
        self.assertEqual(loaded.gate_primary_min, 0.123)
        self.assertEqual(loaded.tier_cuts, ((40.0, "CRITICAL STRIKE"), (20.0, "SOLID HIT")))

    def test_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine_config.json"
            path.write_text(json.dumps({"not_a_field": 1}))
            with self.assertRaises(ValueError):
                load_engine_config(path)


class Session(unittest.TestCase):
    def setUp(self):
        bank = build_target_bank(ENGINE, GAME.extra_targets)
        self.eligible = eligible_targets(bank, GAME.engine_data.domains)

    def test_round_trip_and_full_game(self):
        session = GameSession.start(self.eligible, GAME.boss_names, rounds=2, seed=3)
        guesses = [t for d in GAME.engine_data.domains for t in ENGINE.pools[d][0]] * 3
        for guess in guesses:
            session = GameSession.from_dict(json.loads(json.dumps(session.to_dict())))
            event = session.attack(ENGINE, guess)
            json.dumps(event)
            if event["game_over"]:
                break
        if not session.finished:
            session.end_game()
        self.assertTrue(session.finished)
        self.assertLessEqual(len(session.stage_logs), 2)
        self.assertTrue(all(log["status"] in {"Defeated", "Quit"} for log in session.stage_logs))

    def test_invalid_guess_not_counted(self):
        session = GameSession.start(self.eligible, GAME.boss_names, rounds=1, seed=1)
        event = session.attack(ENGINE, "one two three four")
        self.assertFalse(event["counted"])
        self.assertEqual(session.turns, 0)

    def test_skip_and_quit(self):
        session = GameSession.start(self.eligible, GAME.boss_names, rounds=2, seed=1)
        self.assertEqual(session.skip()["stage_ended"]["status"], "Skipped")
        summary = session.end_game()
        self.assertTrue(session.finished)
        self.assertEqual(summary["stage_logs"][-1]["status"], "Quit")

    def test_same_seed_same_stages(self):
        a = GameSession.start(self.eligible, GAME.boss_names, rounds=4, seed=11)
        b = GameSession.start(self.eligible, GAME.boss_names, rounds=4, seed=11)
        self.assertEqual(a.stages, b.stages)


if __name__ == "__main__":
    unittest.main()
