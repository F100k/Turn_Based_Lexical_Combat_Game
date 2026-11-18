"""Parity with the notebook: re-score golden_cases.json from the export (skipped if no artifacts)."""
import json
import os
import unittest
from pathlib import Path

from lexical import load_game
from lexical.loader import read_manifest

ARTIFACTS = Path(os.environ.get("LEXICAL_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts"))


@unittest.skipUnless((ARTIFACTS / "golden_cases.json").is_file(), f"no export in {ARTIFACTS}")
class GoldenParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.game = load_game(read_manifest(ARTIFACTS)["backend"], ARTIFACTS, check_hashes=True)
        except ImportError as error:   # e.g. sentence-transformers / torch not installed
            raise unittest.SkipTest(f"backend dependencies missing: {error}")
        cls.golden = json.loads((ARTIFACTS / "golden_cases.json").read_text(encoding="utf-8"))

    def test_cases(self):
        for case in self.golden["cases"]:
            with self.subTest(source=case["source"], target=case["target"], guess=case["guess"]):
                r = self.game.engine.evaluate_attack(case["guess"], case["target"], case["domain"],
                                                     set(case["used"]), case["hp_max"])
                r = self.game.engine.resolve_damage(r, case["hp_max"], case["hp_now"])
                e = case["expected"]
                self.assertEqual(r["valid"], e["valid"])
                self.assertEqual(r.get("reason"), e.get("reason"))
                if e["valid"]:
                    self.assertEqual(r["domain_match_type"], e["domain_match_type"])
                    self.assertEqual(r["tier"], e["tier"])
                    self.assertAlmostEqual(r["similarity"], e["similarity"], delta=1e-3)
                    self.assertLessEqual(abs(r["actual_damage"] - e["actual_damage"]), 1)

    def test_target_bank(self):
        mine = [(r["domain"], r["target"], r["eligible"]) for r in self.game.target_bank]
        theirs = [(r["domain"], r["target"], r["eligible"]) for r in self.golden["target_bank"]]
        self.assertEqual(mine, theirs)

    def test_hints(self):
        for h in self.golden["hints"]:
            self.assertEqual(self.game.engine.hint(h["target"], h["domain"], set(h["used"])), h["hint"])


if __name__ == "__main__":
    unittest.main()
