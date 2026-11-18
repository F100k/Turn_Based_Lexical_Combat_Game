"""GameSession: one game of random stages, serialisable to a dict (for a web session or database row).

Same rules as the notebook's Play Game cell: invalid or unknown guesses do not count as a turn,
hint / skip / quit are available, and every stage keeps a log of its guesses.
The engine is passed to each call and never stored, so the session stays a plain dict of state.
"""
import random

from .engine import LexicalCombatEngine
from .target_bank import RANDOM_ROUNDS, random_stages

GATE_SHORT = {"primary": "primary", "secondary": "secondary", "weak_secondary": "weak secondary",
              "out_of_domain": "blocked"}

# Fields of an evaluate_attack() result that a client needs to show one turn.
PUBLIC_RESULT_KEYS = (
    "valid", "reason", "unknown", "word", "similarity", "noise_floor", "ceiling", "relevance",
    "domain_scores", "domain_probs", "domain_lead", "domain_allowed", "domain_match_type", "multiplier",
    "is_generic", "generic_multiplier", "is_echo", "echo_multiplier", "graze_floor", "base_percent",
    "hit_percent", "damage_percent", "damage", "actual_damage", "overkill", "tier", "latency_ms",
)


def public_result(result: dict) -> dict:
    """JSON-safe subset of an evaluate_attack() result."""
    out = {}
    for key in PUBLIC_RESULT_KEYS:
        if key in result:
            value = result[key]
            if isinstance(value, dict):
                value = {k: float(v) for k, v in value.items()}
            elif isinstance(value, float):
                value = float(value)
            out[key] = value
    return out


class GameSession:
    """State of one game. Create with GameSession.start(...), restore with GameSession.from_dict(...)."""

    def __init__(self, stages: list, seed: int):
        self.stages = stages
        self.seed = seed
        self.index = 0
        self.quit = False
        self.stage_logs = []
        self.hp, self.used, self.turns = 0, set(), 0
        self._start_stage()

    @classmethod
    def start(cls, eligible: dict, boss_names: dict, rounds: int = RANDOM_ROUNDS, seed: int = None) -> "GameSession":
        """New game with random topics and targets from the eligible targets per domain."""
        seed = random.randrange(1_000_000_000) if seed is None else int(seed)
        return cls(random_stages(eligible, boss_names, rounds, seed), seed)

    # ---- state ------------------------------------------------------------
    def _start_stage(self) -> None:
        s = self.stage
        self.hp, self.used, self.turns = s["hp"], set(), 0
        self.stage_logs.append({"stage": s["stage"], "enemy": s["enemy"], "domain": s["domain"],
                                "target": s["target"], "hp_max": s["hp"], "hp_left": s["hp"],
                                "turns": 0, "status": "In progress", "guesses": []})

    @property
    def stage(self) -> dict:
        return self.stages[min(self.index, len(self.stages) - 1)]

    @property
    def current_log(self) -> dict:
        return self.stage_logs[-1]

    @property
    def finished(self) -> bool:
        return self.quit or self.index >= len(self.stages)

    @property
    def defeated_count(self) -> int:
        return sum(1 for s in self.stage_logs if s["status"] == "Defeated")

    def state(self) -> dict:
        """What a client needs to draw the current screen."""
        s = self.stage
        return {"finished": self.finished, "stage_number": s["stage"], "stage_count": len(self.stages),
                "domain": s["domain"], "enemy": s["enemy"], "target": s["target"],
                "hp": max(self.hp, 0), "hp_max": s["hp"], "turns": self.turns,
                "used_guesses": sorted(self.used), "defeated": self.defeated_count}

    def _end_stage(self, status: str) -> dict:
        """Close the current stage and start the next one; returns the closed stage log."""
        log = self.current_log
        log.update(status=status, hp_left=max(self.hp, 0), turns=self.turns)
        self.index += 1
        if not self.finished:
            self._start_stage()
        return log

    # ---- actions ------------------------------------------------------------
    def attack(self, engine: LexicalCombatEngine, guess: str) -> dict:
        """Score one guess. Returns {"result", "counted", "stage_ended", "game_over", "stage_log", "state"}."""
        if self.finished:
            return {"result": {"valid": False, "reason": "The game is over."}, "counted": False,
                    "stage_ended": None, "game_over": True, "state": self.state()}
        s = self.stage
        result = engine.evaluate_attack(guess, s["target"], s["domain"], self.used, s["hp"])
        counted = bool(result["valid"])
        if counted:
            result = engine.resolve_damage(result, s["hp"], self.hp)
            self.turns += 1
            self.used.add(result["word"])
            self.hp -= result["actual_damage"]
            self.current_log["guesses"].append({
                "turn": self.turns, "guess": result["word"], "gate": GATE_SHORT[result["domain_match_type"]],
                "relevance": round(result["relevance"], 2), "damage_percent": round(result["hit_percent"], 1),
                "damage": result["actual_damage"], "tier": result["tier"], "boss_hp_after": max(self.hp, 0)})
            self.current_log.update(hp_left=max(self.hp, 0), turns=self.turns)
        ended = self._end_stage("Defeated") if self.hp <= 0 else None
        return {"result": public_result(result), "counted": counted, "stage_ended": ended,
                "game_over": self.finished, "state": self.state()}

    def hint(self, engine: LexicalCombatEngine) -> str:
        if self.finished:
            return ""
        return engine.hint(self.stage["target"], self.stage["domain"], self.used)

    def skip(self) -> dict:
        """Give up the current boss and move on."""
        if self.finished:
            return {"stage_ended": None, "game_over": True, "state": self.state()}
        ended = self._end_stage("Skipped")
        return {"stage_ended": ended, "game_over": self.finished, "state": self.state()}

    def end_game(self) -> dict:
        """Quit now; returns every stage log."""
        if not self.finished:
            self.current_log.update(status="Quit", hp_left=max(self.hp, 0), turns=self.turns)
            self.quit = True
        return {"stage_logs": self.stage_logs, "defeated": self.defeated_count, "state": self.state()}

    # ---- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        return {"stages": self.stages, "seed": self.seed, "index": self.index, "quit": self.quit,
                "stage_logs": self.stage_logs, "hp": self.hp, "used": sorted(self.used), "turns": self.turns}

    @classmethod
    def from_dict(cls, data: dict) -> "GameSession":
        session = cls.__new__(cls)
        session.stages, session.seed = data["stages"], data["seed"]
        session.index, session.quit = data["index"], data["quit"]
        session.stage_logs = data["stage_logs"]
        session.hp, session.used, session.turns = data["hp"], set(data["used"]), data["turns"]
        return session
