"""Engine parameters and game data, loaded from JSON files exported by the notebook."""
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

PACKAGE_DATA = Path(__file__).resolve().parent / "data" / "game_data.json"


@dataclass(frozen=True)
class EngineConfig:
    """Tunable parameters (notebook Section 3). Defaults are pre-calibration values."""
    # Domain gate (debiased score = top-k anchor cosine minus the domain's noise background)
    anchor_topk: int = 2
    gate_primary_min: float = 0.20
    gate_secondary_min: float = 0.08
    gate_secondary_max_gap: float = 0.04
    gate_weak_max_gap: float = 0.10
    domain_primary_bonus: float = 1.00
    domain_secondary_multiplier: float = 0.75
    domain_weak_secondary_multiplier: float = 0.50
    display_softmax_temperature: float = 0.05
    # Target relevance r = (cos - noise) / (ceiling - noise)
    noise_percentile: int = 90
    ceiling_topk: int = 3
    min_calibration_span: float = 0.15
    min_span_mode: str = "median"
    primary_graze_floor: bool = True
    primary_graze_noise_pct: int = 50
    # Damage curve on r in [0, 1]
    relevance_min: float = 0.10
    damage_gamma: float = 1.0
    graze_min_percent: float = 5
    max_hit_percent: float = 45
    chip_percent: float = 4
    # Penalties and input rules
    generic_penalty: float = 0.60
    echo_relative: float = 0.60
    echo_penalty: float = 0.50
    block_target_word_in_guess: bool = True
    tier_cuts: tuple = ((38, "CRITICAL STRIKE"), (26, "STRONG HIT"), (14, "SOLID HIT"))


@dataclass(frozen=True)
class EngineData:
    """Vocabularies the engine encodes."""
    domains: tuple
    anchors: dict
    noise_vocab: tuple
    examples: dict
    pool_vocab: dict
    generic_terms: frozenset
    calibration_targets: dict


@dataclass(frozen=True)
class GameData:
    """Engine vocabularies plus what the game flow needs (stages, bosses, extra targets)."""
    engine_data: EngineData
    game_stages: list
    boss_names: dict
    extra_targets: dict = field(default_factory=dict)


def load_engine_config(path) -> EngineConfig:
    """EngineConfig from engine_config.json (dataclasses.asdict of the calibrated notebook config)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(EngineConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown EngineConfig keys in {path}: {unknown}")
    if "tier_cuts" in raw:
        raw["tier_cuts"] = tuple((float(cut), str(name)) for cut, name in raw["tier_cuts"])
    return EngineConfig(**raw)


def load_game_data(path=None) -> GameData:
    """GameData from game_data.json (defaults to the copy shipped with this package)."""
    raw = json.loads(Path(path or PACKAGE_DATA).read_text(encoding="utf-8"))
    engine_data = EngineData(
        domains=tuple(raw["domains"]),
        anchors={d: list(v) for d, v in raw["anchors"].items()},
        noise_vocab=tuple(raw["noise_vocab"]),
        examples={d: list(v) for d, v in raw["examples"].items()},
        pool_vocab={d: list(v) for d, v in raw["pool_vocab"].items()},
        generic_terms=frozenset(raw["generic_terms"]),
        calibration_targets=dict(raw["calibration_targets"]),
    )
    return GameData(engine_data, list(raw["game_stages"]), dict(raw["boss_names"]),
                    dict(raw.get("extra_targets", {})))
