"""Lexical Combat Game scoring package: engine, encoders, target bank and game session.

Typical use:
    from lexical import load_game, GameSession
    game = load_game("self_trained", "artifacts/self_trained")
    session = GameSession.start(game.eligible, game.game_data.boss_names)
    event = session.attack(game.engine, "chief executive")
"""
from .config import EngineConfig, EngineData, GameData, load_engine_config, load_game_data
from .engine import LexicalCombatEngine
from .loader import LoadedGame, load_game, verify_artifacts
from .session import GameSession, public_result
from .text import clean_text, validate_player_input

__all__ = [
    "EngineConfig", "EngineData", "GameData", "GameSession", "LexicalCombatEngine", "LoadedGame",
    "clean_text", "load_engine_config", "load_game", "load_game_data", "public_result",
    "validate_player_input", "verify_artifacts",
]
