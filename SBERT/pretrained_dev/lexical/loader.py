"""Build one ready-to-play game (engine + target bank) per backend and artifacts folder, once per process."""
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from .config import GameData, load_engine_config, load_game_data
from .engine import LexicalCombatEngine
from .target_bank import build_target_bank, eligible_targets

EXPORT_VERSION = 1
BACKENDS = ("pretrained",)

_CACHE = {}
_LOCK = threading.Lock()


@dataclass
class LoadedGame:
    """Everything the web app needs: a warmed-up engine and the screened targets."""
    backend: str
    engine: LexicalCombatEngine
    game_data: GameData
    target_bank: list
    eligible: dict
    manifest: dict


def read_manifest(folder) -> dict:
    path = Path(folder) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No manifest.json in {folder}. Run the notebook's web-export cell first.")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_artifacts(folder, check_hashes: bool = True) -> list:
    """List of problems with an export folder (empty list = OK)."""
    folder = Path(folder)
    manifest = read_manifest(folder)
    problems = []
    if manifest.get("export_version") != EXPORT_VERSION:
        problems.append(f"export_version {manifest.get('export_version')} != {EXPORT_VERSION}")
    for name, meta in manifest.get("files", {}).items():
        path = folder / name
        if not path.is_file():
            problems.append(f"missing file: {name}")
        elif check_hashes and _sha256(path) != meta["sha256"]:
            problems.append(f"hash mismatch: {name}")
    return problems


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_encoder(backend: str, folder: Path, manifest: dict):
    if backend == "pretrained":
        from .encoders.sentence_transformer import SentenceTransformerEncoder
        return SentenceTransformerEncoder.from_dir(folder)
    raise ValueError(f"Unknown backend {backend!r}; expected one of {BACKENDS}")


def load_game(backend: str, artifacts_dir, check_hashes: bool = False) -> LoadedGame:
    """Load (or reuse) the engine for a backend. artifacts_dir holds the notebook export files."""
    folder = Path(artifacts_dir).resolve()
    key = (backend, str(folder))
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        manifest = read_manifest(folder)
        if manifest.get("backend") != backend:
            raise ValueError(f"{folder} holds a {manifest.get('backend')!r} export, not {backend!r}")
        problems = verify_artifacts(folder, check_hashes)
        if problems:
            raise ValueError(f"Invalid artifacts in {folder}: {problems}")
        game_data_path = folder / "game_data.json"
        game_data = load_game_data(game_data_path if game_data_path.is_file() else None)
        config = load_engine_config(folder / "engine_config.json")
        encoder = _build_encoder(backend, folder, manifest)
        engine = LexicalCombatEngine(encoder, game_data.engine_data, config)
        bank = build_target_bank(engine, game_data.extra_targets)
        loaded = LoadedGame(backend, engine, game_data, bank, eligible_targets(bank, game_data.engine_data.domains),
                            manifest)
        _CACHE[key] = loaded
        return loaded


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
