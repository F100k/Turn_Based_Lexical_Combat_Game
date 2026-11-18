# pretrained_dev — web-ready scorer for the pretrained Sentence Transformer (MiniLM)

Python package that plays the game with the pretrained `sentence-transformers/all-MiniLM-L6-v2` model and the settings calibrated in `pretrained/`, **without the notebook**. It is the step between the research notebook and the Django app in `dev/`.

- The model is **pretrained, not trained by us**; only the game thresholds and damage curve were calibrated in the notebook.
- Same scoring as the notebook: verified against `golden_cases.json` exported by the notebook.
- `sentence-transformers` / `torch` are imported only by the encoder, so the rest of an app can run without them.

## Folder layout

```
pretrained_dev/
├── lexical/                       # the package (copy this into the web project)
│   ├── __init__.py                # public API: load_game, GameSession, ...
│   ├── text.py                    # clean_text, validate_player_input (notebook 2.1)
│   ├── config.py                  # EngineConfig / EngineData + JSON loaders
│   ├── engine.py                  # LexicalCombatEngine (notebook 3, pandas-free)
│   ├── target_bank.py             # target screening + random stages (notebook 8)
│   ├── session.py                 # GameSession, serialisable to a dict (Play Game cell)
│   ├── loader.py                  # load_game(): builds the engine once per process
│   ├── encoders/
│   │   ├── base.py                # Encoder interface
│   │   └── sentence_transformer.py  # SentenceTransformerEncoder (LRU cache, lazy imports)
│   └── data/game_data.json        # default game data (same as the notebook)
├── artifacts/                     # put the notebook export here
├── tests/                         # unittest suite
├── play_cli.py                    # terminal game / golden check
└── requirements.txt
```

## Step 1 — Export the settings from the notebook

1. Open `pretrained/Lexical_Combat_Game_SentenceTransformer_MiniLM.ipynb` in Colab.
2. In cell 0.1 set `RUN_WEB_EXPORT = True` (keep `LOAD_TUNED_PARAMS = True` so calibration is loaded, not re-run).
3. Runtime → Run all. Section **10. Export for the web game** writes `MyDrive/lexical_combat/artifacts/web_export/pretrained/`.
4. Download that folder's files into `pretrained_dev/artifacts/`.

| File | Content |
|---|---|
| `manifest.json` | model id, dim, library versions, file hashes |
| `engine_config.json` | calibrated `EngineConfig` |
| `game_data.json` | anchors, pools, noise words, stages, bosses |
| `golden_cases.json` | expected results for parity tests |
| `model/` | only when `MODEL_CHOICE = "fine_tuned"`: the fine-tuned checkpoint |

The JSON files are small and can be committed. The model weights (~90 MB) are downloaded from Hugging Face on first load into `artifacts/hf_cache/` (git-ignored). Set `HF_HUB_OFFLINE=1` after the first download to never call the network.

## Step 2 — Install, check and play locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch (much smaller)
pip install -r requirements.txt
python play_cli.py --artifacts artifacts --check   # verify hashes + golden cases + latency
python play_cli.py --artifacts artifacts           # play in the terminal
python -m unittest                                  # tests (golden test runs when artifacts/ is filled)
```

Environment variables: `LEXICAL_ST_CACHE_DIR` (model cache folder), `LEXICAL_TORCH_THREADS` (limit CPU threads).

## Step 3 — Use it from code (Django or anything else)

```python
from lexical import GameSession, load_game

game = load_game("pretrained", "path/to/artifacts")        # load once per process (cached)
session = GameSession.start(game.eligible, game.game_data.boss_names, rounds=4)

event = session.attack(game.engine, "chief executive")     # one turn
event["result"]       # gate, relevance, damage, tier, reason (JSON-safe)
event["counted"]      # False for invalid guesses
event["stage_ended"]  # stage log when the boss was defeated
event["state"]        # what the screen needs: HP, target, topic, turns

hint = session.hint(game.engine)
session.skip()
session.end_game()

saved = session.to_dict()                  # store in request.session or a database row
session = GameSession.from_dict(saved)
```

For Django: call `load_game(...)` in `AppConfig.ready()` (or lazily on first request; loading the model takes a few seconds), keep the session dict in `request.session`, and return `event` from the JSON API.

## Notes

- **Do not edit the scoring code here to change game balance.** Change it in the notebook, re-run calibration and export again, so the notebook and the web game stay identical.
- `lexical/` is identical in `self_trained_dev/` except for the encoder file, so both backends can later live in one `combat/lexical/` package.
- MiniLM gives every input a vector (subword tokenizer), so there is no "Unknown word" case here, same as the notebook.
