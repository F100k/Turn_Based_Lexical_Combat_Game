# pretrained_dev: Web-Ready Scorer for the SBERT Model

This package runs the Lexical Combat Game scoring outside the notebook. It loads the Sentence-BERT model (`sentence-transformers/all-MiniLM-L6-v2`, base or fine-tuned) together with the game settings calibrated in `notebook/`, and provides the interface used by the Django application in `dev/`.

- The encoder is a pretrained Sentence-BERT model. The notebook optionally fine-tunes it on WordNet pairs and selects the final model (base or fine-tuned) by cross-validation; the selected model is exported with its calibrated domain-gate thresholds and damage curve.
- The scoring is identical to the notebook. This is verified against `golden_cases.json`, which the notebook exports.
- `sentence-transformers` and `torch` are imported only by the encoder module, so the remaining modules can be used without them.

## Folder Layout

```
pretrained_dev/
├── lexical/                         # scoring package (copied into the web project)
│   ├── __init__.py                  # public API: load_game, GameSession, ...
│   ├── text.py                      # clean_text, validate_player_input (notebook Section 1.4)
│   ├── config.py                    # EngineConfig, EngineData and JSON loaders
│   ├── engine.py                    # LexicalCombatEngine (notebook Section 6.1, without pandas)
│   ├── target_bank.py               # target screening and random stages (notebook Section 6.5c)
│   ├── session.py                   # GameSession, serialisable to a dict (notebook Appendix A)
│   ├── loader.py                    # load_game(): builds the engine once per process
│   ├── encoders/
│   │   ├── base.py                  # Encoder interface
│   │   └── sentence_transformer.py  # SentenceTransformerEncoder (LRU cache, lazy imports)
│   └── data/game_data.json          # default game data (same as the notebook)
├── artifacts/                       # destination for the notebook export
├── tests/                           # unit tests
├── play_cli.py                      # terminal game and golden-case check
└── requirements.txt
```

## Step 1: Export the Model and Settings from the Notebook

1. Open `notebook/Lexical_Combat_SBERT_Report.ipynb` in Google Colab.
2. In cell 0.1, set `RUN_WEB_EXPORT = True`. Keep `TRAIN_FINE_TUNED = "never"` and `LOAD_TUNED_PARAMS = True` so that the saved checkpoint and calibration are reused without retraining.
3. Select Runtime → Run all. **Appendix B. Export for the web game** writes the files below to `MyDrive/lexical_combat/artifacts/web_export/pretrained/`.
4. Copy the contents of that folder into `pretrained_dev/artifacts/`.

| File | Content |
|---|---|
| `manifest.json` | model identifier, embedding dimension, library versions and file hashes |
| `engine_config.json` | calibrated `EngineConfig` |
| `game_data.json` | domain anchors, term pools, noise words, stages and bosses |
| `golden_cases.json` | expected results used by the parity tests |
| `model/` | fine-tuned checkpoint; exported only when the final model is the fine-tuned one |

The JSON files are small and may be committed. Model weights are not committed: the fine-tuned checkpoint is stored in `artifacts/model/`, and the base model (about 90 MB) is downloaded from Hugging Face on first use into `artifacts/hf_cache/`. Both folders are listed in `.gitignore`. After the first download, set `HF_HUB_OFFLINE=1` to prevent further network access.

## Step 2: Install, Verify and Play Locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build of torch
pip install -r requirements.txt
python play_cli.py --artifacts artifacts --check   # verify file hashes, golden cases and latency
python play_cli.py --artifacts artifacts           # play in the terminal
python -m unittest                                 # run the tests (the golden test requires artifacts/)
```

Optional environment variables: `LEXICAL_ST_CACHE_DIR` (model cache folder) and `LEXICAL_TORCH_THREADS` (number of CPU threads).

## Step 3: Use the Package from Code

```python
from lexical import GameSession, load_game

game = load_game("pretrained", "path/to/artifacts")        # load once per process (cached)
session = GameSession.start(game.eligible, game.game_data.boss_names, rounds=4)

event = session.attack(game.engine, "chief executive")     # one turn
event["result"]       # gate, relevance, damage, tier and reason (JSON-serialisable)
event["counted"]      # False for invalid guesses
event["stage_ended"]  # stage summary when the boss is defeated
event["state"]        # data required by the user interface: HP, target, topic and turns

hint = session.hint(game.engine)
session.skip()
session.end_game()

saved = session.to_dict()                  # store in request.session or a database row
session = GameSession.from_dict(saved)
```

In Django, call `load_game(...)` in `AppConfig.ready()` or on the first request, since loading the model takes a few seconds. Store the session dictionary in `request.session` and return `event` from the JSON API.

## Notes

- **Game balance must not be changed in this package.** Change the settings in the notebook, re-run the calibration and export again, so that the notebook and the web game remain identical.
- MiniLM uses a subword tokenizer, so every input receives a vector and no "unknown word" case occurs, as in the notebook.
