# self_trained_dev — web-ready scorer for the self-trained word embeddings

Python package that plays the game with the model trained in `self_trained/` (Word2Vec / FastText / PPMI-SVD), **without the notebook**. It is the step between the research notebook and the Django app in `dev/`.

- Runtime dependency: **numpy only** (no gensim, pandas, torch).
- Same scoring as the notebook: verified against `golden_cases.json` exported by the notebook.
- Nothing is trained here; the model and its calibrated settings come from the notebook export.

## Folder layout

```
self_trained_dev/
├── lexical/                    # the package (copy this into the web project)
│   ├── __init__.py             # public API: load_game, GameSession, ...
│   ├── text.py                 # clean_text, validate_player_input (notebook 2.1)
│   ├── config.py               # EngineConfig / EngineData + JSON loaders
│   ├── engine.py               # LexicalCombatEngine (notebook 3.1, pandas-free)
│   ├── target_bank.py          # target screening + random stages (notebook 11)
│   ├── session.py              # GameSession, serialisable to a dict (Play Game cell)
│   ├── loader.py               # load_game(): builds the engine once per process
│   ├── encoders/
│   │   ├── base.py             # Encoder interface (encode / is_known)
│   │   └── word_vectors.py     # WordVectorEncoder + PhraseMerger (notebook 5.1, no gensim)
│   └── data/game_data.json     # default game data (same as the notebook)
├── artifacts/                  # put the notebook export here (git-ignored)
├── tests/                      # unittest suite
├── play_cli.py                 # terminal game / golden check
└── requirements.txt
```

## Step 1 — Export the model from the notebook

1. Open `self_trained/Lexical_Combat_Game_SelfTrained_WordEmbeddings.ipynb` in Colab.
2. In cell 0.1 set `RUN_WEB_EXPORT = True`. To reuse the trained models instead of training again, also set `RUN_DATA_PREP = False`, `RUN_TRAINING = False`, `RUN_HP_SEARCH = False` (keep `LOAD_TUNED_PARAMS = True`).
3. Runtime → Run all. Section **13. Export for the web game** writes `MyDrive/lexical_combat_selftrained/artifacts/web_export/self_trained/`.
4. Download that folder's files into `self_trained_dev/artifacts/`.

The export keeps the 100k most frequent tokens plus every game word, stores vectors as float16, and prints the size of each file (all must be < 100 MB). It also re-scores the golden cases with the trimmed model and shows any difference from the full model.

| File | Content |
|---|---|
| `manifest.json` | model, composition, sizes, file hashes |
| `vocab.json`, `vectors.npy` | trimmed vocabulary and unit vectors (float16) |
| `counts.json` | corpus counts for SIF weights |
| `phrases.json.gz` | bigram / trigram phrase merges |
| `common_component.npy` | SIF common component |
| `stopwords.json` | stopwords used by SIF |
| `engine_config.json` | calibrated `EngineConfig` |
| `game_data.json` | anchors, pools, noise words, stages, bosses |
| `golden_cases.json` | expected results for parity tests |

**FastText note:** subword vectors are not exported, so on the web a word outside the exported vocabulary becomes "Unknown word" (the turn is not counted). If the playing model is FastText and this matters, set `WEB_EXPORT_MODEL = "word2vec"` in the export cell.

## Step 2 — Check and play locally

```bash
pip install -r requirements.txt
python play_cli.py --artifacts artifacts --check   # verify hashes + golden cases + latency
python play_cli.py --artifacts artifacts           # play in the terminal
python -m unittest                                  # tests (golden test runs when artifacts/ is filled)
```

## Step 3 — Use it from code (Django or anything else)

```python
from lexical import GameSession, load_game

game = load_game("self_trained", "path/to/artifacts")      # load once per process (cached)
session = GameSession.start(game.eligible, game.game_data.boss_names, rounds=4)

event = session.attack(game.engine, "chief executive")     # one turn
event["result"]       # gate, relevance, damage, tier, reason (JSON-safe)
event["counted"]      # False for invalid / unknown words
event["stage_ended"]  # stage log when the boss was defeated
event["state"]        # what the screen needs: HP, target, topic, turns

hint = session.hint(game.engine)
session.skip()
session.end_game()

saved = session.to_dict()                  # store in request.session or a database row
session = GameSession.from_dict(saved)
```

For Django: call `load_game(...)` in `AppConfig.ready()` (or lazily on first request), keep the session dict in `request.session`, and return `event` from the JSON API.

## Notes

- **Do not edit the scoring code here to change game balance.** Change it in the notebook, re-run calibration and export again, so the notebook and the web game stay identical.
- `lexical/` is identical in `pretrained_dev/` except for the encoder file, so both backends can later live in one `combat/lexical/` package.
- The engine is read-only after loading; per-player state lives only in `GameSession`.
