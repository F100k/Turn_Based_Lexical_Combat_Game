# Developer Documentation — `dev/`

This directory houses the Django web application, combat engine interface, presentation templates, and session-based state management for the **Turn-Based Lexical Combat Game**.

---

## Architecture Overview

```
dev/
├── manage.py                    # Django CLI management script
├── db.sqlite3                   # Local SQLite database (MatchHistory & sessions)
├── lexical_game/                # Core Django project configuration
│   ├── settings.py              # Application settings, installed apps, static/templates
│   ├── urls.py                  # Root URL routing (routes to combat.urls)
│   ├── wsgi.py                  # WSGI entrypoint
│   └── asgi.py                  # ASGI entrypoint
├── combat/                      # Combat application (views, API, models, templates)
│   ├── apps.py                  # Pre-warms NLP models during server startup
│   ├── models.py                # MatchHistory database model for leaderboard & tracking
│   ├── scoring.py               # Unified engine factory loading GloVe and SBERT encoders
│   ├── views.py                 # Views for arena, battle state, attack evaluation, & CSV export
│   ├── urls.py                  # Endpoint routing (/api/start/, /api/attack/, /leaderboard/, etc.)
│   ├── tests.py                 # Unit & integration tests for web endpoints
│   └── templates/combat/        # Responsive HTML5 arena and leaderboard templates
│       ├── arena.html           # Main combat interface with CSS character sprites and audio-visual FX
│       └── leaderboard.html     # Top scores and player rankings
└── lexical/                     # Standalone Python scoring library (shared pipeline)
    ├── config.py                # EngineConfig & EngineData dataclasses and loaders
    ├── engine.py                # LexicalCombatEngine (domain gating, relevance, damage curves)
    ├── loader.py                # Engine builder helpers
    ├── session.py               # GameSession state and public_result dictionary sanitization
    ├── target_bank.py           # Target word validation and stage generators
    ├── text.py                  # Input sanitization and token validation
    └── encoders/
        ├── base.py              # Abstract encoder protocol
        └── sentence_transformer.py  # Cached SentenceTransformerEncoder wrapper
```

---

## Core Components

### 1. `dev/combat/scoring.py` (Model Abstraction Layer)
Unifies both embedding architectures behind a single interface (`LexicalCombatEngine`):
- **GloVe Mean Pooling (`GloveEncoder`)**: Reads `glove_module/glove.6B.100d.txt` ($d=100$), removes stopwords (`a`, `the`, `in`, etc.), computes phrase vector averages, and applies $L_2$ normalization.
- **Pretrained Sentence-BERT (`SentenceTransformerEncoder`)**: Wraps `sentence-transformers/all-MiniLM-L6-v2` ($d=384$) with an LRU cache and offline cache folder loading (`HF_HUB_OFFLINE=1`).
- **Shared Data & Config**: Both models load shared anchors and calibrated thresholds from `shared_data/game_data.json` and `shared_data/engine_config.json`.

### 2. `dev/combat/views.py` (Combat Controller & Session State)
Maintains game state inside Django's session dictionary (`request.session['battle']`), ensuring safe turn progression and logging:
- `arena_view(request)`: Renders `arena.html` with default topic and stage metadata.
- `start_endpoint(request)`: Handles `POST /api/start/`. Initializes player HP (100), enemy HP (200), selected model (`sbert` or `glove`), and stage (0 to 3).
- `attack_endpoint(request)`: Handles `POST /api/attack/`. Validates player input (1–3 words, non-empty, cannot repeat target), evaluates semantic damage, triggers enemy counterattack (`8 + (turn % 5) * 2`), and updates match status (`active`, `won`, `lost`).
- `export_endpoint(request)`: Handles `GET /api/export/`. Exports all trial attacks in the current session as a structured CSV (`trial_logs.csv`).
- `leaderboard_page(request)` & `leaderboard_endpoint(request)`: Provides HTML and JSON rankings of completed matches from the `MatchHistory` database table.

### 3. `dev/combat/templates/combat/arena.html` (Presentation Layer)
A pure CSS/HTML5 fantasy game arena requiring no external JS frameworks:
- **Responsive Layout**: Top setup bars, prominent `#target-card`, animated combat stage, battle log, and attack form.
- **CSS Sprite Animation**: Custom CSS Mage (player) and Shadow Beast (boss) sprites with floating idles, attack lunges, impact shakes, and particle effects.
- **Evaluation Banner**: Bottom evaluation pill dynamically updating with attack tier, domain qualification, and damage level.

---

## API Endpoints

| Endpoint | Method | Payload / Params | Response | Description |
|---|---|---|---|---|
| `/` | `GET` | — | HTML | Main battle arena interface |
| `/api/start/` | `POST` | `{"backend": "sbert", "stage": 1, "player_name": "Challenger"}` | JSON | Resets and starts a battle session |
| `/api/attack/` | `POST` | `{"attack_text": "binary search"}` | JSON | Evaluates attack, calculates damage and counterattack |
| `/api/export/` | `GET` | — | CSV File | Downloads `trial_logs.csv` of all session turns |
| `/leaderboard/` | `GET` | — | HTML | Top player rankings and match history |
| `/api/leaderboard/` | `GET` | — | JSON | JSON list of finished matches |

---

## Running the Web Application Manually

```bash
# 1. Activate environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Run database migrations
python dev/manage.py migrate

# 3. Start development server
python dev/manage.py runserver 127.0.0.1:8000
```
Then navigate to [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.
