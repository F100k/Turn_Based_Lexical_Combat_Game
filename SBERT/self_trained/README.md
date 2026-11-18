# Self-Trained Word Embeddings

Notebook: `Lexical_Combat_Game_SelfTrained_WordEmbeddings.ipynb`

This notebook replaces the game's pretrained scorer (MiniLM) with **word embeddings trained from random initialization** on our own corpus. Three model families are trained and compared:

| Model | Type | Role |
|---|---|---|
| PPMI + Truncated SVD | count-based (classic distributional semantics) | simple, explainable baseline |
| Word2Vec (CBOW / skip-gram) | prediction-based | main candidate |
| FastText (skip-gram + character n-grams) | prediction-based + subword | handles unknown / misspelled words |
| MiniLM (`all-MiniLM-L6-v2`) | pretrained Sentence Transformer | reference baseline only |

No pretrained weights are used for the three self-trained models.

## How to run

1. Open the notebook in Google Colab.
2. Set the flags in cell **0.1** (defaults are fine for a full run; `QUICK_RUN = True` tests the pipeline in a few minutes).
3. **Runtime → Run all.**

The first full run takes about 1–1.5 hours on free Colab (corpus download + training). Every step is cached on Google Drive in `MyDrive/lexical_combat_selftrained/`, so later runs load the cache and finish in a few minutes.

## Pipeline, step by step

### Step 0 — Setup
- Installs only missing packages (`gensim`, `datasets`, `sentence-transformers`) without upgrading numpy / pandas / scipy / torch.
- Sets seeds, mounts Google Drive and creates the `corpus/`, `models/`, `artifacts/` folders.

### Step 1 — Data registry (no model calls)
- 4 domains (Business, Technology, Science, Arts), each with 8 anchor sentences for the domain gate.
- Term pools per domain, a noise vocabulary of everyday nouns, calibration cases and a 28-case holdout with all-new targets.
- This data is never written to the training corpus.

### Step 2 — Text utilities and data checks
- `clean_text()`: lowercase, non-letters → spaces. The same function is used by the game and the corpus, which prevents train/serve skew.
- Builds a non-circular gate calibration set (phrases not written in the anchors) and runs leakage checks.

### Step 3 — Game engine and calibration functions
- `LexicalCombatEngine` scores each guess in two steps:
  1. **Domain gate:** top-k anchor cosine minus the domain background → primary / secondary / weak secondary / blocked.
  2. **Damage:** relevance `r = (cos − noise_floor) / (ceiling − noise_floor)` mapped to damage by a calibrated curve.
- Unknown words (OOV) get an "Unknown word" message and the turn is not counted.
- Grid-search functions for gate and damage, with automatic grid expansion when the best value sits on an edge.

### Step 4 — Corpus collection and preprocessing
- **Main source:** Simple English Wikipedia (Hugging Face `wikimedia/wikipedia`, `20231101.simple`).
- **Supplement:** English Wikipedia articles that contain game words still missing or rare (holdout words are never used to select articles). NLTK corpora are the last fallback.
- Preprocessing: sentence split → `clean_text()` → drop sentences < 3 tokens → bigram + trigram `Phrases` (e.g. `board_of_directors`) → one sentence per line.
- Pass criterion: ≥ 95% of game words appear at least 5 times in the corpus.

### Step 5 — Encoder adapter
- `SelfTrainedEncoder` has the same `encode()` signature as `SentenceTransformer`, so the engine works unchanged.
- Phrase vectors are built from word vectors with `mean` or `sif` (SIF weighting + removal of the first principal component).
- Intrinsic evaluation on WordSim-353 and SimLex-999 (evaluation only).
- Selection score (calibration set + intrinsic only, holdout never used):
  `0.35 × gate macro F1 (CV) + 0.35 × damage in-band (LOTO-CV) + 0.10 × pacing (LOTO-CV) + 0.20 × mean Spearman`

### Step 6 — Training from scratch
- **6.1** Writes a search sample (~15% of the corpus) and a 20k-sentence sample for SIF.
- **6.2** PPMI-SVD: window {2, 5} × dim {100, 300} × composition {mean, sif}.
- **6.3** Word2Vec search: vector_size {100, 300} × window {5, 10} × sg {CBOW, skip-gram}. FastText search: vector_size {100, 300} at the best Word2Vec window.
- **6.4** Final training of the best config per family on the full corpus, within a time budget (fewer epochs or a corpus share if needed). `MODEL_CHOICE = "auto"` picks the family with the best selection score.

### Step 7 — Model loading and evaluation
- Loads the chosen model (and MiniLM as a reference).
- Adapter unit tests: output shape, unit norm, empty / OOV input, `evaluate_attack()`.
- Intrinsic metrics, nearest neighbours, anisotropy, t-SNE plot and silhouette score per domain.

### Step 8 — Calibration for the playing model
- Gate diagnosis on the old holdout with default thresholds.
- Gate grid (5-fold grouped CV) → damage grid (per-target loss + pacing, leave-one-target-out CV).
- Saved per model as `tuned_params_<model>.json`.
- Pass criterion: pacing met on every stage and out-of-domain leak ≤ 5%.

### Step 9 — Report and model comparison (calibration side)
- Banded cases, gate leak vs damaging leak, pacing per stage for the active model.
- Every model (PPMI-SVD, Word2Vec, FastText, MiniLM) gets its own calibration and is compared on intrinsic and game metrics.

### Step 10 — Holdout and error analysis
- The 28-case holdout is evaluated **once**, after every choice is fixed.
- Comparison table of all models on the holdout.
- Error analysis: cases our model gets wrong but MiniLM gets right (and the reverse), with likely causes (OOV, rare words, polysemy, phrase averaging, corpus bias).

### Step 11 — Play
- Re-screens the target bank with the active model.
- Simulated battles (3 classic stages + 1 random stage) showing gate → relevance → damage per turn.

### Step 12 — Report
- Builds a Notion-ready Markdown report from this run's numbers and saves it as `artifacts/report_selftrained.md`.

### Play Game (last cell)
- Interactive widget game: random topics and targets, with Attack / Hint / Skip / Quit / New game.

## Folder contents

| Folder | Content | In Git? |
|---|---|---|
| `artifacts/` | tuned parameters, search results, holdout logs, report | Yes |
| `corpus/` | processed corpus, phrase models, word counts | No (too large; cached on Drive) |
| `models/` | trained Word2Vec / FastText / PPMI-SVD vectors + `config.json` | No (too large; cached on Drive) |

## Data-leakage rules
- Game data (anchors, pools, calibration, holdout) is used only to measure corpus coverage, never written to the corpus.
- The holdout is never used to choose hyper-parameters, models, thresholds or supplement articles.
- WordSim-353 / SimLex-999 are used for evaluation only.

## Limitations
- The corpus is small compared with MiniLM's training data, so rare words get weaker vectors.
- Static embeddings have one vector per word (polysemy) and phrases are averaged from words (word order is lost).
- Multi-threaded gensim training is not bit-for-bit reproducible.
- Small calibration set and holdout (28 cases, 1 annotator).

## Data licenses
Wikipedia text: CC BY-SA 3.0 / GFDL (per the Hugging Face dataset card).
