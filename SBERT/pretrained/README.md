# Pretrained Sentence Transformer (MiniLM) Baseline

Notebook: `Lexical_Combat_Game_SentenceTransformer_MiniLM.ipynb`

This notebook scores the game with a **pretrained Sentence Transformer**, `sentence-transformers/all-MiniLM-L6-v2`. It encodes a whole phrase into one vector (no word-vector averaging). It is the contextual baseline that the static word-embedding models (self-trained Word2Vec / FastText / PPMI-SVD, and GloVe) are compared against.

- **Default (`MODEL_CHOICE = "base"`):** the pretrained model is used as is. No model weights are trained.
- **Optional fine-tuning (Appendix):** the pretrained model can be fine-tuned on WordNet pairs. This is off by default (`RUN_FINE_TUNING = False`).
- **Calibration** only tunes the game's thresholds and damage curve, not the model itself.

## How to run

1. Open the notebook in Google Colab.
2. Set the flags in cell **0.1**.
3. **Runtime → Run all.**

Results are saved on Google Drive in `MyDrive/lexical_combat/artifacts/`. The run does not wait for input; the only interaction is the Google Drive pop-up.

## Pipeline, step by step

### Step 0 — Setup
- Installs only missing packages (`sentence-transformers`; `nltk` and `wordfreq` only for fine-tuning) without upgrading numpy / pandas / scipy / torch.
- Sets seeds and mounts Google Drive.

### Step 1 — Data registry (no model calls)
- 4 domains (Business, Technology, Science, Arts), each with 8 anchor sentences for the domain gate.
- Term pools per domain, a noise vocabulary of everyday nouns, calibration cases and a 28-case holdout with all-new targets.
- Game stages and holdout annotations.

### Step 2 — Text utilities and data checks
- `clean_text()`: lowercase, non-letters → spaces.
- Input validation (1–3 words, no repeated guesses, target word blocked).
- Builds a non-circular gate calibration set (phrases not written in the anchors) and runs leakage checks.

### Step 3 — Game engine
- `LexicalCombatEngine` scores each guess in two steps:
  1. **Domain gate:** top-k anchor cosine minus the domain background → primary (×1.00) / secondary (×0.75) / weak secondary (×0.50) / blocked.
  2. **Damage:** relevance `r = (cos − noise_floor) / (ceiling − noise_floor)` mapped to damage by a calibrated curve (tiers: Critical, Strong, Solid, Graze).
- The engine only calls `model.encode(...)`, so other encoders can be plugged in.

### Step 4 — Model selection
- Loads the pretrained MiniLM, or a fine-tuned checkpoint if `MODEL_CHOICE = "fine_tuned"` (falls back to base if none exists).
- Builds the engine with default (pre-calibration) settings.

### Step 5 — Calibration
- **5.2** Gate diagnosis on the old holdout with default thresholds.
- **5.3** Gate grid search: ANCHOR_TOPK {1, 2, 3} × thresholds, plateau selection, 5-fold grouped CV.
- **5.4** Damage-curve grid search: per-target loss + pacing penalty, leave-one-target-out CV.
- **5.5** Runs the grids (or loads a matching `tuned_params_<model>.json`) and applies the result.
- **5.6** Gate review: confusion matrix and worst errors.

### Step 6 — Report
- Banded cases, gate leak vs damaging leak, pacing per stage, tuned vs CV.
- Base vs fine-tuned comparison (each model recalibrated), when a fine-tuned checkpoint exists.

### Step 7 — Holdout
- Labels from inline annotations plus optional `annotator_*.xlsx / .csv` files (majority vote and kappa with ≥ 2 annotators).
- Evaluates the active model on the 28-case holdout once and checks the success criteria.

### Step 8 — Play
- Screens the target bank with the active model.
- Runs one simulated battle, or an interactive `input()` game when `PLAY_INTERACTIVE = True`.

### Appendix — Fine-tuning (optional)
- Builds training pairs from WordNet synonyms / definitions + curated pairs (calibration and holdout phrases excluded).
- Mines hard negatives with the base model and fine-tunes with MultipleNegativesRankingLoss (MNRL).
- Saves the checkpoint; it is used on the next run with `MODEL_CHOICE = "fine_tuned"`.

### Step 9 — Regression check and run summary
- Compares this run with the embedded results of the pre-refactor notebook (same model only).
- Prints the status and time of every section.

### Play Game (last cell)
- Interactive widget game: random topics and targets, with Attack / Hint / Skip / Quit / New game.

## Comparison with the self-trained models
The engine, data, calibration and holdout are the same as in `../self_trained/`, so results compare directly. The self-trained notebook also loads MiniLM as a reference and puts it in the same comparison tables.

| | This notebook | `self_trained/` |
|---|---|---|
| Model | MiniLM (Sentence Transformer) | Word2Vec, FastText, PPMI-SVD |
| Weights | pretrained (optional fine-tuning) | trained from random initialization |
| Phrase vector | encoded as a whole phrase | averaged from word vectors (mean / SIF) |
| Unknown words | handled by the subword tokenizer | OOV for Word2Vec / PPMI; FastText uses n-grams |
