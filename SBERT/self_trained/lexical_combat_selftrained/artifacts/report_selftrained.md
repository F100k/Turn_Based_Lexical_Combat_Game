# Lexical Combat Game — โมเดลที่ train เอง: ผลการทดลอง (2026-11-17)

**โมเดลที่ใช้เล่น:** PPMI-SVD — self-trained PPMI-SVD (mean, e125dab7dca5) · **โมเดลที่ train เองที่ดีที่สุด (calibration set):** PPMI-SVD · QUICK_RUN = False

## 1. Corpus

| source | articles | sentences | tokens |
|---|---|---|---|
| wikimedia/wikipedia/20231101.simple | 241757 | 3649891 | 37929604 |
| wikimedia/wikipedia/20231101.en (targeted) | 883 | 126887 | 2109079 |

- บทความ 242,640 · ประโยค 3,776,778 · tokens 31,682,210 · vocab (≥5) 292,875 · phrase tokens 163,323
- License: Wikipedia text, CC BY-SA 3.0 and GFDL (per the dataset card); https://huggingface.co/datasets/wikimedia/wikipedia
- Coverage ของคำในเกม: 99.9% (non-holdout 100.0%) → ผ่าน เกณฑ์ 95%
- Targeted supplement: สแกน 300,000 บทความ เก็บ 883 บทความ

## 2. Training

| model | params | composition | train s | corpus share | size MB | selection score |
|---|---|---|---|---|---|---|
| PPMI-SVD | {"windows": [2, 5], "max_dim": 300, "vocab": 50059, "cds_alpha": 0.75, "window_weight": "(W - d + 1) / W", "svd": "randomized, n_iter=4", "eigen_weight": 0.5} | mean | 148.100 | 1.000 | 120.900 | 0.673 |
| Word2Vec | {"negative": 10, "min_count": 5, "sample": 0.001, "vector_size": 300, "window": 10, "sg": 1, "epochs": 3} | mean | 1733.300 | 1.000 | 362.200 | 0.631 |
| FastText | {"negative": 10, "min_count": 5, "sample": 0.001, "vector_size": 100, "window": 10, "sg": 1, "epochs": 3, "min_n": 3, "max_n": 6, "bucket": 500000} | sif | 1903.800 | 1.000 | 327.900 | 0.591 |

## 3. Intrinsic evaluation

| model | WordSim-353 Spearman | WordSim-353 OOV pairs | WordSim-353 vocab-OOV pairs | SimLex-999 Spearman | SimLex-999 OOV pairs | SimLex-999 vocab-OOV pairs | anisotropy (noise-pair mean cos) | domain silhouette (cosine) | vocab | dim |
|---|---|---|---|---|---|---|---|---|---|---|
| PPMI-SVD | 0.684 | 0.034 | 0.034 | 0.326 | 0.047 | 0.047 | 0.129 | 0.122 | 50059 | 300 |
| Word2Vec | 0.599 | 0.003 | 0.003 | 0.323 | 0.004 | 0.004 | 0.462 | 0.163 | 292875 | 300 |
| FastText | 0.507 | 0.000 | 0.003 | 0.224 | 0.000 | 0.004 | 0.301 | 0.212 | 292875 | 100 |
| MiniLM (ref) | 0.729 | 0.000 | — | 0.446 | 0.000 | — | 0.291 | 0.112 | subword | 384 |

## 4. Calibration set (แต่ละโมเดลใช้ calibration ของตัวเอง)

| metric | PPMI-SVD | Word2Vec | FastText | MiniLM (ref) |
|---|---|---|---|---|
| role | self-trained | self-trained | self-trained | reference (pretrained) |
| WordSim-353 Spearman | 0.684 | 0.599 | 0.507 | 0.729 |
| WordSim-353 OOV pairs | 0.034 | 0.003 | 0.000 | 0.000 |
| SimLex-999 Spearman | 0.326 | 0.323 | 0.224 | 0.446 |
| SimLex-999 OOV pairs | 0.047 | 0.004 | 0.000 | 0.000 |
| domain silhouette (cosine) | 0.122 | 0.163 | 0.212 | 0.112 |
| AUC related vs noise | 0.986 | 0.989 | 0.996 | 0.990 |
| Anisotropy (noise-pair mean cos) | 0.129 | 0.462 | 0.301 | 0.291 |
| Mean cos strong/medium | 0.398 | 0.565 | 0.543 | 0.461 |
| Out-of-domain mean cos (blocked pairs) | 0.028 | 0.299 | 0.052 | 0.206 |
| Mean r blocked | 0.014 | 0.257 | 0.053 | 0.040 |
| Gate AUC in vs out (debiased score) | 0.975 | 0.868 | 0.953 | 0.952 |
| gate macro F1 (tuned) | 0.756 | 0.679 | 0.709 | 0.705 |
| gate macro F1 (CV) | 0.756 | 0.673 | 0.688 | 0.663 |
| in-domain -> primary (CV) | 0.911 | 0.938 | 0.893 | 0.848 |
| ambiguous pass (CV) | 0.621 | 0.448 | 0.552 | 0.483 |
| out-of-domain leak (CV) | 0.021 | 0.031 | 0.031 | 0.046 |
| gate leak % (tuning set) | 0.003 | 0.000 | 0.003 | 0.012 |
| damaging leak % (tuning set) | 0.018 | 0.028 | 0.028 | 0.025 |
| damage in-band (tuned) | 0.680 | 0.580 | 0.580 | 0.640 |
| damage in-band (LOTO-CV) | 0.640 | 0.560 | 0.600 | 0.600 |
| pacing met | 6/6 | 6/6 | 5/6 | 6/6 |
| soft grid edges | none | none | none | none |

## 5. Holdout (28 เคส, labels: single_annotator, ประเมินครั้งเดียว)

| metric | PPMI-SVD | Word2Vec | FastText | MiniLM (ref) |
|---|---|---|---|---|
| role | self-trained | self-trained | self-trained | reference (pretrained) |
| WordSim-353 Spearman | 0.684 | 0.599 | 0.507 | 0.729 |
| SimLex-999 Spearman | 0.326 | 0.323 | 0.224 | 0.446 |
| holdout gate macro F1 | 0.655 | 0.483 | 0.444 | 0.470 |
| holdout in-domain -> primary | 0.765 | 0.824 | 0.824 | 0.706 |
| holdout ambiguous pass | 0.667 | 0.667 | 0.667 | 0.333 |
| holdout gate leak (0 dmg) | 0 | 0 | 1 | 0 |
| holdout damaging leak | 0 | 3 | 3 | 1 |
| holdout damage in-band | 0.607 | 0.500 | 0.429 | 0.607 |
| holdout unknown-word guesses | 5 | 1 | 0 | 0 |
| calibration gate macro F1 (CV) | 0.756 | 0.673 | 0.688 | 0.663 |
| calibration damage in-band (CV) | 0.640 | 0.560 | 0.600 | 0.600 |
| pacing met (calibration battles) | 6/6 | 6/6 | 5/6 | 6/6 |

**ours wrong, MiniLM right** (6)

| Target | Guess | Human Label | ours: gate | MiniLM: gate | likely causes |
|---|---|---|---|---|---|
| brand | logo | in_domain | out_of_domain | in_domain | blocked: a rival domain's anchors are closer (corpus topic bias / sense mixing) |
| brand | packaging | in_domain | in_domain | in_domain | gate right, damage band wrong (relevance scale / ceiling of this target) |
| email | chat app | in_domain | in_domain | in_domain | phrase = average of word vectors (no phrase vector, short context); gate right, damage band wrong (relevance scale / ceiling of this target) |
| email | seashell | out_of_domain | invalid | out_of_domain | guess OOV (no vector); rare in corpus (min word freq 8) -> data sparsity |
| comet | solar system | in_domain | in_domain | in_domain | gate right, damage band wrong (relevance scale / ceiling of this target) |
| comet | doorknob | out_of_domain | invalid | out_of_domain | guess OOV (no vector); rare in corpus (min word freq 13) -> data sparsity |

**MiniLM wrong, ours right** (3)

| Target | Guess | Human Label | ours: gate | MiniLM: gate | likely causes |
|---|---|---|---|---|---|
| email | spam filter | in_domain | in_domain | out_of_domain | phrase = average of word vectors (no phrase vector, short context) |
| email | mail server | in_domain | in_domain | out_of_domain | phrase = average of word vectors (no phrase vector, short context) |
| comet | light year | in_domain | in_domain | out_of_domain | phrase = average of word vectors (no phrase vector, short context) |

**both wrong** (8)

| Target | Guess | Human Label | ours: gate | MiniLM: gate | likely causes |
|---|---|---|---|---|---|
| brand | fashion label | ambiguous | ambiguous | out_of_domain | phrase = average of word vectors (no phrase vector, short context) |
| email | inbox | in_domain | invalid | out_of_domain | guess OOV (no vector); rare in corpus (min word freq 4) -> data sparsity |
| email | newsletter | in_domain | out_of_domain | out_of_domain | blocked: a rival domain's anchors are closer (corpus topic bias / sense mixing) |
| comet | space shuttle | ambiguous | in_domain | in_domain | gate right, damage band wrong (relevance scale / ceiling of this target) |
| comet | horoscope | out_of_domain | invalid | in_domain | guess OOV (no vector); rare in corpus (min word freq 10) -> data sparsity |
| cartoon | caricature | in_domain | invalid | in_domain | guess OOV (no vector); rare in corpus (min word freq 38) -> data sparsity |
| cartoon | puppet show | in_domain | in_domain | in_domain | phrase = average of word vectors (no phrase vector, short context); gate right, damage band wrong (relevance scale / ceiling of this target) |
| cartoon | television channel | ambiguous | out_of_domain | out_of_domain | — |

## 6. เกณฑ์ผ่าน

| criterion | pass |
|---|---|
| Phase 2: coverage ≥ 95% | True |
| Phase 5: adapter unit tests | True |
| Phase 6: pacing ทุกด่าน + leak ≤ 5% (active model) | True |
| Holdout: Holdout gate macro F1 >= 0.75 | False |
| Holdout: Holdout damaging leak = 0 | True |
| Holdout: Holdout in-domain -> primary >= 80% | False |
| Holdout: Holdout damage in-band >= 70% | False |
| Holdout: Pacing goals met (Section 6 battles) | True |
| Holdout: Pacing goals met (LOTO-CV) | False |
| Holdout: No tuned parameter on a grid edge | True |

> ตัวเลขทั้งหมดรายงานตามจริงจากรอบนี้ ไม่มีการแก้ข้อมูลทดสอบ · MiniLM เป็น reference baseline เท่านั้น