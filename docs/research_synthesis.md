# Research Synthesis — Entity Resolution: SOTA, Winning Solutions & Score-Maximization

Source: four parallel literature/practitioner sweeps (blocking · matching · cross-script · competitions), Day 1 (2026-09-25).
Companion to [findingsfromeda.md](findingsfromeda.md) and [strategy_to_0.985.md](strategy_to_0.985.md).

Purpose: ground every pipeline choice in **published evidence** + how people actually won the
structurally-identical Kaggle *Foursquare Location Matching* competition (predict a variable-size
match-set per record — the same shape as our per-S1-entity task).

> Every number below is from a cited paper/writeup (see §9). Our own measured numbers
> (blocking recall 0.81, 89% multi-match, 38.7% non-Latin misses) live in findingsfromeda.md.

## TL;DR — the research validates our direction and sharpens the knobs

SOTA and the Foursquare winners converge on the pipeline our EDA already pointed to:
**canonicalize/transliterate → multi-key union blocking + embedding-ANN for the residual → GBDT
pairwise scorer → isotonic-calibrated *single* threshold tuned for F₀.₅ → predict a variable-size
match-set.** Direction unchanged; four knobs sharpened (§7). The one trap caught: **plain FAISS
IVF-PQ caps at ~0.94 recall** — fatal for a 0.98 target unless we add a refine step or a low-dim index.

## 1. Blocking — the recall lever (0.81 → 0.98)

Recall is won by a **union of many complementary keys**, not one clever key. Every independent
source says this (Sparkly, Splink, RecordLinkage, the Papadakis benchmark, the Foursquare winner).

### 1.1 Method landscape (recall as reported in the cited papers)

| Method | Best reported recall | ~10M offline? | Verdict for us |
|---|---|---|---|
| **Sparkly** (BM25 top-k, Lucene) | **98.7–100% @ k=50** (15 datasets) | Yes (CPU/RAM) | **lexical backbone** |
| Token/standard + meta-blocking | hits γ=0.9 on all (Papadakis) | Yes (slow @2M) | union engine + purge |
| Q-gram / char-n-gram keys | raises recall on dirty/typo data | Yes | typo/OCR robustness |
| Sorted-neighborhood (multi-pass) | recall via multi-pass union | Yes (linear) | cheap complementary key |
| **SC-Block** (sup. contrastive + FAISS) | **99.5% @ tuned k** (Abt-Buy) | Tight (needs labels) | residual booster |
| UniBlocker (dense, no fine-tune) | 89.9 mean PC; **+Sparkly ens. 91.7** | Tight | residual booster |
| DeepBlocker | top on 6/16 hard sets | **No** (OOM >100K) | skip at our scale |
| LSH (MinHash/HP/CP) | high recall only w/ huge cand sets | **No** (MinHash OOM) | **avoid** |
| Canopy clustering | depends on loose threshold | Yes | subsumed by top-k |

### 1.2 The recall recipe — a layered union (this is the plan)

1. **Canonicalize *before* blocking.** NFKD accent-fold, transliterate Devanagari→Latin
   (`indic-transliteration` / ICU), handle-normalize (strip TLD, split camelCase, drop `@`).
   This converts most of our 38.7% cross-script miss into ordinary lexical matches **for free** —
   the single highest-leverage change, and it costs no model.
2. **Multi-key lexical union, top-k per record.** Sparkly-style BM25 top-k (98.7–100% @ k=50,
   linear output size) is the backbone, unioned with q-gram, phonetic, token-set and
   sorted-token keys. *"A long list of strict keys beats a few loose ones"* (Splink).
3. **Embedding + ANN on the residual only**, unioned in — never as the sole blocker
   (UniBlocker+Sparkly ensemble > either alone; SC-Block hits 99.5% at tuned k). Retrieve
   **top-k ≥ 10–20 in *both* directions**: top-k is asymmetric, and our 89% multi-match rate
   makes any top-1 scheme fatal.
4. **Meta-block / purge** oversized generic-token blocks (`inc`, `llc`, `restaurant`, `pvt`) —
   up to ~10× precision for minor recall cost, and it keeps the union tractable at 10M.

### 1.3 FAISS index choice — the hard evidence (Karapiperis et al. 2025, ER-specific benchmark)

| Index | Recall | Fits 10M @ 16 GB RAM? |
|---|---|---|
| IVF-PQ (plain) | **~0.94 — a hard cap** | trivially (~160 MB) |
| **IVF-PQ + refine** (re-rank top-100 w/ full vectors) | **→ ~0.98** | yes ✓ |
| ScaNN | ~0.95 | yes (if installable) |
| HNSW | **0.996** | 384-dim → ~18 GB ✗ · **≤256-dim → fits ✓** |

**→ plain IVF-PQ silently tops out at ~0.94** — the one real trap for a 0.98 target. Two exits:
(a) **LEALLA-large @ 256-dim** → the index fits RAM *raw*, so use HNSW (0.996); or
(b) keep a higher-dim encoder but add the **refine step** (IVF-PQ shortlist → re-rank top-100 on
full vectors). Our 4 GB VRAM constrains only the *encoder*; the index always lives in CPU RAM.
Also: embedding recall **decays with corpus size** (0.962→0.800 as 10K→2M in the benchmark) —
so tune k against a held-out recall target, don't fix it blind (the SC-Block recipe).

## 2. Matching — the precision lever (hold P ≈ 0.99 to protect F₀.₅)

A calibrated GBDT is already ~0.99 pair-level on structured EM; the F₀.₅ points (β=0.5 weights
precision 2×) are won at the **decision layer**, not by a fancier model. Ranked by confidence × effort:

1. **One global threshold, tuned on F₀.₅, set *above* 0.5.** For β=0.5 the Bayes-optimal cutoff sits
   *higher* than the F1 point. Sweep a **single** threshold directly on macro-F₀.₅ — **not** a
   per-entity threshold (overfits badly at our 3.67 candidates/entity).
2. **Hard-negative mining** on same-name / different-business pairs (high name-sim + conflicting
   address/country/postal). These are exactly the false positives F₀.₅ punishes hardest.
3. **Isotonic (or Platt) calibration** — the *enabler* for a trustworthy threshold. Won't raise F1
   on its own; calibrate *after* training, on the **natural** class distribution (SMOTE/oversampling
   wrecks tree calibration — never resample before calibrating).
4. **Name×address interaction features + a country/postal-mismatch gate.** Cheap, high-signal.
   **Pre-score filtering** on hard conflicts (different country, geo-impossible distance) is free precision.
5. **Predict a variable-size set** — keep *every* candidate ≥ threshold; never collapse to top-1.
   This is exactly how the Foursquare winner emitted match-sets.
6. **No unconstrained connected-components / transitive closure for v1.** Transitivity drags a whole
   cluster into a false merge through one weak edge — catastrophic under precision-weighted scoring.
   Add only *constrained* clustering later (high-confidence edges + mutual top-k + a size cap), and
   only if it lifts validation F₀.₅.
7. *V2 option:* a **cross-architecture** ensemble — add a Ditto/DistilBERT matcher (Apache, 66M, fits
   4 GB) for the textual/dirty cases GBDT is weakest on. Must be cross-arch; same-backbone ensembles
   don't help. Ship the GBDT alone first.

### 2.1 Matcher-model context (reported F1 on standard EM benchmarks)

| Model | Amazon-Google | Abt-Buy | Note |
|---|---|---|---|
| Magellan (classical / GBDT) | ~49–71 | ~43–63 | **wins structured EM (our case)** |
| DeepMatcher | ~69 | ~63 | superseded |
| Ditto (PLM, DistilBERT) | ~75 | ~89 | wins *textual/dirty*; V2 candidate |
| R-SupCon (contrastive) | high | **93.7** | source-aware sampling is the lever |

Takeaways: classical GBDT wins **structured** EM (ours); PLMs win **dirty/textual**; LLMs beat
neither on clean structured EM and are disqualified anyway (online / license / >8B). R-SupCon's
single biggest lever was **source-aware negative sampling** — removing it collapsed Abt-Buy
93.7 → 38.2. Relevant only if we ever train a contrastive encoder ourselves.

## 3. Cross-script / multilingual — closing our #1 recall hole (38.7% of misses are non-Latin)

Our biggest measured leak is non-Latin (Devanagari, accented Latin, `.com`-handle forms). The
research says most of it is **not** an embedding problem — it's a normalization problem:

1. **Romanize both sides, then reuse the lexical union.** Transliterate to a common script (ICU /
   `indic-transliteration` / `unidecode` for the long tail), NFKD accent-fold, casefold. After this,
   "मुंबई" and "Mumbai" collide on ordinary BM25/q-gram keys — no model needed. This is the free win.
2. **Embed only the residual** with a multilingual encoder whose *training* covered the scripts we
   see. Cross-lingual alignment quality (Tatoeba/BUCC) predicts blocking recall far better than raw
   MTEB rank — pick the encoder on alignment, not on English retrieval score.
3. **Guard against script-confusable false merges** (transliteration collisions across languages) at
   the matcher, via the country/language gate in §2.4.

### 3.1 Encoder alignment (higher = better cross-lingual retrieval; drives residual recall)

| Encoder | Params | Dim | Tatoeba-ish alignment | License | Fit @10M/16 GB |
|---|---|---|---|---|---|
| **LEALLA-large** | 147M | **256** | ~83.5 | Apache-2.0 | **index fits RAM raw ✓** |
| LaBSE | 471M | 768 | ~83.7 (best) | Apache-2.0 | needs PQ/refine |
| multilingual-e5-small | 118M | 384 | strong | MIT | needs ≤256-dim or PQ |
| MiniLM-multilingual (paraphrase) | ~118M | 384 | good | Apache-2.0 | needs PQ/refine |

**Pick: LEALLA-large.** Near-LaBSE alignment at ⅓ the size and **256-dim**, so a full-precision
HNSW index over 10M fits our 16 GB box (0.996 recall) — it sidesteps the IVF-PQ 0.94 cap entirely.
LaBSE is the quality ceiling if we accept PQ+refine; e5-small the light alternative.

## 4. Model & license shortlist (offline, ≤8B, permissive license — all verified)

| Role | Pick | License | Why |
|---|---|---|---|
| Pairwise scorer | **LightGBM** / XGBoost | MIT / Apache-2.0 | classical wins structured EM; fast, calibratable |
| Blocking encoder | **LEALLA-large** (147M, 256-dim) | Apache-2.0 | alignment + fits RAM raw (see §3.1) |
| — quality alt | LaBSE (471M, 768-dim) | Apache-2.0 | best alignment; needs PQ+refine |
| — light alt | multilingual-e5-small (118M) | MIT | smallest viable |
| V2 matcher (opt) | Ditto / DistilBERT (66M) | Apache-2.0 | textual/dirty cases; cross-arch ensemble |
| Lexical blocker | Sparkly (`sparkly-em`) | BSD/Apache | BM25 top-k backbone |

**Disqualified:** Jellyfish (CC-BY-NC, 7–13B), jina-embeddings-v3 (CC-BY-NC), any online/hosted LLM
(offline rule), 14B distills (>8B + hardware). License and offline-capability were checked, not assumed.

## 5. What the winners & production systems actually did

- **Foursquare Location Matching (Kaggle, 1st place)** — the structural twin of our task
  (variable-size match-set per record). Recipe: heavy string normalization → **candidate generation
  by top-k neighbors (both lexical and embedding), never top-1** → GBDT pairwise classifier on
  hand-built similarity features → threshold → emit the set of everything above it. Top teams
  explicitly warned that **transitive-closure post-processing over-merged and hurt** unless tightly
  constrained — independent confirmation of §2.6.
- **Splink** (UK MoJ, production record-linkage at 100M+): union of many strict blocking rules;
  Fellegi-Sunter weights; single tunable threshold. Validates §1 and §2.1.
- **Zingg / dedupe.io** (production ER): blocking → pairwise classifier → clustering, with clustering
  kept conservative. Same three-stage shape.
- **Papadakis blocking benchmark**: meta-blocking + block-purging is what makes a token-union
  tractable and precise at scale (§1.2 step 4).

## 6. Dead-ends to avoid (evidence-backed)

- **Plain FAISS IVF-PQ** as the final index — 0.94 recall cap (§1.3). Add refine or go low-dim.
- **LSH / MinHash** — dominated on the recall/candidate trade-off and OOMs at our scale.
- **DeepBlocker** — best-in-class quality but OOMs above ~100K; not viable at 10M.
- **Unconstrained connected-components / transitive closure** — over-merges, tanks precision (§2.6).
- **Per-entity thresholds** — overfit at 3.67 cands/entity; use one global threshold (§2.1).
- **Resampling (SMOTE) before calibration** — breaks tree probability calibration (§2.3).
- **Jellyfish / jina-v3 / online LLMs / >8B distills** — license or offline/hardware disqualified (§4).

## 7. What this changes in strategy_to_0.985.md (the four sharpened knobs)

1. **Embedding index:** was "faiss-cpu IVF-PQ, settled." → **LEALLA-large @ 256-dim + HNSW** (fits RAM
   raw, 0.996), *or* IVF-PQ **+ refine** if we keep a higher-dim encoder. Plain IVF-PQ = 0.94 cap.
2. **Transliteration is a blocking (recall) step, not a matcher nicety** — romanize *before* union
   blocking; it reclaims most of the 38.7% cross-script hole for free.
3. **Decision layer:** make explicit it's a **single global** F₀.₅-tuned threshold set **above 0.5**,
   emitting a variable-size set — not any per-entity threshold.
4. **Add a precision guard:** **no unconstrained connected-components in v1**; only constrained
   clustering later, gated on validation F₀.₅.

## 8. First-submission recipe (no new training — evidence-backed baseline)

The literature supports a strong first submission built only from tools we already have:

1. **Normalize + transliterate** every record (NFKD fold + romanize + handle-normalize) — §1.2.1.
2. **Union blocking**: exact-name ∪ token-set ∪ q-gram ∪ sorted-token keys, top-k both directions,
   purge generic-token blocks. Measure blocking recall on the held-out entities — this is the ceiling.
3. **RapidFuzz + TF-IDF pairwise features** → **LightGBM** (already in `features.py`/`model.py`).
4. **Isotonic-calibrate**, then sweep **one global threshold** on macro-F₀.₅ (above 0.5).
5. **Emit the variable-size set** (all candidates ≥ threshold); apply the country/postal gate.
6. Validate with `utils/validate_submission.py` before upload.

Embeddings (LEALLA-large residual) and a Ditto V2 matcher are **deferred** — add only after this
classical baseline's blocking recall and F₀.₅ are measured, so each addition is attributable.

## 9. Sources

Blocking:
- Paulsen et al., **Sparkly: Blocking for ER using Top-k similarity search** (VLDB 2023).
- **UniBlocker** — unsupervised dense blocking; +Sparkly ensemble (2023/24).
- Brunner & Stockinger / Thirumuruganathan, **DeepBlocker** (VLDB 2020).
- Zeakis et al. / **SC-Block** — supervised contrastive blocking (2023/24).
- Papadakis et al., **blocking & meta-blocking benchmark / survey** (VLDBJ, ACM CSUR).
- Karapiperis et al., **ANN-index benchmark for ER** (2025) — the IVF-PQ 0.94 / HNSW 0.996 / refine numbers.

Matching & decision:
- Konda et al., **Magellan** (VLDB 2016); Mudgal et al., **DeepMatcher** (SIGMOD 2018).
- Li et al., **Ditto** (VLDB 2021); **R-SupCon** — contrastive EM w/ source-aware sampling.
- Fbeta / threshold-optimality: F₀.₅ Bayes-threshold-above-F1 result; **isotonic/Platt calibration** (Niculescu-Mizil & Caruana).

Cross-lingual encoders:
- Feng et al., **LaBSE** (ACL 2022); **LEALLA** (lightweight LaBSE, EACL 2023); **multilingual-e5**.

Systems / competitions:
- **Foursquare Location Matching** — Kaggle 1st-place & top solution writeups.
- **Splink** (UK MoJ), **Zingg**, **dedupe.io** — production ER architecture docs.





