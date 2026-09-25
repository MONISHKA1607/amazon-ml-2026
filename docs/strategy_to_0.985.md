# Strategy → macro-F₀.₅ = 0.985 (Amazon ML Challenge 2026)

Target to make the final 50: **≥ 0.985**. This doc is grounded in the actual
problem statement (`Guidelines/…problem_statement.pdf`), our EDA
([findingsfromeda.md](findingsfromeda.md)), and the SOTA/winner evidence in
[research_synthesis.md](research_synthesis.md) (which validates this whole
pipeline and sharpened the four knobs marked **[R]** below).

## The bar, decoded

Metric = **macro-F₀.₅ per Source-1 entity**, F₀.₅ = 1.25·P·R / (0.25·P + R),
singletons included (correct-empty = 1.0, any false merge on them = 0.0).

With a **perfect matcher**, per-entity F₀.₅ ≈ 1.25·R/(0.25+R):

| blocking recall R | perfect-matcher ceiling |
|---|---|
| 0.81 (today) | 0.81 – 0.955 (measured macro 0.923) |
| 0.90 | ~0.978 |
| **0.98** | **~0.996** (→ 0.985 macro with P≈0.99) |

**0.985 is unreachable until blocking recall ≈ 0.98.** The rules say it
outright: *"Invest in a strong blocking strategy — it determines the upper
bound of your recall."* Recall 0.81 → ~0.98 is the whole game.

## What the target changes vs the first plan

- **Blocking recall is now the ONLY first-order lever.** Model choice (all 6
  within 0.5 pp) and calibration are second-order — set once, don't tune.
- **Embeddings + transliteration move from V2 → V1 (essential).** They're the
  retrieval tools that close the recall hole, not nice-to-haves. Compliant:
  LaBSE is Apache-2.0 / 0.47B; transliteration is an offline rule-based
  transform. Neither is an "external identity lookup" — those (geocoding APIs,
  registries) stay banned (instant DQ).
- **Candidate count (436/entity) is NOT a score lever.** `candidate_pairs.tsv`
  is not leaderboard-scored — only audited for reduction ratio in the package
  review. So blocking = **recall-max**; precision is enforced at the matcher,
  not by shrinking candidates. Prune only for the audit, never at recall's cost.

## P1 — Recall attack (priority): 0.81 → ≥0.98

1. **Normalize both sides first** (cheap, biggest gain — attacks the 38.7%
   cross-script + `.com`-handle misses): romanize non-Latin (`indic-transliteration`,
   offline), accent-fold (NFKD), split camelCase + strip domain suffixes,
   canonicalize legal suffixes (Pvt/Private, Ltd/Limited) and address abbrevs
   (Rd/Road).
2. **Multi-key UNION blocking:** char n-gram (3–5) + token + sorted-token +
   numeric (phone/PIN) keys. Union recall beats any single key.
3. **Semantic retrieval for the residual [R]:** multilingual embedding + ANN for
   pairs no lexical key shares (different words, same business). Retrieve **top-k ≥
   10–20 in both directions** (top-k is asymmetric; 89% multi-match makes top-1 fatal).
   Union it *in* — never use it as the sole blocker.
4. **Purge oversized generic-token blocks [R]** (`inc`/`llc`/`restaurant`/`pvt`) —
   ~10× precision for minor recall cost, and it keeps the union tractable at 10M.
5. Measure the recall-vs-cost frontier at each step; stop when recall ≥ 0.98.

## P2 — Precision attack: hold P ≈ 0.99

1. **Disambiguation features:** the sim=100 hard negatives (different businesses,
   identical names — `Vega Zeo` vs `Vega`) can ONLY be split by address / phone /
   PIN / geo-token signal. Add strong address + numeric exact-match features and
   name×address interactions.
2. **Hard-negative mining:** retrain on those sim=100 near-duplicate
   different-entity pairs so the model learns to reject them.
3. Single **calibrated** GBDT (xgboost/lightgbm), isotonic calibration on the
   **natural** distribution (never SMOTE before calibrating — it breaks it). Clear
   each candidate independently to emit a **variable-size set**, using **one global
   threshold set above 0.5**, swept on macro-F₀.₅ — **[R] not a per-entity threshold**
   (overfits at 3.67 cands/entity).
4. **Singleton guard:** predict empty unless a candidate clears a HIGH bar — a
   false merge on a singleton is a flat 0.0 (5.6% of all entities).
5. **[R] No unconstrained connected-components / transitive closure in v1** — one
   weak edge drags a whole cluster into a false merge and tanks precision. Add only
   *constrained* clustering (high-conf edges + mutual top-k + size cap) later, gated
   on validation F₀.₅.

## Honest risks

- **Embeddings are now feasible on this box (the #1 blocker is resolved).** It has
  an **RTX 3050 Laptop GPU (4 GB VRAM) + 16 GB RAM**, and the toolchain is installed:
  faiss-cpu 1.15.1, sentence-transformers 6.1.0, torch 2.6+cu124 (CUDA live). GPU
  encoding of ~10M records ≈ **40 min**. The 4 GB VRAM constrains only the *encoder*
  (fp16 + modest batch); the index always lives in CPU RAM. No cloud needed.
- **[R] Index trap — plain IVF-PQ caps at ~0.94 recall** (Karapiperis 2025), fatal at
  this bar. Fix: use **LEALLA-large @ 256-dim** so a full-precision **HNSW** index
  over 10M fits RAM raw (0.996 recall), *or* keep a higher-dim encoder and add an
  **IVF-PQ + refine** step (re-rank top-100 on full vectors → ~0.98). LEALLA-large
  (Apache, 147M) is the pick over MiniLM/LaBSE — near-LaBSE cross-lingual alignment,
  ⅓ the size, and the only one whose raw index fits (see research_synthesis.md §3–4).
- **Precision floor from genuine ambiguity:** some sim=100 pairs may be
  unresolvable with the given signal → a hard cap on those entities. Size unknown
  until measured.
- Our 0.923 is optimistic (small negative pool); the real leaderboard starts
  lower, so the true climb is bigger than 0.923 → 0.985 looks.

## Compliance (locked to the rules)

- Final model MIT/Apache-2.0, ≤8B: xgboost (Apache), lightgbm (MIT), LaBSE
  (Apache) all pass.
- Two output files in `output/`; matches ⊆ candidates; every test S1 = one row;
  S2-/S3- IDs only; run `validate_submission.py` before every upload.
- 15 uploads total (5/day × 3 days) — spend them on **measured** gains, not guesses.

## Build order

1. **Ceiling diagnostic + normalization/transliteration blocking** (this box) →
   measure recall gain. **← START HERE; infra-independent.**
2. **Embedding infra [R]:** encode on GPU (fp16, **LEALLA-large**, 256-dim), build a
   full-precision **HNSW** index in system RAM (fits raw at 256-dim → 0.996 recall).
   Fallback if we swap to a higher-dim encoder: **IVF-PQ + refine**, never plain IVF-PQ.
3. Semantic retrieval → recall ≥ 0.98.
4. Precision matcher (disambiguation features + hard-neg mining + calibrated threshold).
5. Full-test run (streaming/shard), validator, package.

---
*Bottom line: 0.985 is attainable but only via blocking recall ≈ 0.98 with
precision ≈ 0.99. Everything else is noise at this bar.*
