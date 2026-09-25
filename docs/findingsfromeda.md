# Findings from EDA — Business Entity Resolution (Amazon ML Challenge 2026)

Source: `notebooks/01_thorough_EDA.ipynb` · Day 1 (2026-09-25)
Task: for each **Source-1** entity, find all matching **S2/S3** records (0, 1, or many).
Metric: **macro F₀.₅ per S1 entity** — precision weighted 2× over recall; a correct
singleton scores 1.0, any false merge on a singleton scores 0.0.

> **[full]** = measured on the real dataset (direct counts).
> **[sample]** = a leakage-free validation subsample (3,627 train + 1,210 val entities,
> small negative pool) — directionally right but **optimistic** vs. the real leaderboard.

## Dataset at a glance [full]

| file | records |
|---|---|
| train Source-1 (reference) | **2,206,821 entities** |
| train Source-2 | 5,034,616 |
| train Source-3 | 5,285,603 |
| test Source-1 | **1,732,544 entities** |
| test Source-2 | 4,887,273 |
| test Source-3 | 5,082,316 |

Match distribution over the 2.21M train S1 entities:
- **Two or more matches: 89.0%** (1,964,417)
- Exactly one match: 5.4% (119,157)
- **Singletons (no match): 5.6%** (123,247)
- Avg matches per non-singleton entity: **3.67** (7.64M matched IDs total)

## Headline findings

**1. This is a multi-match problem, not a singleton problem.** 89% of S1 entities match
≥2 records. The decision layer must let **each candidate clear the threshold
independently** — argmax / one-best logic would cap us at ~11% of entities. Singleton
handling is a minor lever (5.6%); multi-match precision is the main one.

**2. Blocking is the bottleneck, not the matcher.** [sample] Best of 6 tuned models =
**xgboost 0.9229**, but **blocking recall is only 0.810** — that is the real cap on
quality. The model scores *above* 0.810 only because F₀.₅ is precision-weighted and
forgives the ~19% of matches blocking never surfaces (pair-level P/R ≈ 0.99 → the matcher
is near-perfect on what it sees). **→ Recall gained in blocking (P1) is worth far more
than more model tuning (P2).**

**3. Cross-script / cross-encoding recall hole — the #1 recall lever.** [sample] Of the
3,289 missed true matches, **38.7% are non-Latin** vs only **0.5% of retrieved** ones — the
recall hole is overwhelmingly cross-script. It's broader than Devanagari: it also covers
accented Latin (`Édge`, `góldenwestin`) and website-handle forms (`arielindia.com`,
`VaexwinvestCom`, `Kalyangrocery.Com`). Token / char-trigram keys can't be shared across
scripts. The **only** fixes are transliteration (romanize both sides) or multilingual
embeddings for retrieval — phonetic keys (Soundex/Metaphone) are ASCII-only and do **not**
help here.

**4. Name features lead — but address *overlap* is not dead (only the address flags are).**
Top single-feature ROC-AUCs: `name_tfidf_cos` 0.983, `name_token_set` 0.979, `name_tok_jacc`
0.977, `name_ng_jacc` 0.971 — yet `addr_tok_jacc` (0.969) is the **5th-best** feature and
`addr_token_set`/`addr_ratio` (~0.94) are strong too. Only the address *presence flags*
(`addr_both_empty`/`addr_one_empty`, AUC≈0.50) carry no signal. `country_match` (0.587) helps
weakly in-sample but is a **generalization trap** — test adds France, unseen in train; never
use country as a hard gate / filter / one-hot.

**5. Precision-heavy metric + multi-match majority ⇒ per-candidate thresholding, tuned
conservatively.** F₀.₅ punishes false merges 2×. Score each candidate independently and
keep those above a swept threshold — do **not** "predict empty when the top-1/top-2 margin
is small": with 89% multi-match, a small margin usually means *two real matches*, not
ambiguity. Reserve empty-prediction for entities where no candidate clears the bar.

## Feature set: 20 ranked → lean 18 [sample]
- **Dropped, redundant** (|r|>0.95, kept the higher-AUC twin): `name_qratio` (r=1.0 with
  `name_ratio`), `addr_token_sort` (r=0.98 with `addr_token_set`). That's the 20 → 18.
- **Flagged dead but still in the 18** (AUC≈0.50): `addr_both_empty`, `addr_one_empty` —
  the notebook lists them as zero-signal but the lean-18 only drops the redundant twins.
  Harmless to a GBDT, but drop them too → 16 for a cleaner set.
- **Kept, top signal:** `name_tfidf_cos`, `name_token_set`, `name_tok_jacc`, `name_ng_jacc`,
  `addr_tok_jacc`, + remaining name/address similarity; `country_match` flagged (#4).
- **Gap:** every feature is surface-form string overlap → blind to same-business-different
  -script. A multilingual name-embedding cosine (e.g. LaBSE, Apache-2.0, 0.47B params) closes
  it — but per #3 it pays off most as a *blocking/retrieval* signal (a matching feature never
  fires on a pair blocking didn't generate). Now feasible on this box (RTX 3050 GPU +
  toolchain installed) → promoted to **V1 for retrieval**; see
  [strategy_to_0.985.md](strategy_to_0.985.md).

## Model bake-off — 6 models, per-entity macro-F₀.₅ [sample]
train 113,465 pairs / 3,627 entities · val 36,535 pairs / 1,210 entities · pos_weight 9.8 · **blocking recall 0.810** · **436 candidates/entity** (⚠ ~120× over-generation vs 3.67 true matches — big reduction-ratio lever)

| model | val macro-F₀.₅ | thr | pair P | pair R | fit s | license |
|---|---|---|---|---|---|---|
| **xgboost** | **0.9229** | 0.50 | 0.997 | 0.992 | 40 | Apache-2.0 ✓ |
| lightgbm | 0.9219 | 0.10 | 0.996 | 0.993 | 42 | MIT ✓ |
| catboost | 0.9212 | 0.80 | 0.997 | 0.989 | 150 | Apache-2.0 ✓ |
| rand_forest | 0.9204 | 0.45 | 0.996 | 0.988 | 443 | BSD-3 |
| logreg | 0.9195 | 0.90 | 0.996 | 0.989 | 4.5 | BSD-3 |
| hist_gb | 0.9181 | 0.95 | 0.998 | 0.981 | 36 | BSD-3 |

- All six within **0.5 pp** (0.918–0.923) → **model choice is not the lever; blocking recall is.**
- The threshold spread (0.10 → 0.95) is a **calibration** symptom — calibrate (isotonic/Platt) before thresholding so the decision policy is stable.
- **Final model = xgboost (best, Apache-2.0) or lightgbm (MIT, ~tied)** — both satisfy MIT/Apache-2.0 + ≤8B; the three sklearn models are BSD-3 (not the best here anyway).
- Best (xgboost) params: `learning_rate 0.1, max_depth 8, n_estimators 400, subsample 0.8`.
- ⚠ Subsample, small negative pool → optimistic. The full ~10M-record haystack has far more look-alike negatives; the real leaderboard will be lower. **Blocking recall (0.810) is the trustworthy number.**

## Guidelines compliance — verified against all three docs
- **Two output files** in `output/`: `matching_results.tsv` (the only file scored on the
  leaderboard) + `candidate_pairs.tsv` (the post-filter set fed to the model; **matches ⊆
  candidates**).
- Every test S1 entity = exactly one row; empty = singleton; **S2-/S3- IDs only**; no dup
  rows; no repeated ID in a list. Hard reject: S1 self-match, bad prefix, missing entity,
  dup. A well-formed but nonexistent ID only *lowers score* (per the validator).
- **Smaller `candidate_pairs` = better** (reduction ratio; audited for top teams) →
  blocking is two-objective: recall **and** candidates/entity.
- No external lookups (APIs / geocoding / registries) = DQ risk. Country is an **open set**.
- Final model **MIT/Apache-2.0, ≤8B params.** Validator:
  `student_resource/utils/validate_submission.py` — run before every submit; `--check-ids`
  may OOM on this box (~2 GB free), so use the default light mode.

## Next steps per role (team plan)
- **P1 — Blocking (priority):** transliteration/romanization pass to close the cross-script
  hole (biggest recall lever); union with token blocks; rank within block by cheap
  similarity, keep **top-K per entity**. Current fan-out is **436 candidates/entity** for ~3.67
  true matches — ~120× over-generation, so top-K pruning is a large reduction-ratio win *and*
  lifts precision. Report the recall-vs-candidates/entity frontier, pick the knee. Freeze
  versioned `candidate_pairs_vN.tsv`.
- **P2 — Matching:** single **calibrated** GBDT (xgboost/lightgbm) on the 18 features;
  per-candidate threshold swept on per-entity macro-F₀.₅; independent-clear multi-match;
  conservative singleton rule. Embeddings / cascade / ensemble = V2+, not the first submission.
- **P3 — Error analysis:** expect `FN_transliteration` + `FN_blocking_failure` to dominate;
  watch chain / short-name FPs on the multi-match majority. Add a **non-Latin subgroup** slice.
- **P4 — Ops:** wire the validator into every run; fill `Documentation_template.md` §2–§5
  from this notebook; keep EXPERIMENT_LOG / SUBMISSION_LOG.

---
*Validation is leakage-free: GroupKFold by S1 entity, scored on true per-entity macro-F₀.₅
with blocking-missed positives forced to misses. A leave-one-country-out run (train without
India, validate on it) is the cheapest proxy for unseen-country (France) behaviour.*
