# amazon-ml-2026 — Business Entity Resolution

Working, tested pipeline for the Amazon ML Challenge 2026 (Business Entity
Resolution: match Source 2 / Source 3 records to Source 1 entities).

Every module below has been run end-to-end against a synthetic dataset
matching the real schema (`entity_id, business_name, business_address,
country`, plus `train_ground_truth.tsv` with `source1_entity_id,
matched_entity_ids`) — including a France-only test entity, to confirm the
pipeline is genuinely country-agnostic. **Drop in the real dataset and it
runs the same way.**

## Setup

```bash
pip install -r requirements.txt
```

Place the real challenge data at:
```
dataset/train/{train_source1,train_source2,train_source3,train_ground_truth}.tsv
dataset/test/{test_source1,test_source2,test_source3}.tsv
```

## Pipeline (in order)

```bash
# 1. EDA — always run this first on the real data
python scripts/eda.py --data-dir dataset --split train

# 2. Candidate generation (Person 1's track). Versioned, frozen artifacts.
python scripts/make_candidates.py --data-dir dataset --split train --version v1
# prints blocking recall — the ceiling on everything downstream.
# Re-run with a subset of blocks to test one at a time, e.g.:
python scripts/make_candidates.py --split train --version v1_exact --blocks exact_name,name_tokens

# 3. Train the matcher (Person 2's track) against a frozen candidate version.
python scripts/train_matcher.py --data-dir dataset --candidate-version v1 --model-version v1
# splits by S1 entity (leakage-free), trains LightGBM, sweeps thresholds on
# macro F0.5, saves models/matcher_v1.pkl (model + TF-IDF vectorizers + threshold)

# 4. Generate the test-set candidates + build a full submission.
python scripts/create_submission.py --data-dir dataset \
    --candidate-version v1 --model-version v1 --submission-id 01

# 5. Validate before uploading — this IS the official challenge validator
#    (confirmed working against 7 pass/fail scenarios, see below).
python utils/validate_submission.py \
    --matching outputs/submissions/submission_01/matching_results.tsv \
    --candidate outputs/submissions/submission_01/candidate_pairs.tsv \
    --test-dir dataset/test
# Add --check-ids for the full ID-existence cross-check (slower/more memory,
# off by default — a nonexistent ID only lowers your score, it doesn't reject).

# Optional, stricter dev-time check (catches a matched ID that never appeared
# as a candidate as a hard FAIL, where the official validator only warns —
# useful while iterating, since it usually means a real pipeline bug):
python scripts/validate_submission.py \
    --matching outputs/submissions/submission_01/matching_results.tsv \
    --candidate outputs/submissions/submission_01/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Note the one behavioral difference between the two validators:** the
official `utils/validate_submission.py` treats a matched ID that's absent
from `candidate_pairs.tsv` as a warning, not a failure — it will still let
you submit. `scripts/validate_submission.py` treats the same thing as a
hard error, on the theory that it almost always signals a real bug in your
pipeline (the final matcher should only ever pick from its own candidates).
Use the official one as the actual gate; use the stricter one while
debugging.

## Module map

| File | Owns |
|---|---|
| `src/utils.py` | TSV I/O (`sep="\t"` always), ID-list parsing, experiment logging |
| `src/data_loader.py` | Load sources/ground truth, leakage-free entity-level train/val split, EDA |
| `src/normalize.py` | Unicode-safe (NFKC, script-preserving) name/address normalization |
| `src/blocking.py` | Multi-pass candidate generation via **inverted indices** (no nested loops), blocking recall |
| `src/features.py` | RapidFuzz + sparse TF-IDF pairwise features, vectorized (no per-pair Python loops) |
| `src/model.py` | LightGBM training, hard-negative mining helper |
| `src/decision.py` | Threshold sweep on macro F0.5, singleton/margin logic, multi-match predictions |
| `src/scorer.py` | **Exact** per-entity macro F0.5 — verified against the official worked example (0.714) |
| `utils/validate_submission.py` | **The official challenge validator** (your copy) — run this before every real submission |

## Parallelism contract

- Person 1 owns `blocking.py` + `make_candidates.py` → produces versioned,
  frozen `outputs/candidates/candidate_pairs_vN.tsv`.
- Person 2 owns `features.py` / `model.py` / `decision.py` +
  `train_matcher.py` → always trains against the **latest frozen** candidate
  version, never waits for the next one.
- Either person can run `create_submission.py` + `validate_submission.py`
  independently against any frozen `(candidate_version, model_version)` pair
  — every combination is a complete, submittable pipeline.

## Next steps once the real dataset is in place

1. Run `scripts/eda.py` on the real train split, update assumptions if the
   real singleton %, multi-match %, or country mix differs a lot from the toy data.
2. Read 50 real positive pairs / 50 hard negatives / 50 singletons by hand.
3. Re-run `make_candidates.py`, check real blocking recall — likely lower
   than the toy 100%, since real noise (typos, transliteration, landmarks)
   is much messier. Add blocks / tune `min_token_len` accordingly.
4. Add embedding features (`src/features.py`) as a V4 once the classical
   model is stable — check the model's license (MIT/Apache-2.0, reasonable
   size) before adopting it.
5. Wire in `add_margin_features` / `singleton_aware_predictions` from
   `src/decision.py` once you have a validated baseline threshold.
