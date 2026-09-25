#!/usr/bin/env python3
"""
Run the frozen (candidate version + model version) pipeline over the TEST
set and produce matching_results.tsv + candidate_pairs.tsv exactly as
required for submission.

Usage:
  python scripts/create_submission.py --data-dir dataset \
      --candidate-version v1 --model-version v1 --submission-id 01
"""
import argparse
import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_split  # noqa: E402
from src.normalize import add_normalized_columns  # noqa: E402
from src.blocking import generate_all_candidates  # noqa: E402
from src.features import compute_features, FEATURE_COLUMNS  # noqa: E402
from src.decision import scores_to_predictions  # noqa: E402
from src.utils import write_tsv, format_id_list, timer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--candidate-version", default="v1")
    ap.add_argument("--model-version", default="v1")
    ap.add_argument("--submission-id", default="01")
    ap.add_argument("--threshold", type=float, default=None, help="override the model's saved threshold")
    ap.add_argument("--blocks", default=None)
    args = ap.parse_args()

    ds = load_split(args.data_dir, "test")
    s1 = add_normalized_columns(ds.source1)
    s2 = add_normalized_columns(ds.source2)
    s3 = add_normalized_columns(ds.source3)

    block_names = args.blocks.split(",") if args.blocks else None
    with timer("test blocking"):
        candidates = generate_all_candidates(s1, s2, s3, block_names)

    with open(f"models/matcher_{args.model_version}.pkl", "rb") as f:
        bundle = pickle.load(f)
    model, name_vec, addr_vec = bundle["model"], bundle["name_vectorizer"], bundle["addr_vectorizer"]
    threshold = args.threshold if args.threshold is not None else bundle["threshold"]

    other_lookup = {**{eid: "S2" for eid in s2["entity_id"]}, **{eid: "S3" for eid in s3["entity_id"]}}
    candidates["source"] = candidates["candidate_entity_id"].map(other_lookup)
    cand2 = candidates[candidates["source"] == "S2"]
    cand3 = candidates[candidates["source"] == "S3"]

    with timer("test feature engineering"):
        feats2 = compute_features(cand2, s1, s2, name_vec, addr_vec) if len(cand2) else cand2
        feats3 = compute_features(cand3, s1, s3, name_vec, addr_vec) if len(cand3) else cand3
        features_df = pd.concat([feats2, feats3], ignore_index=True)

    scores = model.predict(features_df[FEATURE_COLUMNS].fillna(0.0).values)
    scored_df = features_df[["source1_entity_id", "candidate_entity_id"]].copy()
    scored_df["score"] = scores

    predictions = scores_to_predictions(scored_df, threshold)

    # --- write candidate_pairs.tsv: the FINAL set actually scored by the model ---
    cand_rows = []
    for s1_id, group in candidates.groupby("source1_entity_id"):
        cand_rows.append((s1_id, format_id_list(group["candidate_entity_id"])))
    for s1_id in s1["entity_id"]:
        if s1_id not in {r[0] for r in cand_rows}:
            cand_rows.append((s1_id, ""))
    cand_out = pd.DataFrame(cand_rows, columns=["source1_entity_id", "candidate_entity_ids"])

    # --- write matching_results.tsv: EVERY test S1 entity, one row each ---
    match_rows = []
    for s1_id in s1["entity_id"]:
        match_rows.append((s1_id, format_id_list(predictions.get(s1_id, set()))))
    match_out = pd.DataFrame(match_rows, columns=["source1_entity_id", "matched_entity_ids"])

    out_dir = f"outputs/submissions/submission_{args.submission_id}"
    write_tsv(match_out, os.path.join(out_dir, "matching_results.tsv"))
    write_tsv(cand_out, os.path.join(out_dir, "candidate_pairs.tsv"))
    print(f"submission {args.submission_id}: threshold={threshold}, "
          f"{sum(1 for _, m in match_rows if m)} entities matched, "
          f"{sum(1 for _, m in match_rows if not m)} singleton-predicted")
    print(f"wrote -> {out_dir}/matching_results.tsv and candidate_pairs.tsv")
    print("Run utils/validate_submission.py (the official validator) next, before uploading.")


if __name__ == "__main__":
    main()
