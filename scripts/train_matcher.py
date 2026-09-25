#!/usr/bin/env python3
"""
Train the matching model (Person 2's deliverable) on a frozen candidate set.

Usage:
  python scripts/train_matcher.py --data-dir dataset --candidate-version v1 --model-version v1

Splits by Source-1 entity (leakage-free), trains LightGBM, sweeps thresholds
on the held-out validation entities using macro F0.5, and saves everything
needed for inference: model, TF-IDF vectorizers, chosen threshold.
"""
import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_split, ground_truth_to_dict, entity_level_split  # noqa: E402
from src.normalize import add_normalized_columns  # noqa: E402
from src.features import compute_features, build_tfidf_matrix, FEATURE_COLUMNS  # noqa: E402
from src.model import label_candidates, train_model  # noqa: E402
from src.decision import sweep_thresholds  # noqa: E402
from src.utils import read_tsv, timer, log_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--candidate-version", default="v1")
    ap.add_argument("--model-version", default="v1")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--owner", default="P2")
    args = ap.parse_args()

    ds = load_split(args.data_dir, "train")
    gt_dict = ground_truth_to_dict(ds.ground_truth)
    s1 = add_normalized_columns(ds.source1)
    s2 = add_normalized_columns(ds.source2)
    s3 = add_normalized_columns(ds.source3)

    cand_path = f"outputs/candidates/candidate_pairs_{args.candidate_version}.tsv"
    candidates = read_tsv(cand_path)
    candidates = candidates[candidates["source1_entity_id"].isin(s1["entity_id"])]

    # Fit TF-IDF once, shared across S2 and S3 candidates.
    name_vectorizer, _ = build_tfidf_matrix(s1["norm_name"], s2["norm_name"], s3["norm_name"])
    addr_vectorizer, _ = build_tfidf_matrix(s1["norm_address"], s2["norm_address"], s3["norm_address"])

    other_lookup = {**{eid: "S2" for eid in s2["entity_id"]}, **{eid: "S3" for eid in s3["entity_id"]}}
    candidates["source"] = candidates["candidate_entity_id"].map(other_lookup)
    cand2 = candidates[candidates["source"] == "S2"]
    cand3 = candidates[candidates["source"] == "S3"]

    with timer("feature engineering"):
        feats2 = compute_features(cand2, s1, s2, name_vectorizer, addr_vectorizer) if len(cand2) else cand2
        feats3 = compute_features(cand3, s1, s3, name_vectorizer, addr_vectorizer) if len(cand3) else cand3
        import pandas as pd
        features_df = pd.concat([feats2, feats3], ignore_index=True)

    labels = label_candidates(features_df, gt_dict)
    print(f"positives={int(labels.sum())} / {len(labels)} candidate pairs "
          f"({100*labels.mean():.3f}% positive rate)")

    train_ids, val_ids = entity_level_split(ds.source1, val_frac=args.val_frac, seed=args.seed)
    train_mask = features_df["source1_entity_id"].isin(train_ids)
    val_mask = features_df["source1_entity_id"].isin(val_ids)

    with timer("model training"):
        model = train_model(
            features_df[train_mask], labels[train_mask], features_df.loc[train_mask, "source1_entity_id"]
        )

    val_scores = model.predict(features_df.loc[val_mask, FEATURE_COLUMNS].fillna(0.0).values)
    scored_val = features_df.loc[val_mask, ["source1_entity_id", "candidate_entity_id"]].copy()
    scored_val["score"] = val_scores

    val_truth = {k: v for k, v in gt_dict.items() if k in val_ids}
    sweep = sweep_thresholds(scored_val, val_truth, all_s1_ids=val_ids)
    print(sweep.head(10).to_string(index=False))
    best = sweep.iloc[0]
    print(f"\nbest threshold={best['threshold']} macro_f05={best['macro_f05']:.4f}")

    os.makedirs("models", exist_ok=True)
    with open(f"models/matcher_{args.model_version}.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "name_vectorizer": name_vectorizer,
            "addr_vectorizer": addr_vectorizer,
            "threshold": float(best["threshold"]),
            "feature_columns": FEATURE_COLUMNS,
        }, f)
    print(f"saved models/matcher_{args.model_version}.pkl")

    log_experiment(
        "experiments/logs/experiments.csv",
        experiment_id=f"model_{args.model_version}",
        owner=args.owner,
        candidate_version=args.candidate_version,
        model_version=args.model_version,
        threshold=best["threshold"],
        macro_f05=best["macro_f05"],
        precision=best["micro_precision"],
        recall=best["micro_recall"],
        notes="train_matcher.py run",
    )


if __name__ == "__main__":
    main()
