#!/usr/bin/env python3
"""
Train the matching model on a frozen candidate set.

Usage:
  python scripts/train_matcher.py \
      --data-dir dataset \
      --candidate-version v2 \
      --model-version v1

The train/validation split is by Source-1 entity so that all candidates
belonging to one entity stay in exactly one split.

TF-IDF vectorizers are fitted only on the training-side records and then
used to transform validation candidates.
"""

import argparse
import os
import pickle
import sys

import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
    ),
)

from src.data_loader import (  # noqa: E402
    load_split,
    ground_truth_to_dict,
    entity_level_split,
)
from src.normalize import add_normalized_columns  # noqa: E402
from src.features import (  # noqa: E402
    compute_features,
    build_tfidf_vectorizer,
    FEATURE_COLUMNS,
)
from src.model import label_candidates, train_model  # noqa: E402
from src.decision import sweep_thresholds  # noqa: E402
from src.utils import read_tsv, timer, log_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-dir",
        default="dataset",
    )
    ap.add_argument(
        "--candidate-version",
        default="v2",
    )
    ap.add_argument(
        "--model-version",
        default="v1",
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    ap.add_argument(
        "--owner",
        default="P2",
    )

    args = ap.parse_args()

    # ---------------------------------------------------------
    # 1. Load and normalize source data
    # ---------------------------------------------------------

    ds = load_split(
        args.data_dir,
        "train",
    )

    gt_dict = ground_truth_to_dict(
        ds.ground_truth
    )

    print("Normalizing Source 1...")
    s1 = add_normalized_columns(
        ds.source1
    )

    print("Normalizing Source 2...")
    s2 = add_normalized_columns(
        ds.source2
    )

    print("Normalizing Source 3...")
    s3 = add_normalized_columns(
        ds.source3
    )

    # ---------------------------------------------------------
    # 2. Load frozen candidate set
    # ---------------------------------------------------------

    cand_path = (
        f"outputs/candidates/"
        f"candidate_pairs_{args.candidate_version}.tsv"
    )

    if not os.path.isfile(cand_path):
        raise FileNotFoundError(
            f"Candidate file not found: {cand_path}\n"
            f"Run scripts/make_candidates.py first."
        )

    print(
        f"Loading candidate set: {cand_path}"
    )

    candidates = read_tsv(
        cand_path
    )

    candidates = candidates[
        candidates[
            "source1_entity_id"
        ].isin(
            s1["entity_id"]
        )
    ].copy()

    print(
        f"Candidate rows: {len(candidates):,}"
    )

    # ---------------------------------------------------------
    # 3. Entity-level train/validation split
    # ---------------------------------------------------------

    train_ids, val_ids = entity_level_split(
        ds.source1,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    train_ids = set(train_ids)
    val_ids = set(val_ids)

    print(
        f"Training entities: "
        f"{len(train_ids):,}"
    )

    print(
        f"Validation entities: "
        f"{len(val_ids):,}"
    )

    # ---------------------------------------------------------
    # 4. Identify candidate source
    # ---------------------------------------------------------

    other_lookup = {
        **{
            eid: "S2"
            for eid in s2["entity_id"]
        },
        **{
            eid: "S3"
            for eid in s3["entity_id"]
        },
    }

    candidates["source"] = candidates[
        "candidate_entity_id"
    ].map(
        other_lookup
    )

    if candidates["source"].isna().any():
        missing = int(
            candidates["source"].isna().sum()
        )
        raise ValueError(
            f"{missing} candidate rows refer to "
            "unknown Source-2/Source-3 entity IDs."
        )

    cand2 = candidates[
        candidates["source"] == "S2"
    ].copy()

    cand3 = candidates[
        candidates["source"] == "S3"
    ].copy()

    print(
        f"S2 candidate rows: {len(cand2):,}"
    )

    print(
        f"S3 candidate rows: {len(cand3):,}"
    )

    # ---------------------------------------------------------
    # 5. Fit TF-IDF ONLY on training-side records
    # ---------------------------------------------------------

    train_cand2 = cand2[
        cand2["source1_entity_id"].isin(
            train_ids
        )
    ]

    train_cand3 = cand3[
        cand3["source1_entity_id"].isin(
            train_ids
        )
    ]

    train_s1 = s1[
        s1["entity_id"].isin(
            train_ids
        )
    ]

    train_s2_ids = set(
        train_cand2[
            "candidate_entity_id"
        ]
    )

    train_s3_ids = set(
        train_cand3[
            "candidate_entity_id"
        ]
    )

    train_s2 = s2[
        s2["entity_id"].isin(
            train_s2_ids
        )
    ]

    train_s3 = s3[
        s3["entity_id"].isin(
            train_s3_ids
        )
    ]

    print(
        "Fitting name TF-IDF on training-side records..."
    )

    name_vectorizer = build_tfidf_vectorizer(
        train_s1["norm_name"],
        train_s2["norm_name"],
        train_s3["norm_name"],
    )

    print(
        "Fitting address TF-IDF on training-side records..."
    )

    addr_vectorizer = build_tfidf_vectorizer(
        train_s1["norm_address"],
        train_s2["norm_address"],
        train_s3["norm_address"],
    )

    # ---------------------------------------------------------
    # 6. Feature engineering
    # ---------------------------------------------------------

    with timer("feature engineering"):

        feats2 = (
            compute_features(
                cand2,
                s1,
                s2,
                name_vectorizer,
                addr_vectorizer,
            )
            if len(cand2)
            else cand2
        )

        feats3 = (
            compute_features(
                cand3,
                s1,
                s3,
                name_vectorizer,
                addr_vectorizer,
            )
            if len(cand3)
            else cand3
        )

        features_df = pd.concat(
            [
                feats2,
                feats3,
            ],
            ignore_index=True,
        )

    print(
        f"Feature rows: "
        f"{len(features_df):,}"
    )

    # ---------------------------------------------------------
    # 7. Construct labels
    # ---------------------------------------------------------

    labels = label_candidates(
        features_df,
        gt_dict,
    )

    print(
        f"positives={int(labels.sum()):,} / "
        f"{len(labels):,} candidate pairs "
        f"({100 * labels.mean():.3f}% positive rate)"
    )

    # ---------------------------------------------------------
    # 8. Train LightGBM on training entities only
    # ---------------------------------------------------------

    train_mask = features_df[
        "source1_entity_id"
    ].isin(
        train_ids
    )

    val_mask = features_df[
        "source1_entity_id"
    ].isin(
        val_ids
    )

    print(
        f"Training candidate rows: "
        f"{int(train_mask.sum()):,}"
    )

    print(
        f"Validation candidate rows: "
        f"{int(val_mask.sum()):,}"
    )

    with timer("model training"):

        model = train_model(
            features_df[
                train_mask
            ],
            labels[
                train_mask
            ],
            features_df.loc[
                train_mask,
                "source1_entity_id",
            ],
        )

    # ---------------------------------------------------------
    # 9. Score validation candidates
    # ---------------------------------------------------------

    val_scores = model.predict(
        features_df.loc[
            val_mask,
            FEATURE_COLUMNS,
        ].fillna(0.0).values
    )

    scored_val = features_df.loc[
        val_mask,
        [
            "source1_entity_id",
            "candidate_entity_id",
        ],
    ].copy()

    scored_val["score"] = val_scores

    val_truth = {
        k: v
        for k, v in gt_dict.items()
        if k in val_ids
    }

    # ---------------------------------------------------------
    # 10. Threshold sweep using official macro F0.5
    # ---------------------------------------------------------

    sweep = sweep_thresholds(
        scored_val,
        val_truth,
        all_s1_ids=val_ids,
    )

    print()
    print(
        sweep.head(10).to_string(
            index=False
        )
    )

    best = sweep.iloc[0]

    print()
    print(
        f"best threshold="
        f"{best['threshold']}"
    )

    print(
        f"macro_f05="
        f"{best['macro_f05']:.4f}"
    )

    print(
        f"micro_precision="
        f"{best['micro_precision']:.4f}"
    )

    print(
        f"micro_recall="
        f"{best['micro_recall']:.4f}"
    )

    # ---------------------------------------------------------
    # 11. Save model artifact
    # ---------------------------------------------------------

    os.makedirs(
        "models",
        exist_ok=True,
    )

    model_path = (
        f"models/"
        f"matcher_{args.model_version}.pkl"
    )

    with open(
        model_path,
        "wb",
    ) as f:

        pickle.dump(
            {
                "model": model,
                "name_vectorizer": name_vectorizer,
                "addr_vectorizer": addr_vectorizer,
                "threshold": float(
                    best["threshold"]
                ),
                "feature_columns": FEATURE_COLUMNS,
                "candidate_version": args.candidate_version,
            },
            f,
        )

    print(
        f"saved {model_path}"
    )

    # ---------------------------------------------------------
    # 12. Log experiment
    # ---------------------------------------------------------

    log_experiment(
        "experiments/logs/experiments.csv",
        experiment_id=(
            f"model_{args.model_version}"
        ),
        owner=args.owner,
        candidate_version=args.candidate_version,
        model_version=args.model_version,
        threshold=best["threshold"],
        macro_f05=best["macro_f05"],
        precision=best["micro_precision"],
        recall=best["micro_recall"],
        notes=(
            "train_matcher.py run; "
            "entity-level split; "
            "TF-IDF fitted on training-side "
            "records only"
        ),
    )


if __name__ == "__main__":
    main()