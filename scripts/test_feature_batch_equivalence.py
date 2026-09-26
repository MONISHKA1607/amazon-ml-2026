from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data_loader import load_split
from src.features import (
    build_tfidf_vectorizer,
    compute_features,
    compute_features_batch,
)
from src.normalize import add_normalized_columns


def main() -> None:
    SAMPLE_S1 = 1000
    SAMPLE_S2 = 5000
    SAMPLE_S3 = 5000

    print("Loading data...")
    dataset = load_split("dataset", "train")

    s1 = dataset.source1.copy()
    s2 = dataset.source2.copy()
    s3 = dataset.source3.copy()
    gt = dataset.ground_truth.copy()

    s1 = s1.head(SAMPLE_S1).copy()
    s2 = s2.head(SAMPLE_S2).copy()
    s3 = s3.head(SAMPLE_S3).copy()

    print("Normalizing...")
    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)

    # ------------------------------------------------------------
    # Build a small candidate set using deterministic IDs.
    # ------------------------------------------------------------
    candidates_s2 = pd.DataFrame(
        {
            "source1_entity_id": s1["entity_id"].values,
            "candidate_entity_id": s2["entity_id"].iloc[
                : len(s1)
            ].values,
        }
    )

    # Repeat candidates so we have enough rows for a meaningful test.
    candidates_s2 = pd.concat(
        [
            candidates_s2,
            pd.DataFrame(
                {
                    "source1_entity_id": s1["entity_id"].iloc[
                        : len(s1)
                    ].values,
                    "candidate_entity_id": s2["entity_id"].iloc[
                        len(s1) : 2 * len(s1)
                    ].values,
                }
            ),
        ],
        ignore_index=True,
    )

    candidates_s2["blocks_matched"] = "test"
    candidates_s2["block_score"] = 1

    # ------------------------------------------------------------
    # Fit small training-only TF-IDF vectorizers.
    # ------------------------------------------------------------
    name_vectorizer = build_tfidf_vectorizer(
        pd.concat(
            [
                s1["norm_name"],
                s2["norm_name"],
            ],
            ignore_index=True,
        )
    )

    address_vectorizer = build_tfidf_vectorizer(
        pd.concat(
            [
                s1["norm_address"],
                s2["norm_address"],
            ],
            ignore_index=True,
        )
    )

    print("Computing original features...")
    original = compute_features(
        candidates_s2,
        s1,
        s2,
        name_tfidf_vectorizer=name_vectorizer,
        addr_tfidf_vectorizer=address_vectorizer,
    )

    print("Computing batch features...")
    batch = compute_features_batch(
        candidates_s2,
        s1,
        s2,
        name_tfidf_vectorizer=name_vectorizer,
        addr_tfidf_vectorizer=address_vectorizer,
    )

    original = original.sort_values(
        ["source1_entity_id", "candidate_entity_id"]
    ).reset_index(drop=True)

    batch = batch.sort_values(
        ["source1_entity_id", "candidate_entity_id"]
    ).reset_index(drop=True)

    if list(original.columns) != list(batch.columns):
        print("FAIL: feature columns differ.")
        print("Original:", list(original.columns))
        print("Batch:", list(batch.columns))
        raise SystemExit(1)

    for column in original.columns:
        a = original[column]
        b = batch[column]

        if pd.api.types.is_numeric_dtype(a):
            if not (
                (a.fillna(0) - b.fillna(0)).abs() <= 1e-10
            ).all():
                print(f"FAIL: numeric column differs: {column}")
                raise SystemExit(1)
        else:
            if not a.fillna("").equals(b.fillna("")):
                print(f"FAIL: column differs: {column}")
                raise SystemExit(1)

    print(f"Compared {len(original):,} candidate pairs.")
    print("All feature columns including TF-IDF match exactly.")
    print("PASS: batch feature computation is fully equivalent.")


if __name__ == "__main__":
    main()