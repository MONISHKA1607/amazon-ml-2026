#!/usr/bin/env python3

import os
import pickle
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_split
from src.normalize import add_normalized_columns
from src.blocking import generate_all_candidates
from src.features import compute_features, FEATURE_COLUMNS


TARGET_S1 = "S1-312975533"
TARGET_CANDIDATE = "S3-688080907"


def main():
    print("Loading training data...")

    ds = load_split("dataset", "train")

    s1 = ds.source1[
        ds.source1["entity_id"].eq(TARGET_S1)
    ].copy()

    s3 = ds.source3[
        ds.source3["entity_id"].eq(TARGET_CANDIDATE)
    ].copy()

    if s1.empty:
        raise ValueError(f"Missing S1 entity: {TARGET_S1}")

    if s3.empty:
        raise ValueError(f"Missing candidate: {TARGET_CANDIDATE}")

    # We need the full S3 pool because candidate generation depends
    # on posting-list frequencies and block collisions.
    print("Loading full Source-3...")

    other = ds.source3.copy()

    print("Normalizing...")

    s1 = add_normalized_columns(s1)
    other = add_normalized_columns(other)

    print("Generating candidates...")

    candidates = generate_all_candidates(
        s1,
        pd.DataFrame(
            columns=ds.source2.columns
        ),
        other,
    )

    target_candidates = candidates[
        candidates["candidate_entity_id"].eq(TARGET_CANDIDATE)
    ].copy()

    print("\nTarget candidate after blocking:")

    if target_candidates.empty:
        print("NOT GENERATED")
        return

    print(
        target_candidates.to_string(index=False)
    )

    print("\nLoading model...")

    with open("models/matcher_v1.pkl", "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    name_vec = bundle["name_vectorizer"]
    addr_vec = bundle["addr_vectorizer"]
    threshold = bundle["threshold"]

    print(f"Model threshold: {threshold}")

    print("\nComputing features...")

    features = compute_features(
        target_candidates,
        s1,
        other,
        name_vec,
        addr_vec,
    )

    scores = model.predict(
        features[FEATURE_COLUMNS].fillna(0.0).values
    )

    result = features[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "blocks_matched",
            "block_score",
        ]
        + FEATURE_COLUMNS
    ].copy()

    result["score"] = scores
    result["threshold"] = threshold
    result["passes_threshold"] = result["score"] >= threshold

    print("\nTARGET FEATURES + SCORE:")
    print(
        result.to_string(index=False)
    )

    print("\nRelevant comparison:")
    print(
        result[
            [
                "candidate_entity_id",
                "block_score",
                "name_ratio",
                "name_partial_ratio",
                "name_token_sort_ratio",
                "name_token_set_ratio",
                "addr_ratio",
                "addr_token_sort_ratio",
                "name_exact",
                "addr_exact",
                "country_match",
                "name_token_jaccard",
                "addr_token_jaccard",
                "numeric_overlap",
                "name_tfidf_cosine",
                "addr_tfidf_cosine",
                "score",
                "threshold",
                "passes_threshold",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()