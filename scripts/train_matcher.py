from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import pickle
import random
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.blocking import (
    build_blocking_context,
    generate_candidates_batch,
)
from src.blocking_config import BlockingConfig
from src.data_loader import load_split
from src.features import (
    FEATURE_COLUMNS,
    build_tfidf_vectorizer,
    compute_features_batch,
)
from src.model import label_candidates
from src.normalize import add_normalized_columns


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train matcher using batched candidate generation."
    )

    parser.add_argument(
        "--data-dir",
        default="dataset",
    )

    parser.add_argument(
        "--sample-s1",
        type=int,
        default=None,
        help="Use only the first N S1 entities. Default: all.",
    )

    parser.add_argument(
        "--sample-other",
        type=int,
        default=None,
        help="Use only the first N S2/S3 records. Default: all.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=250,
    )

    parser.add_argument(
        "--hard-negatives",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--random-negatives",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        default="models/matcher_streaming_dryrun.pkl",
    )

    return parser.parse_args()


def split_s1_entities(
    s1_df: pd.DataFrame,
    val_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entity-level split.

    The split happens BEFORE candidate generation and negative sampling,
    preventing the same S1 entity from appearing in both train and
    validation.
    """

    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be between 0 and 1.")

    ids = s1_df["entity_id"].astype(str).tolist()

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * val_frac))

    val_ids = set(ids[:n_val])

    train_mask = ~s1_df["entity_id"].astype(str).isin(val_ids)

    train_s1 = s1_df.loc[train_mask].copy()
    val_s1 = s1_df.loc[~train_mask].copy()

    return train_s1, val_s1


def build_ground_truth_sets(
    ground_truth: pd.DataFrame,
) -> dict[str, set[str]]:
    """
    Convert the challenge ground-truth table into:

        source1_entity_id -> set(candidate_entity_id)
    """

    result: dict[str, set[str]] = {}

    for row in ground_truth.itertuples(index=False):
        s1_id = str(row.source1_entity_id)

        value = row.matched_entity_ids

        if pd.isna(value) or str(value).strip() == "":
            result[s1_id] = set()
            continue

        result[s1_id] = {
            token.strip()
            for token in str(value).split(",")
            if token.strip()
        }

    return result


def select_training_candidates(
    candidates: pd.DataFrame,
    labels: pd.Series,
    hard_negatives: int,
    random_negatives: int,
    seed: int,
) -> pd.DataFrame:
    """
    Keep:

      - every recovered positive
      - up to N hard negatives based on name_ratio
      - up to M deterministic random negatives

    Sampling is performed independently per S1 entity.
    """

    if candidates.empty:
        return candidates.copy()

    work = candidates.copy()
    work["label"] = labels.to_numpy()

    selected_parts: list[pd.DataFrame] = []

    rng = np.random.default_rng(seed)

    for s1_id, group in work.groupby(
        "source1_entity_id",
        sort=False,
    ):
        positives = group[group["label"] == 1]

        negatives = group[group["label"] == 0]

        # -------------------------------------------------------------
        # All recovered positives are always retained.
        # -------------------------------------------------------------
        selected = [positives]

        # -------------------------------------------------------------
        # Hard negatives.
        #
        # name_ratio is used because it is cheap and deterministic.
        # -------------------------------------------------------------
        if hard_negatives > 0 and not negatives.empty:
            hard = negatives.sort_values(
                "name_ratio",
                ascending=False,
                kind="mergesort",
            ).head(hard_negatives)

            selected.append(hard)

            remaining = negatives.drop(index=hard.index)
        else:
            remaining = negatives

        # -------------------------------------------------------------
        # Random negatives.
        # -------------------------------------------------------------
        if random_negatives > 0 and not remaining.empty:
            n = min(random_negatives, len(remaining))

            random_indices = rng.choice(
                remaining.index.to_numpy(),
                size=n,
                replace=False,
            )

            random_part = remaining.loc[random_indices]

            selected.append(random_part)

        selected_parts.append(
            pd.concat(selected, ignore_index=False)
        )

    if not selected_parts:
        return work.iloc[0:0].copy()

    return pd.concat(
        selected_parts,
        ignore_index=True,
    )


def fit_small_tfidf(
    train_s1: pd.DataFrame,
    train_other: list[pd.DataFrame],
):
    """
    Fit TF-IDF on training-side entity text only.

    This dry-run implementation intentionally uses the available
    training records directly. We will introduce a controlled
    TF-IDF fitting sample before the full-scale run.
    """

    name_parts = [train_s1["norm_name"]]
    addr_parts = [train_s1["norm_address"]]

    for df in train_other:
        name_parts.append(df["norm_name"])
        addr_parts.append(df["norm_address"])

    name_text = pd.concat(
        name_parts,
        ignore_index=True,
    )

    addr_text = pd.concat(
        addr_parts,
        ignore_index=True,
    )

    print(
        f"Fitting name TF-IDF on {len(name_text):,} records..."
    )

    name_vectorizer = build_tfidf_vectorizer(name_text)

    print(
        f"Fitting address TF-IDF on {len(addr_text):,} records..."
    )

    addr_vectorizer = build_tfidf_vectorizer(addr_text)

    return name_vectorizer, addr_vectorizer


def train_lightgbm(
    features: pd.DataFrame,
    labels: pd.Series,
    num_rounds: int,
):
    X = features[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    y = labels.astype(int)

    dataset = lgb.Dataset(
        X,
        label=y,
        feature_name=FEATURE_COLUMNS,
        free_raw_data=False,
    )

    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "is_unbalance": True,
        "seed": 42,
    }

    model = lgb.train(
        params,
        dataset,
        num_boost_round=num_rounds,
    )

    return model


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    start_time = time.time()

    print("=" * 72)
    print("STREAMING MATCHER TRAINING")
    print("=" * 72)

    print("\n[1/8] Loading training data...")

    dataset = load_split(
        args.data_dir,
        "train",
    )

    s1 = dataset.source1.copy()
    s2 = dataset.source2.copy()
    s3 = dataset.source3.copy()
    ground_truth = dataset.ground_truth.copy()

    print(f"S1 records: {len(s1):,}")
    print(f"S2 records: {len(s2):,}")
    print(f"S3 records: {len(s3):,}")
    print(f"GT rows:    {len(ground_truth):,}")

    # ---------------------------------------------------------------
    # Optional dry-run subset.
    # ---------------------------------------------------------------

    if args.sample_s1 is not None:
        if args.sample_s1 <= 0:
            raise ValueError("--sample-s1 must be positive.")

        s1 = s1.head(args.sample_s1).copy()

        print(
            f"\nDRY-RUN: restricting S1 to "
            f"{len(s1):,} records."
        )

    if args.sample_other is not None:
        if args.sample_other <= 0:
            raise ValueError("--sample-other must be positive.")

        s2 = s2.head(args.sample_other).copy()
        s3 = s3.head(args.sample_other).copy()

        print(
            f"DRY-RUN: restricting S2/S3 to "
            f"{len(s2):,} records each."
        )

    # ---------------------------------------------------------------
    # Normalize.
    # ---------------------------------------------------------------

    print("\n[2/8] Normalizing data...")

    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)

    # ---------------------------------------------------------------
    # Entity-level train/validation split.
    # ---------------------------------------------------------------

    print("\n[3/8] Creating entity-level train/validation split...")

    train_s1, val_s1 = split_s1_entities(
        s1,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    print(f"Train S1: {len(train_s1):,}")
    print(f"Val S1:   {len(val_s1):,}")

    # ---------------------------------------------------------------
    # Ground truth.
    # ---------------------------------------------------------------

    gt_dict = build_ground_truth_sets(
        ground_truth
    )

    # ---------------------------------------------------------------
    # Build blocking contexts ONCE.
    # ---------------------------------------------------------------

    print("\n[4/8] Building blocking contexts...")

    config = BlockingConfig()

    print("Building S2 blocking context...")
    s2_context = build_blocking_context(
        s2,
        config=config,
    )

    print("Building S3 blocking context...")
    s3_context = build_blocking_context(
        s3,
        config=config,
    )

    # ---------------------------------------------------------------
    # Generate sampled training candidates.
    # ---------------------------------------------------------------

    print("\n[5/8] Generating training candidates...")

    training_parts: list[pd.DataFrame] = []

    batch_counter = 0

    for start in range(
        0,
        len(train_s1),
        args.batch_size,
    ):
        end = min(
            start + args.batch_size,
            len(train_s1),
        )

        batch = train_s1.iloc[start:end].copy()

        print(
            f"  Train batch {start:,}:{end:,}"
        )

        for source_name, other_df, context in [
            ("s2", s2, s2_context),
            ("s3", s3, s3_context),
        ]:
            candidates, stats = generate_candidates_batch(
                batch,
                context,
                config,
                source_name,
            )

            if candidates.empty:
                continue

            labels = label_candidates(
                candidates,
                gt_dict,
            )

            # -------------------------------------------------------
            # Compute only the features needed for negative sampling.
            #
            # For the dry run, compute the full feature set.
            # -------------------------------------------------------

            features = compute_features_batch(
                candidates,
                batch,
                other_df,
            )

            candidates_with_features = candidates.copy()

            for column in features.columns:
                if column not in {
                    "source1_entity_id",
                    "candidate_entity_id",
                }:
                    candidates_with_features[column] = (
                        features[column].to_numpy()
                    )

            selected = select_training_candidates(
                candidates_with_features,
                labels,
                hard_negatives=args.hard_negatives,
                random_negatives=args.random_negatives,
                seed=args.seed + batch_counter,
            )

            if not selected.empty:
                selected["source_name"] = source_name
                training_parts.append(selected)

        batch_counter += 1

    if not training_parts:
        raise RuntimeError(
            "No training candidates were produced."
        )

    training_df = pd.concat(
        training_parts,
        ignore_index=True,
    )

    print(
        f"\nSampled training rows: "
        f"{len(training_df):,}"
    )

    print(
        "Positive rows:",
        int(training_df["label"].sum()),
    )

    print(
        "Negative rows:",
        int((training_df["label"] == 0).sum()),
    )

    # ---------------------------------------------------------------
    # Fit TF-IDF.
    # ---------------------------------------------------------------

    print("\n[6/8] Fitting TF-IDF...")

    name_vectorizer, addr_vectorizer = fit_small_tfidf(
        train_s1,
        [s2, s3],
    )

    # ---------------------------------------------------------------
    # Recompute selected training features with TF-IDF enabled.
    # ---------------------------------------------------------------

    print(
        "\nComputing final sampled training features..."
    )

    final_feature_parts: list[pd.DataFrame] = []

    for source_name, other_df in [
        ("s2", s2),
        ("s3", s3),
    ]:
        part = training_df[
            training_df["source_name"] == source_name
        ].copy()

        if part.empty:
            continue

        features = compute_features_batch(
            part[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                    "blocks_matched",
                    "block_score",
                ]
            ],
            train_s1,
            other_df,
            name_tfidf_vectorizer=name_vectorizer,
            addr_tfidf_vectorizer=addr_vectorizer,
        )

        features["label"] = part["label"].to_numpy()

        final_feature_parts.append(features)

    train_features = pd.concat(
        final_feature_parts,
        ignore_index=True,
    )

    labels = train_features["label"].astype(int)

    # ---------------------------------------------------------------
    # Train model.
    # ---------------------------------------------------------------

    print("\n[7/8] Training LightGBM...")

    model = train_lightgbm(
        train_features,
        labels,
        num_rounds=args.num_rounds,
    )

    # ---------------------------------------------------------------
    # Save artifact.
    # ---------------------------------------------------------------

    print("\n[8/8] Saving model artifact...")

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "name_tfidf_vectorizer": name_vectorizer,
        "addr_tfidf_vectorizer": addr_vectorizer,
        "blocking_config": config.to_dict(),
        "training_rows": len(train_features),
        "positive_rows": int(labels.sum()),
        "negative_rows": int((labels == 0).sum()),
        "seed": args.seed,
    }

    with output_path.open("wb") as f:
        pickle.dump(
            artifact,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    elapsed = time.time() - start_time

    print("\n" + "=" * 72)
    print("TRAINING DRY RUN COMPLETE")
    print("=" * 72)

    print(f"Training rows: {len(train_features):,}")
    print(f"Positive rows: {int(labels.sum()):,}")
    print(f"Negative rows: {int((labels == 0).sum()):,}")
    print(f"Model:         {output_path}")
    print(f"Elapsed:       {elapsed / 60:.2f} minutes")


if __name__ == "__main__":
    main()