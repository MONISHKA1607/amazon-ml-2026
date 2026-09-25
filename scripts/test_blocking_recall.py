import sys
import random

sys.path.insert(0, ".")

import pandas as pd

from src.data_loader import load_split, ground_truth_to_dict
from src.normalize import add_normalized_columns
from src.blocking import generate_all_candidates, compute_blocking_recall
from src.blocking_config import BlockingConfig


SEED = 42
N_S1_SAMPLE = 1000
N_NEGATIVE_S2 = 5000
N_NEGATIVE_S3 = 5000


def load_selected_records(
    path: str,
    selected_ids: set[str],
    n_random_negatives: int,
    seed: int,
) -> pd.DataFrame:
    """
    Read a large TSV in chunks and keep:

    1. every record whose entity_id is in selected_ids
    2. a deterministic sample of random negative records

    This avoids loading the entire large Source-2/Source-3 dataframe
    into memory.
    """

    selected_parts = []
    negative_parts = []

    selected_found = set()

    # Reservoir sampling for negative examples.
    reservoir = []
    rng = random.Random(seed)

    total_non_selected_seen = 0

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=usecols,
        chunksize=100_000,
    ):
        # -----------------------------------------------------------
        # Keep every required true-match record.
        # -----------------------------------------------------------

        selected_mask = chunk["entity_id"].isin(selected_ids)

        if selected_mask.any():
            selected_chunk = chunk.loc[selected_mask].copy()

            selected_parts.append(selected_chunk)

            selected_found.update(
                selected_chunk["entity_id"].tolist()
            )

        # -----------------------------------------------------------
        # Reservoir sample records that are NOT true matches.
        # -----------------------------------------------------------

        negative_chunk = chunk.loc[~selected_mask]

        for row in negative_chunk.itertuples(index=False):
            total_non_selected_seen += 1

            if len(reservoir) < n_random_negatives:
                reservoir.append(row)
            else:
                j = rng.randint(
                    1,
                    total_non_selected_seen,
                )

                if j <= n_random_negatives:
                    reservoir[j - 1] = row

    missing = selected_ids - selected_found

    if missing:
        raise RuntimeError(
            f"Could not find {len(missing)} selected entity IDs "
            f"in {path}. Examples: {sorted(missing)[:10]}"
        )

    negative_df = pd.DataFrame(
        reservoir,
        columns=usecols,
    )

    if selected_parts:
        selected_df = pd.concat(
            selected_parts,
            ignore_index=True,
        )
    else:
        selected_df = pd.DataFrame(
            columns=usecols
        )

    result = pd.concat(
        [selected_df, negative_df],
        ignore_index=True,
    )

    # Remove any accidental duplicate entity IDs.
    result = result.drop_duplicates(
        subset=["entity_id"],
        keep="first",
    ).reset_index(drop=True)

    return result


def main():
    print("Loading Source-1 and ground truth...")

    ds = load_split("dataset", "train")

    if ds.ground_truth is None:
        raise RuntimeError(
            "Training ground truth was not loaded."
        )

    gt_dict = ground_truth_to_dict(
        ds.ground_truth
    )

    print(
        f"Source 1: "
        f"{len(ds.source1):,}"
    )

    print(
        f"Ground-truth S1 rows: "
        f"{len(gt_dict):,}"
    )

    # ---------------------------------------------------------------
    # Select deterministic S1 sample.
    # ---------------------------------------------------------------

    rng = random.Random(SEED)

    all_s1_ids = list(
        ds.source1["entity_id"]
    )

    sampled_s1_ids = rng.sample(
        all_s1_ids,
        N_S1_SAMPLE,
    )

    sampled_s1_ids_set = set(
        sampled_s1_ids
    )

    s1 = ds.source1[
        ds.source1["entity_id"].isin(
            sampled_s1_ids_set
        )
    ].copy()

    # ---------------------------------------------------------------
    # Find true S2/S3 IDs.
    # ---------------------------------------------------------------

    true_s2_ids = set()
    true_s3_ids = set()

    for s1_id in sampled_s1_ids:
        true_ids = gt_dict.get(
            s1_id,
            set(),
        )

        for entity_id in true_ids:
            if entity_id.startswith("S2-"):
                true_s2_ids.add(entity_id)

            elif entity_id.startswith("S3-"):
                true_s3_ids.add(entity_id)

    print()
    print(
        f"Sampled S1 entities: "
        f"{len(s1):,}"
    )

    print(
        f"True S2 matches: "
        f"{len(true_s2_ids):,}"
    )

    print(
        f"True S3 matches: "
        f"{len(true_s3_ids):,}"
    )

    # ---------------------------------------------------------------
    # Read only required S2 records + random negatives.
    # ---------------------------------------------------------------

    print()
    print(
        "Reading selected S2 records "
        "and random negatives..."
    )

    s2 = load_selected_records(
        path="dataset/train/train_source2.tsv",
        selected_ids=true_s2_ids,
        n_random_negatives=N_NEGATIVE_S2,
        seed=SEED,
    )

    print(
        f"S2 benchmark records: "
        f"{len(s2):,}"
    )

    # ---------------------------------------------------------------
    # Read only required S3 records + random negatives.
    # ---------------------------------------------------------------

    print()
    print(
        "Reading selected S3 records "
        "and random negatives..."
    )

    s3 = load_selected_records(
        path="dataset/train/train_source3.tsv",
        selected_ids=true_s3_ids,
        n_random_negatives=N_NEGATIVE_S3,
        seed=SEED,
    )

    print(
        f"S3 benchmark records: "
        f"{len(s3):,}"
    )

    # ---------------------------------------------------------------
    # Normalize only the small benchmark.
    # ---------------------------------------------------------------

    print()
    print("Normalizing benchmark data...")

    s1 = add_normalized_columns(
        s1
    )

    s2 = add_normalized_columns(
        s2
    )

    s3 = add_normalized_columns(
        s3
    )

    # ---------------------------------------------------------------
    # Blocking configuration.
    # ---------------------------------------------------------------

    config = BlockingConfig(
        max_block_frequency=500,
        max_candidates_per_block=500,
        max_candidates_per_entity=500,
    )

    print()
    print("Blocking configuration:")
    print(config.to_dict())

    # ---------------------------------------------------------------
    # Generate candidates.
    # ---------------------------------------------------------------

    print()
    print("Generating candidates...")

    candidates, stats = generate_all_candidates(
        s1_df=s1,
        s2_df=s2,
        s3_df=s3,
        config=config,
    )

    print()
    print("Candidate generation complete.")

    print()
    print("Candidate statistics:")
    print(stats)

    print()
    print("Candidate shape:")
    print(candidates.shape)

    # ---------------------------------------------------------------
    # Restrict ground truth to sampled S1s.
    # ---------------------------------------------------------------

    sampled_gt = {
        s1_id: gt_dict.get(
            s1_id,
            set(),
        )
        for s1_id in sampled_s1_ids
    }

    # ---------------------------------------------------------------
    # Blocking recall.
    # ---------------------------------------------------------------

    recall_stats = compute_blocking_recall(
        candidates_df=candidates,
        gt_dict=sampled_gt,
    )

    print()
    print("=" * 70)
    print("BLOCKING RECALL")
    print("=" * 70)

    print(
        f"Blocking recall: "
        f"{recall_stats['blocking_recall']:.6f}"
    )

    print(
        f"Total true pairs: "
        f"{recall_stats['total_true_pairs']:,}"
    )

    print(
        f"Found true pairs: "
        f"{recall_stats['total_found_pairs']:,}"
    )

    missed = (
        recall_stats["total_true_pairs"]
        - recall_stats["total_found_pairs"]
    )

    print(
        f"Missed true pairs: "
        f"{missed:,}"
    )

    print(
        f"S1 entities with missed matches: "
        f"{recall_stats['n_entities_with_missed_matches']:,}"
    )

    print()
    print("Candidate volume:")

    print(
        f"Total candidates: "
        f"{recall_stats['candidate_pair_count']:,}"
    )

    print(
        f"Average candidates/S1: "
        f"{recall_stats['avg_candidates_per_s1']:.2f}"
    )

    print(
        f"Median candidates/S1: "
        f"{recall_stats['median_candidates_per_s1']:.2f}"
    )

    print(
        f"P95 candidates/S1: "
        f"{recall_stats['p95_candidates_per_s1']:.2f}"
    )

    print(
        f"Maximum candidates/S1: "
        f"{recall_stats['max_candidates_per_s1']:,}"
    )

    print()
    print("Missed examples:")

    for s1_id, missed_ids in recall_stats[
        "missed_examples"
    ]:
        print(
            f"  {s1_id}: "
            f"{sorted(missed_ids)[:10]}"
        )


if __name__ == "__main__":
    main()