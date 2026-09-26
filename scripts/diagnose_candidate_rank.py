import sys

sys.path.insert(0, ".")

import pandas as pd

from src.data_loader import load_split
from src.normalize import add_normalized_columns
from src.blocking import generate_candidates
from src.blocking_config import BlockingConfig


MISSES = {
    "S1-312975533": "S3-688080907",
}


def inspect_target(
    s1_row,
    target_id,
    s2,
    s3,
    config,
):
    s1_id = s1_row["entity_id"]

    print("\n" + "=" * 90)
    print(f"{s1_id} -> {target_id}")
    print("=" * 90)

    if target_id.startswith("S2-"):
        other_df = s2
        source_name = "S2"
    else:
        other_df = s3
        source_name = "S3"

    candidates, stats = generate_candidates(
        s1_df=pd.DataFrame([s1_row]),
        other_df=other_df,
        config=config,
        source_name=source_name,
    )

    print("\nCandidate statistics:")
    print(stats)

    if candidates.empty:
        print("\nNo candidates generated.")
        return

    # Reconstruct the exact ranking used by src.blocking._rank_candidates.
    #
    # Production ranking:
    #   1. Higher block_score first.
    #   2. Candidate ID ascending as deterministic tie-breaker.

    candidates = candidates.sort_values(
        by=[
            "source1_entity_id",
            "block_score",
            "candidate_entity_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    candidates["rank"] = candidates.index + 1

    target_rows = candidates[
        candidates["candidate_entity_id"].eq(target_id)
    ]

    print("\nTarget candidate:")

    if target_rows.empty:
        print("NOT GENERATED")
    else:
        print(
            target_rows[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                    "blocks_matched",
                    "block_score",
                    "rank",
                ]
            ].to_string(index=False)
        )

    print("\nTop 30 candidates:")
    print(
        candidates.head(30).to_string(index=False)
    )


def main():
    print("Loading training data...")

    ds = load_split("dataset", "train")

    s1_ids = list(MISSES.keys())

    s1 = ds.source1[
        ds.source1["entity_id"].isin(s1_ids)
    ].copy()

    print("Loading benchmark negatives...")

    NEGATIVE_N = 5000

    print(f"S1 records: {len(s1)}")

    print("\nNormalizing...")

    s1 = add_normalized_columns(s1)

    config = BlockingConfig(
        max_block_frequency=500,
        max_candidates_per_block=500,
        max_candidates_per_entity=5000,
    )

    print("\nBlocking configuration:")
    print(config.to_dict())

    for _, s1_row in s1.iterrows():
        s1_id = s1_row["entity_id"]
        target_id = MISSES[s1_id]

        print("\n" + "#" * 100)
        print(f"DIAGNOSTIC: {s1_id} -> {target_id}")
        print("#" * 100)

        if target_id.startswith("S2-"):
            target_df = ds.source2
            target_source = "S2"
        else:
            target_df = ds.source3
            target_source = "S3"

        target_row = target_df[
            target_df["entity_id"].eq(target_id)
        ].copy()

        negatives = target_df[
            ~target_df["entity_id"].eq(target_id)
        ].head(NEGATIVE_N)

        other_df = pd.concat(
            [target_row, negatives],
            ignore_index=True,
        )

        other_df = add_normalized_columns(other_df)

        print(f"Target source: {target_source}")
        print(f"Other records: {len(other_df)}")

        inspect_target(
            s1_row=s1_row,
            target_id=target_id,
            s2=other_df if target_source == "S2" else pd.DataFrame(),
            s3=other_df if target_source == "S3" else pd.DataFrame(),
            config=config,
        )

if __name__ == "__main__":
    main()