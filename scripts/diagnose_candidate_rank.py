import sys

sys.path.insert(0, ".")

import pandas as pd

from src.data_loader import load_split
from src.normalize import add_normalized_columns
from src.blocking import generate_candidates
from src.blocking_config import BlockingConfig


MISSES = {
    "S1-312975533": "S3-688080907",
    "S1-833436524": "S2-910790078",
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

    # Reconstruct the weighted ranking used by src.blocking._rank_candidates.
    block_weights = {
        "exact_name": 8,
        "address_numeric_anchor": 8,
        "rare_name_token": 4,
        "rare_address_token": 4,
        "rare_translit_address_token": 4,
        "numeric_tokens": 3,
        "country_name_prefix": 3,
        "name_tokens": 2,
        "address_tokens": 2,
        "translit_name_tokens": 2,
        "translit_address_tokens": 2,
        "char_ngrams": 1,
        "translit_char_ngrams": 1,
    }


    def weighted_score(blocks_matched):
        blocks = set(blocks_matched.split("|"))
        return sum(
            block_weights.get(block, 1)
            for block in blocks
        )


    candidates["weighted_score"] = candidates["blocks_matched"].apply(
        weighted_score
    )

    candidates = candidates.sort_values(
        by=[
            "source1_entity_id",
            "weighted_score",
            "block_score",
            "candidate_entity_id",
        ],
        ascending=[
            True,
            False,
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
                    "weighted_score",
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

    target_s2_id = MISSES["S1-833436524"]
    target_s3_id = MISSES["S1-312975533"]

    s2_target = ds.source2[
        ds.source2["entity_id"].eq(target_s2_id)
    ].copy()

    s3_target = ds.source3[
        ds.source3["entity_id"].eq(target_s3_id)
    ].copy()

    print("Loading benchmark negatives...")

    NEGATIVE_N = 5000

    s2_neg = ds.source2[
        ~ds.source2["entity_id"].eq(target_s2_id)
    ].head(NEGATIVE_N)

    s3_neg = ds.source3[
        ~ds.source3["entity_id"].eq(target_s3_id)
    ].head(NEGATIVE_N)

    s2 = pd.concat(
        [s2_target, s2_neg],
        ignore_index=True,
    )

    s3 = pd.concat(
        [s3_target, s3_neg],
        ignore_index=True,
    )

    print(f"S1 records: {len(s1)}")
    print(f"S2 records: {len(s2)}")
    print(f"S3 records: {len(s3)}")

    print("\nNormalizing...")

    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)

    config = BlockingConfig(
        max_block_frequency=500,
        max_candidates_per_block=500,
        max_candidates_per_entity=5000,
    )

    print("\nBlocking configuration:")
    print(config.to_dict())

    for _, s1_row in s1.iterrows():
        target_id = MISSES[s1_row["entity_id"]]

        inspect_target(
            s1_row=s1_row,
            target_id=target_id,
            s2=s2,
            s3=s3,
            config=config,
        )


if __name__ == "__main__":
    main()