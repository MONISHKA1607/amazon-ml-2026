import sys

sys.path.insert(0, ".")

import pandas as pd

from src.data_loader import load_split
from src.normalize import add_normalized_columns
from src.blocking import _address_numeric_anchor_keys
from src.blocking_config import BlockingConfig


TARGET_S1 = "S1-833436524"
TARGET_S2 = "S2-910790078"


def main():
    print("Loading training data...")

    ds = load_split("dataset", "train")

    s1 = ds.source1[
        ds.source1["entity_id"].eq(TARGET_S1)
    ].copy()

    target = ds.source2[
        ds.source2["entity_id"].eq(TARGET_S2)
    ].copy()

    # Use the same 5000-negative diagnostic pool.
    negatives = ds.source2[
        ~ds.source2["entity_id"].eq(TARGET_S2)
    ].head(5000)

    s2 = pd.concat(
        [target, negatives],
        ignore_index=True,
    )

    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)

    config = BlockingConfig(
        max_block_frequency=500,
        max_candidates_per_block=500,
        max_candidates_per_entity=500,
    )

    target_row = s1.iloc[0]

    target_anchor = set(
        _address_numeric_anchor_keys(
            target_row["norm_address"],
            target_row["numeric_tokens"],
            config.address_anchor_min_token_length,
        )
    )

    print("\nS1 anchors:")
    for key in sorted(target_anchor):
        print(f"  {key}")

    print("\nCounting S2 anchor frequencies...")

    counts = {}

    for _, row in s2.iterrows():
        anchors = _address_numeric_anchor_keys(
            row["norm_address"],
            row["numeric_tokens"],
            config.address_anchor_min_token_length,
        )

        for anchor in anchors:
            counts[anchor] = counts.get(anchor, 0) + 1

    print("\nTARGET ANCHOR FREQUENCIES")
    print("-" * 80)

    for anchor in sorted(target_anchor):
        print(
            f"{anchor:35s} -> {counts.get(anchor, 0)}"
        )

    print("\nTARGET RECORD:")
    print(
        s2[
            s2["entity_id"].eq(TARGET_S2)
        ][
            [
                "entity_id",
                "business_name",
                "business_address",
                "numeric_tokens",
            ]
        ].to_string(index=False)
    )

    print("\nRelevant S2 records sharing 404||karnal:")
    print("-" * 80)

    shared = []

    for _, row in s2.iterrows():
        anchors = set(
            _address_numeric_anchor_keys(
                row["norm_address"],
                row["numeric_tokens"],
                config.address_anchor_min_token_length,
            )
        )

        if "404||karnal" in anchors:
            shared.append(
                [
                    row["entity_id"],
                    row["business_name"],
                    row["business_address"],
                ]
            )

    print(f"Count: {len(shared)}")

    if shared:
        print(
            pd.DataFrame(
                shared,
                columns=[
                    "entity_id",
                    "business_name",
                    "business_address",
                ],
            ).head(30).to_string(index=False)
        )


if __name__ == "__main__":
    main()