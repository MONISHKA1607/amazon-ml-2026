import sys

sys.path.insert(0, ".")

import pandas as pd

from src.data_loader import load_split
from src.normalize import add_normalized_columns


MISSES = {
    "S1-312975533": {"S3-688080907"},
    "S1-833436524": {"S2-910790078"},
}


def show_row(df, entity_id):
    row = df[df["entity_id"] == entity_id]

    if row.empty:
        print(f"NOT FOUND: {entity_id}")
        return

    row = row.iloc[0]

    print(f"entity_id:        {row['entity_id']}")
    print(f"business_name:    {row['business_name']}")
    print(f"business_address: {row['business_address']}")
    print(f"country:          {row['country']}")

    normalized_cols = [
        "norm_name",
        "translit_name",
        "norm_address",
        "translit_address",
        "numeric_tokens",
    ]

    print()
    print("Normalized:")
    for col in normalized_cols:
        if col in row.index:
            print(f"{col}: {row[col]}")


def main():
    print("Loading training data...")

    ds = load_split("dataset", "train")

    s1_ids = set(MISSES.keys())

    s1 = ds.source1[
        ds.source1["entity_id"].isin(s1_ids)
    ].copy()

    s2_ids = {
        entity_id
        for ids in MISSES.values()
        for entity_id in ids
        if entity_id.startswith("S2-")
    }

    s3_ids = {
        entity_id
        for ids in MISSES.values()
        for entity_id in ids
        if entity_id.startswith("S3-")
    }

    s2 = ds.source2[
        ds.source2["entity_id"].isin(s2_ids)
    ].copy()

    s3 = ds.source3[
        ds.source3["entity_id"].isin(s3_ids)
    ].copy()

    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)

    print()
    print("=" * 80)
    print("REMAINING MISSES")
    print("=" * 80)

    for s1_id, true_ids in MISSES.items():

        print()
        print(f"SOURCE 1: {s1_id}")
        print("-" * 80)

        show_row(s1, s1_id)

        for candidate_id in true_ids:

            print()
            print(f"TRUE MATCH: {candidate_id}")
            print("-" * 80)

            if candidate_id.startswith("S2-"):
                show_row(s2, candidate_id)
            else:
                show_row(s3, candidate_id)


if __name__ == "__main__":
    main()