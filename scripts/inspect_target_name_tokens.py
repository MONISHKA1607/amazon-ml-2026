import sys

sys.path.insert(0, ".")

import pandas as pd

from src.data_loader import load_split
from src.normalize import add_normalized_columns
from src.blocking import (
    _build_token_frequency,
    _build_rare_token_index,
    _safe_tokens,
)
from src.blocking_config import BlockingConfig


TARGET_S1 = "S1-312975533"
TARGET_S3 = "S3-688080907"


def main():
    print("Loading training data...")

    ds = load_split("dataset", "train")

    # ------------------------------------------------------------------
    # Load only the target records.
    # ------------------------------------------------------------------

    s1 = ds.source1[
        ds.source1["entity_id"].eq(TARGET_S1)
    ].copy()

    s3 = ds.source3[
        ds.source3["entity_id"].eq(TARGET_S3)
    ].copy()

    s1 = add_normalized_columns(s1)
    s3 = add_normalized_columns(s3)

    s1_row = s1.iloc[0]
    s3_row = s3.iloc[0]

    print()
    print("=" * 80)
    print("TARGET NAME-BLOCK DIAGNOSTIC")
    print("=" * 80)

    print()
    print("SOURCE 1")
    print("-" * 80)
    print("entity_id:", s1_row["entity_id"])
    print("norm_name:", s1_row["norm_name"])

    print(
        "tokens min_length=4:",
        _safe_tokens(
            s1_row["norm_name"],
            4,
        ),
    )

    print(
        "tokens min_length=3:",
        _safe_tokens(
            s1_row["norm_name"],
            3,
        ),
    )

    print(
        "tokens min_length=2:",
        _safe_tokens(
            s1_row["norm_name"],
            2,
        ),
    )

    print()
    print("SOURCE 3")
    print("-" * 80)
    print("entity_id:", s3_row["entity_id"])
    print("norm_name:", s3_row["norm_name"])

    print(
        "tokens min_length=4:",
        _safe_tokens(
            s3_row["norm_name"],
            4,
        ),
    )

    print(
        "tokens min_length=3:",
        _safe_tokens(
            s3_row["norm_name"],
            3,
        ),
    )

    print(
        "tokens min_length=2:",
        _safe_tokens(
            s3_row["norm_name"],
            2,
        ),
    )

    # ------------------------------------------------------------------
    # Stream through S3 to calculate only token frequencies.
    #
    # We do NOT normalize the complete 5.28M-row dataframe in memory.
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STREAMING S3 TOKEN FREQUENCIES")
    print("=" * 80)

    token_frequency = {}

    s3_path = "dataset/train/train_source3.tsv"

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            s3_path,
            sep="\t",
            dtype=str,
            chunksize=5000,
            keep_default_na=False,
        ),
        start=1,
    ):
        names = chunk["business_name"]

        for value in names:
            # We only need the existing normalization logic for names.
            # Import locally to avoid unnecessary dataframe allocation.
            from src.normalize import normalize_name

            normalized = normalize_name(value)

            for token in _safe_tokens(
                normalized,
                4,
            ):
                token_frequency[token] = (
                    token_frequency.get(token, 0) + 1
                )

        if chunk_no % 100 == 0:
            print(
                f"Processed {chunk_no * 5000:,} S3 rows..."
            )

    target_tokens = sorted(
        set(
            _safe_tokens(
                s1_row["norm_name"],
                4,
            )
        )
        |
        set(
            _safe_tokens(
                s3_row["norm_name"],
                4,
            )
        )
    )

    print()
    print("TOKEN FREQUENCIES")
    print("-" * 80)

    for token in target_tokens:
        print(
            f"{token:20s} -> "
            f"{token_frequency.get(token, 0):,}"
        )

    # ------------------------------------------------------------------
    # Check shorter-token frequencies separately.
    #
    # This is particularly important for "sun".
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SHORT-TOKEN FREQUENCIES")
    print("=" * 80)

    short_tokens = [
        "sun",
        "inc",
        "new",
        "the",
        "and",
        "private",
        "limited",
    ]

    short_frequency = {
        token: 0
        for token in short_tokens
    }

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            s3_path,
            sep="\t",
            dtype=str,
            chunksize=5000,
            keep_default_na=False,
        ),
        start=1,
    ):
        from src.normalize import normalize_name

        for value in chunk["business_name"]:
            normalized = normalize_name(value)

            tokens = set(
                _safe_tokens(
                    normalized,
                    3,
                )
            )

            for token in short_tokens:
                if token in tokens:
                    short_frequency[token] += 1

    for token in short_tokens:
        print(
            f"{token:20s} -> "
            f"{short_frequency[token]:,}"
        )

    print()
    print("=" * 80)
    print("SHORT TOKEN EXPERIMENT")
    print("=" * 80)

    SHORT_TOKEN_MIN_LENGTH = 3
    SHORT_TOKEN_MAX_FREQUENCY = 10000

    print(
        "min_length:",
        SHORT_TOKEN_MIN_LENGTH,
    )

    print(
        "max_frequency:",
        SHORT_TOKEN_MAX_FREQUENCY,
    )

    print()

    for token in ["sun", "inc", "new", "the", "and"]:
        frequency = short_frequency.get(token, 0)

        is_usable = (
            len(token) >= SHORT_TOKEN_MIN_LENGTH
            and frequency <= SHORT_TOKEN_MAX_FREQUENCY
        )

        print(
            f"{token:10s} "
            f"frequency={frequency:,} "
            f"usable={is_usable}"
        )

    print()
    print(
        "Target token 'sun' would be eligible:",
        (
            short_frequency["sun"]
            <= SHORT_TOKEN_MAX_FREQUENCY
        ),
    )

    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    print(
        "The target shared token 'sun' occurs in only "
        f"{short_frequency['sun']:,} S3 records."
    )

    print(
        "This makes it a candidate for a dedicated short-name "
        "blocking strategy without lowering the global token "
        "length threshold."
    )


if __name__ == "__main__":
    main()