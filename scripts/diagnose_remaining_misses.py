import sys

sys.path.insert(0, ".")

from src.data_loader import load_split
from src.normalize import add_normalized_columns
from src.blocking import (
    _build_inverted_index,
    _build_token_frequency,
    _build_rare_token_index,
    _safe_tokens,
    _char_ngrams_for_block,
    _numeric_tokens_for_block,
    _country_name_prefix_for_block,
    _address_numeric_anchor_keys,
)
from src.blocking_config import BlockingConfig


MISSES = {
    "S1-312975533": "S3-688080907",
    "S1-833436524": "S2-910790078",
}


def inspect_pair(s1_row, target_row, source_name, config):
    print("\n" + "=" * 90)
    print(f"TARGET: {s1_row['entity_id']} -> {target_row['entity_id']}")
    print("=" * 90)

    print(f"\nSource-1 name:      {s1_row['norm_name']}")
    print(f"Source-{source_name[-1]} name:      {target_row['norm_name']}")

    print(f"\nSource-1 address:   {s1_row['norm_address']}")
    print(f"Source-{source_name[-1]} address:   {target_row['norm_address']}")

    print(f"\nSource-1 translit name:    {s1_row['translit_name']}")
    print(f"Target translit name:      {target_row['translit_name']}")

    print(f"\nSource-1 translit address: {s1_row['translit_address']}")
    print(f"Target translit address:   {target_row['translit_address']}")

    s1_name_tokens = set(
        _safe_tokens(s1_row["norm_name"], config.min_token_length)
    )
    target_name_tokens = set(
        _safe_tokens(target_row["norm_name"], config.min_token_length)
    )

    s1_addr_tokens = set(
        _safe_tokens(s1_row["norm_address"], config.min_token_length)
    )
    target_addr_tokens = set(
        _safe_tokens(target_row["norm_address"], config.min_token_length)
    )

    s1_translit_name_tokens = set(
        _safe_tokens(
            s1_row["translit_name"],
            config.min_token_length,
        )
    )
    target_translit_name_tokens = set(
        _safe_tokens(
            target_row["translit_name"],
            config.min_token_length,
        )
    )

    s1_translit_addr_tokens = set(
        _safe_tokens(
            s1_row["translit_address"],
            config.min_token_length,
        )
    )
    target_translit_addr_tokens = set(
        _safe_tokens(
            target_row["translit_address"],
            config.min_token_length,
        )
    )

    print("\nTOKEN OVERLAPS")
    print("-" * 90)

    print(
        "name tokens:",
        sorted(s1_name_tokens & target_name_tokens),
    )

    print(
        "address tokens:",
        sorted(s1_addr_tokens & target_addr_tokens),
    )

    print(
        "translit name tokens:",
        sorted(s1_translit_name_tokens & target_translit_name_tokens),
    )

    print(
        "translit address tokens:",
        sorted(
            s1_translit_addr_tokens & target_translit_addr_tokens
        ),
    )

    s1_name_ngrams = set(
        _char_ngrams_for_block(
            s1_row["norm_name"],
            config.char_ngram_size,
        )
    )
    target_name_ngrams = set(
        _char_ngrams_for_block(
            target_row["norm_name"],
            config.char_ngram_size,
        )
    )

    s1_translit_name_ngrams = set(
        _char_ngrams_for_block(
            s1_row["translit_name"],
            config.char_ngram_size,
        )
    )
    target_translit_name_ngrams = set(
        _char_ngrams_for_block(
            target_row["translit_name"],
            config.char_ngram_size,
        )
    )

    print("\nCHAR-NGRAM OVERLAPS")
    print("-" * 90)

    print(
        "name char ngrams:",
        len(s1_name_ngrams & target_name_ngrams),
        sorted(s1_name_ngrams & target_name_ngrams),
    )

    print(
        "translit name char ngrams:",
        len(
            s1_translit_name_ngrams
            & target_translit_name_ngrams
        ),
        sorted(
            s1_translit_name_ngrams
            & target_translit_name_ngrams
        ),
    )

    s1_numeric = set(
        _numeric_tokens_for_block(s1_row["numeric_tokens"])
    )
    target_numeric = set(
        _numeric_tokens_for_block(
            target_row["numeric_tokens"]
        )
    )

    print("\nNUMERIC TOKENS")
    print("-" * 90)
    print("Source-1:", sorted(s1_numeric))
    print("Target:  ", sorted(target_numeric))
    print("Overlap: ", sorted(s1_numeric & target_numeric))

    s1_country = _country_name_prefix_for_block(
        s1_row["country"],
        s1_row["norm_name"],
    )

    target_country = _country_name_prefix_for_block(
        target_row["country"],
        target_row["norm_name"],
    )

    print("\nCOUNTRY / NAME PREFIX")
    print("-" * 90)
    print("Source-1:", s1_country)
    print("Target:  ", target_country)

    s1_anchor = set(
        _address_numeric_anchor_keys(
            s1_row["norm_address"],
            s1_row["numeric_tokens"],
            config.address_anchor_min_token_length,
        )
    )

    target_anchor = set(
        _address_numeric_anchor_keys(
            target_row["norm_address"],
            target_row["numeric_tokens"],
            config.address_anchor_min_token_length,
        )
    )

    print("\nADDRESS NUMERIC ANCHORS")
    print("-" * 90)
    print("Source-1:", sorted(s1_anchor))
    print("Target:  ", sorted(target_anchor))
    print("Overlap: ", sorted(s1_anchor & target_anchor))

    print("\nSUMMARY")
    print("-" * 90)

    checks = {
        "exact_name": (
            s1_row["norm_name"] == target_row["norm_name"]
            and bool(s1_row["norm_name"])
        ),
        "name_tokens": bool(
            s1_name_tokens & target_name_tokens
        ),
        "address_tokens": bool(
            s1_addr_tokens & target_addr_tokens
        ),
        "char_ngrams": bool(
            s1_name_ngrams & target_name_ngrams
        ),
        "numeric_tokens": bool(
            s1_numeric & target_numeric
        ),
        "country_name_prefix": bool(
            s1_country
            and target_country
            and s1_country == target_country
        ),
        "translit_exact_name": (
            s1_row["translit_name"]
            == target_row["translit_name"]
            and bool(s1_row["translit_name"])
        ),
        "translit_name_tokens": bool(
            s1_translit_name_tokens
            & target_translit_name_tokens
        ),
        "translit_char_ngrams": bool(
            s1_translit_name_ngrams
            & target_translit_name_ngrams
        ),
        "translit_address_tokens": bool(
            s1_translit_addr_tokens
            & target_translit_addr_tokens
        ),
        "address_numeric_anchor": bool(
            s1_anchor & target_anchor
        ),
    }

    for block, matched in checks.items():
        print(f"{block:28s}: {'YES' if matched else 'NO'}")


def main():
    print("Loading training data...")
    ds = load_split("dataset", "train")

    s1_ids = list(MISSES.keys())
    s2_id = "S2-910790078"
    s3_id = "S3-688080907"

    s1 = ds.source1[
        ds.source1["entity_id"].isin(s1_ids)
    ].copy()

    s2 = ds.source2[
        ds.source2["entity_id"].eq(s2_id)
    ].copy()

    s3 = ds.source3[
        ds.source3["entity_id"].eq(s3_id)
    ].copy()

    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)

    config = BlockingConfig(
        max_block_frequency=500,
        max_candidates_per_block=500,
        max_candidates_per_entity=500,
    )

    s1_row = s1.iloc[0]
    s1_row_2 = s1.iloc[1]

    inspect_pair(
        s1_row,
        s3.iloc[0],
        "source3",
        config,
    )

    inspect_pair(
        s1_row_2,
        s2.iloc[0],
        "source2",
        config,
    )


if __name__ == "__main__":
    main()