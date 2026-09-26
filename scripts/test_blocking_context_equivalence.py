#!/usr/bin/env python3

import json
import os
import sys

import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
    ),
)

from src.blocking_config import BlockingConfig
from src.data_loader import load_split
from src.normalize import add_normalized_columns
from src.blocking import (
    generate_candidates,
    build_blocking_context,
    generate_candidates_batch,
)


def main():
    with open(
        "configs/blocking_v2.json",
        "r",
        encoding="utf-8",
    ) as f:
        config_data = json.load(f)

    config = BlockingConfig(
        version=config_data["version"],
        enabled_blocks=config_data["enabled_blocks"],
        min_token_length=config_data["min_token_length"],
        max_block_frequency=config_data["max_block_frequency"],
        max_candidates_per_block=config_data["max_candidates_per_block"],
        max_candidates_per_entity=config_data["max_candidates_per_entity"],
        char_ngram_size=config_data["char_ngram_size"],
        rare_token_frequency=config_data["rare_token_frequency"],
        address_anchor_min_token_length=config_data[
            "address_anchor_min_token_length"
        ],
        short_name_token_min_length=config_data[
            "short_name_token_min_length"
        ],
        short_name_token_max_frequency=config_data[
            "short_name_token_max_frequency"
        ],
    )

    ds = load_split(
        "dataset",
        "train",
    )

    # Small deterministic sample.
    s1 = add_normalized_columns(
        ds.source1.head(100)
    )

    s2 = add_normalized_columns(
        ds.source2.head(5000)
    )

    block_names = config.enabled_blocks

    # --------------------------------------------------------------
    # Old path: build indices inside generate_candidates()
    # --------------------------------------------------------------

    old_candidates, old_stats = generate_candidates(
        s1_df=s1,
        other_df=s2,
        block_names=block_names,
        config=config,
        source_name="S2",
    )

    # --------------------------------------------------------------
    # New path: build once, then generate from context
    # --------------------------------------------------------------

    context = build_blocking_context(
        other_df=s2,
        block_names=block_names,
        config=config,
    )

    new_candidates, new_stats = generate_candidates_batch(
        s1_batch_df=s1,
        blocking_context=context,
        config=config,
        source_name="S2",
    )

    old_set = set(
        zip(
            old_candidates["source1_entity_id"],
            old_candidates["candidate_entity_id"],
            old_candidates["blocks_matched"],
            old_candidates["block_score"],
        )
    )

    new_set = set(
        zip(
            new_candidates["source1_entity_id"],
            new_candidates["candidate_entity_id"],
            new_candidates["blocks_matched"],
            new_candidates["block_score"],
        )
    )

    print(f"old rows: {len(old_candidates):,}")
    print(f"new rows: {len(new_candidates):,}")
    print(f"old unique rows: {len(old_set):,}")
    print(f"new unique rows: {len(new_set):,}")

    print(
        f"rows only in old: "
        f"{len(old_set - new_set):,}"
    )

    print(
        f"rows only in new: "
        f"{len(new_set - old_set):,}"
    )

    if old_set != new_set:
        print("\nERROR: candidate sets differ.")

        print("\nExamples only in old:")
        for row in list(old_set - new_set)[:10]:
            print(row)

        print("\nExamples only in new:")
        for row in list(new_set - old_set)[:10]:
            print(row)

        raise SystemExit(1)

    print("\nPASS: old and new candidate sets are identical.")


if __name__ == "__main__":
    main()