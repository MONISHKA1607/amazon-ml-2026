#!/usr/bin/env python3

import json
import os
import sys

import numpy as np
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
from src.blocking import generate_candidates
from src.features import (
    compute_features,
    compute_features_batch,
)


def load_config():
    with open(
        "configs/blocking_v2.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    return BlockingConfig(
        version=data["version"],
        enabled_blocks=data["enabled_blocks"],
        min_token_length=data["min_token_length"],
        max_block_frequency=data["max_block_frequency"],
        max_candidates_per_block=data[
            "max_candidates_per_block"
        ],
        max_candidates_per_entity=data[
            "max_candidates_per_entity"
        ],
        char_ngram_size=data[
            "char_ngram_size"
        ],
        rare_token_frequency=data[
            "rare_token_frequency"
        ],
        address_anchor_min_token_length=data[
            "address_anchor_min_token_length"
        ],
        short_name_token_min_length=data[
            "short_name_token_min_length"
        ],
        short_name_token_max_frequency=data[
            "short_name_token_max_frequency"
        ],
    )


def main():
    config = load_config()

    ds = load_split(
        "dataset",
        "train",
    )

    s1 = add_normalized_columns(
        ds.source1.head(100)
    )

    s2 = add_normalized_columns(
        ds.source2.head(5000)
    )

    candidates, _ = generate_candidates(
        s1_df=s1,
        other_df=s2,
        block_names=config.enabled_blocks,
        config=config,
        source_name="S2",
    )

    # Keep the comparison manageable.
    candidates = candidates.head(
        5000
    ).reset_index(drop=True)

    old_features = compute_features(
        candidates,
        s1,
        s2,
    )

    referenced_s1 = s1[
        s1["entity_id"].isin(
            candidates[
                "source1_entity_id"
            ]
        )
    ].copy()

    referenced_s2 = s2[
        s2["entity_id"].isin(
            candidates[
                "candidate_entity_id"
            ]
        )
    ].copy()

    new_features = compute_features_batch(
        candidates,
        referenced_s1,
        referenced_s2,
    )

    feature_columns = [
        "name_ratio",
        "name_partial_ratio",
        "name_token_sort_ratio",
        "name_token_set_ratio",
        "addr_ratio",
        "addr_token_sort_ratio",
        "name_exact",
        "addr_exact",
        "country_match",
        "name_len_diff",
        "addr_len_diff",
        "name_token_jaccard",
        "addr_token_jaccard",
        "numeric_overlap",
    ]

    for column in feature_columns:
        old_values = old_features[
            column
        ].to_numpy()

        new_values = new_features[
            column
        ].to_numpy()

        if not np.allclose(
            old_values,
            new_values,
            equal_nan=True,
        ):
            print(
                f"ERROR: mismatch in {column}"
            )
            raise SystemExit(1)

    print(
        f"Compared {len(candidates):,} candidate pairs."
    )

    print(
        "All non-TF-IDF feature columns match exactly."
    )

    print(
        "\nPASS: batch feature computation "
        "matches the existing implementation."
    )


if __name__ == "__main__":
    main()