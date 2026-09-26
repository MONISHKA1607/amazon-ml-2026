#!/usr/bin/env python3

import json
import os
import sys

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
    iter_candidate_batches,
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
        ds.source1.head(200)
    )

    s2 = add_normalized_columns(
        ds.source2.head(5000)
    )

    # --------------------------------------------------------------
    # Generate everything in one call.
    # --------------------------------------------------------------

    full_candidates, _ = generate_candidates(
        s1_df=s1,
        other_df=s2,
        block_names=config.enabled_blocks,
        config=config,
        source_name="S2",
    )

    full_set = set(
        zip(
            full_candidates[
                "source1_entity_id"
            ],
            full_candidates[
                "candidate_entity_id"
            ],
            full_candidates[
                "blocks_matched"
            ],
            full_candidates[
                "block_score"
            ],
        )
    )

    # --------------------------------------------------------------
    # Generate through batches.
    # --------------------------------------------------------------

    batch_set = set()
    batch_count = 0

    for start, end, candidates, stats in iter_candidate_batches(
        s1_df=s1,
        other_df=s2,
        batch_size=50,
        block_names=config.enabled_blocks,
        config=config,
        source_name="S2",
    ):
        batch_count += 1

        print(
            f"batch {batch_count}: "
            f"S1 rows {start}:{end}, "
            f"candidates={len(candidates):,}"
        )

        batch_set.update(
            zip(
                candidates[
                    "source1_entity_id"
                ],
                candidates[
                    "candidate_entity_id"
                ],
                candidates[
                    "blocks_matched"
                ],
                candidates[
                    "block_score"
                ],
            )
        )

    print()
    print(
        f"full candidate rows: "
        f"{len(full_set):,}"
    )

    print(
        f"batched candidate rows: "
        f"{len(batch_set):,}"
    )

    print(
        f"only in full: "
        f"{len(full_set - batch_set):,}"
    )

    print(
        f"only in batches: "
        f"{len(batch_set - full_set):,}"
    )

    if full_set != batch_set:
        print(
            "\nERROR: batched generation "
            "does not match full generation."
        )
        raise SystemExit(1)

    print(
        "\nPASS: batched generation is "
        "identical to full generation."
    )


if __name__ == "__main__":
    main()