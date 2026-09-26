#!/usr/bin/env python3

import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
    ),
)

from src.data_loader import load_split  # noqa: E402
from src.normalize import add_normalized_columns  # noqa: E402
from src.blocking_config import BlockingConfig  # noqa: E402
from src.blocking import (  # noqa: E402
    _build_inverted_index,
    _build_rare_token_index,
    _build_token_frequency,
    _safe_tokens,
    _char_ngrams_for_block,
    _numeric_tokens_for_block,
    _country_name_prefix_for_block,
    _address_numeric_anchor_keys,
)


MISSES = [
    ("S1-833436524", "S2-910790078"),
    ("S1-622661924", "S2-161277479"),
    ("S1-394340130", "S2-802580138"),
    ("S1-906812446", "S3-9926688"),
    ("S1-196088436", "S2-586372006"),
    ("S1-86409640", "S3-38031351"),
    ("S1-569884173", "S3-4099910"),
    ("S1-361139890", "S3-89215837"),
    ("S1-361139890", "S3-900447172"),
]


def build_indices(other_df, config):
    name_frequency = _build_token_frequency(
        other_df,
        "norm_name",
        config.min_token_length,
    )

    address_frequency = _build_token_frequency(
        other_df,
        "norm_address",
        config.min_token_length,
    )

    indices = {}

    indices["exact_name"] = _build_inverted_index(
        other_df,
        lambda row: (
            [row.norm_name]
            if getattr(row, "norm_name", "")
            else []
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    indices["name_tokens"] = _build_inverted_index(
        other_df,
        lambda row: _safe_tokens(
            getattr(row, "norm_name", ""),
            config.min_token_length,
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    indices["short_name_tokens"] = _build_inverted_index(
        other_df,
        lambda row: _safe_tokens(
            getattr(row, "norm_name", ""),
            config.short_name_token_min_length,
        ),
        config.short_name_token_max_frequency,
        config.max_candidates_per_block,
    )

    indices["address_tokens"] = _build_inverted_index(
        other_df,
        lambda row: _safe_tokens(
            getattr(row, "norm_address", ""),
            config.min_token_length,
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    indices["char_ngrams"] = _build_inverted_index(
        other_df,
        lambda row: _char_ngrams_for_block(
            getattr(row, "norm_name", ""),
            config.char_ngram_size,
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    indices["numeric_tokens"] = _build_inverted_index(
        other_df,
        lambda row: _numeric_tokens_for_block(
            getattr(row, "numeric_tokens", ""),
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    indices["country_name_prefix"] = _build_inverted_index(
        other_df,
        lambda row: _country_name_prefix_for_block(
            getattr(row, "country", ""),
            getattr(row, "norm_name", ""),
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    indices["translit_exact_name"] = _build_inverted_index(
        other_df,
        lambda row: (
            [row.translit_name]
            if getattr(row, "translit_name", "")
            else []
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    indices["translit_name_tokens"] = _build_inverted_index(
        other_df,
        lambda row: _safe_tokens(
            getattr(row, "translit_name", ""),
            config.min_token_length,
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    indices["translit_char_ngrams"] = _build_inverted_index(
        other_df,
        lambda row: _char_ngrams_for_block(
            getattr(row, "translit_name", ""),
            config.char_ngram_size,
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    indices["translit_address_tokens"] = _build_inverted_index(
        other_df,
        lambda row: _safe_tokens(
            getattr(row, "translit_address", ""),
            config.min_token_length,
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    indices["address_numeric_anchor"] = _build_inverted_index(
        other_df,
        lambda row: _address_numeric_anchor_keys(
            getattr(row, "norm_address", ""),
            getattr(row, "numeric_tokens", ""),
            config.address_anchor_min_token_length,
        ),
        config.max_block_frequency,
        config.max_candidates_per_block,
    )

    rare_name_index = _build_rare_token_index(
        other_df,
        "norm_name",
        name_frequency,
        config.rare_token_frequency,
        config.min_token_length,
    )

    rare_address_index = _build_rare_token_index(
        other_df,
        "norm_address",
        address_frequency,
        config.rare_token_frequency,
        config.min_token_length,
    )

    translit_address_frequency = _build_token_frequency(
        other_df,
        "translit_address",
        config.min_token_length,
    )

    rare_translit_address_index = _build_rare_token_index(
        other_df,
        "translit_address",
        translit_address_frequency,
        config.rare_token_frequency,
        config.min_token_length,
    )

    return (
        indices,
        rare_name_index,
        rare_address_index,
        rare_translit_address_index,
    )


def target_block_keys(row, config):
    return {
        "exact_name": (
            [row.norm_name]
            if row.norm_name
            else []
        ),
        "name_tokens": _safe_tokens(
            row.norm_name,
            config.min_token_length,
        ),
        "short_name_tokens": _safe_tokens(
            row.norm_name,
            config.short_name_token_min_length,
        ),
        "address_tokens": _safe_tokens(
            row.norm_address,
            config.min_token_length,
        ),
        "char_ngrams": _char_ngrams_for_block(
            row.norm_name,
            config.char_ngram_size,
        ),
        "numeric_tokens": _numeric_tokens_for_block(
            row.numeric_tokens,
        ),
        "country_name_prefix": _country_name_prefix_for_block(
            row.country,
            row.norm_name,
        ),
        "translit_exact_name": (
            [row.translit_name]
            if row.translit_name
            else []
        ),
        "translit_name_tokens": _safe_tokens(
            row.translit_name,
            config.min_token_length,
        ),
        "translit_char_ngrams": _char_ngrams_for_block(
            row.translit_name,
            config.char_ngram_size,
        ),
        "translit_address_tokens": _safe_tokens(
            row.translit_address,
            config.min_token_length,
        ),
        "address_numeric_anchor": _address_numeric_anchor_keys(
            row.norm_address,
            row.numeric_tokens,
            config.address_anchor_min_token_length,
        ),
    }


def main():
    print("Loading training data...")

    ds = load_split(
        "dataset",
        "train",
    )

    print("Loading target S1 records...")

    s1_ids = {
        s1_id
        for s1_id, _ in MISSES
    }

    s1 = ds.source1[
        ds.source1["entity_id"].isin(s1_ids)
    ].copy()

    s1 = add_normalized_columns(s1)

    config = BlockingConfig(
        max_block_frequency=500,
        max_candidates_per_block=500,
        max_candidates_per_entity=500,
    )

    for source_name, source_df, targets in [
        (
            "S2",
            ds.source2,
            {
                candidate
                for _, candidate in MISSES
                if candidate.startswith("S2-")
            },
        ),
        (
            "S3",
            ds.source3,
            {
                candidate
                for _, candidate in MISSES
                if candidate.startswith("S3-")
            },
        ),
    ]:

        if not targets:
            continue

        print()
        print("=" * 80)
        print(
            f"Building {source_name} blocking indices"
        )
        print("=" * 80)

        other = add_normalized_columns(
            source_df
        )

        (
            indices,
            rare_name_index,
            rare_address_index,
            rare_translit_address_index,
        ) = build_indices(
            other,
            config,
        )

        target_lookup = {
            row.entity_id: row
            for row in other.itertuples(
                index=False
            )
            if row.entity_id in targets
        }

        for s1_id, target_id in MISSES:

            if not target_id.startswith(
                source_name
            ):
                continue

            s1_row = s1[
                s1["entity_id"].eq(s1_id)
            ].iloc[0]

            target_row = target_lookup[
                target_id
            ]

            keys_by_block = target_block_keys(
                s1_row,
                config,
            )

            candidate_scores = defaultdict(int)
            candidate_blocks = defaultdict(set)

            for block_name, keys in keys_by_block.items():

                index = indices.get(
                    block_name
                )

                if index is None:
                    continue

                for key in set(keys):

                    if not key:
                        continue

                    for candidate_id in index.get(
                        key,
                        [],
                    ):

                        candidate_scores[
                            candidate_id
                        ] += 1

                        candidate_blocks[
                            candidate_id
                        ].add(
                            block_name
                        )

            # Rare-name evidence.
            for token in _safe_tokens(
                s1_row.norm_name,
                config.min_token_length,
            ):

                for candidate_id in rare_name_index.get(
                    token,
                    [],
                ):

                    candidate_scores[
                        candidate_id
                    ] += 1

                    candidate_blocks[
                        candidate_id
                    ].add(
                        "rare_name_token"
                    )

            # Rare-address evidence.
            for token in _safe_tokens(
                s1_row.norm_address,
                config.min_token_length,
            ):

                for candidate_id in rare_address_index.get(
                    token,
                    [],
                ):

                    candidate_scores[
                        candidate_id
                    ] += 1

                    candidate_blocks[
                        candidate_id
                    ].add(
                        "rare_address_token"
                    )

            # Rare transliterated-address evidence.
            for token in _safe_tokens(
                s1_row.translit_address,
                config.min_token_length,
            ):

                for candidate_id in rare_translit_address_index.get(
                    token,
                    [],
                ):

                    candidate_scores[
                        candidate_id
                    ] += 1

                    candidate_blocks[
                        candidate_id
                    ].add(
                        "rare_translit_address_token"
                    )

            target_score = candidate_scores.get(
                target_id,
                0,
            )

            target_blocks = sorted(
                candidate_blocks.get(
                    target_id,
                    set(),
                )
            )

            ranked = sorted(
                candidate_scores.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )

            rank_lookup = {
                candidate_id: rank + 1
                for rank, (
                    candidate_id,
                    _,
                ) in enumerate(ranked)
            }

            target_rank = rank_lookup.get(
                target_id
            )

            print()
            print("-" * 80)
            print(
                f"{s1_id} -> {target_id}"
            )
            print("-" * 80)

            print(
                f"Target block score: {target_score}"
            )

            print(
                "Target blocks: "
                + (
                    "|".join(target_blocks)
                    if target_blocks
                    else "NONE"
                )
            )

            print(
                f"Unbounded rank: "
                f"{target_rank}"
            )

            print(
                "Target generated by "
                "blocking evidence: "
                f"{target_id in candidate_scores}"
            )

            if target_score:
                stronger = sum(
                    1
                    for candidate_id, score
                    in ranked
                    if score > target_score
                )

                print(
                    f"Candidates with stronger "
                    f"raw block score: {stronger:,}"
                )

            print(
                f"Target numeric tokens: "
                f"{target_row.numeric_tokens}"
            )

            print(
                f"Target normalized address: "
                f"{target_row.norm_address}"
            )


if __name__ == "__main__":
    main()