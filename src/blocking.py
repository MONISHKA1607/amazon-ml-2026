"""
Multi-pass candidate generation for business entity resolution.

The blocker generates a manageable candidate set for every Source-1 entity
using multiple blocking strategies:

    1. exact normalized name
    2. name tokens
    3. address tokens
    4. character n-grams
    5. numeric/address tokens
    6. country + name prefix

Common blocks are filtered using posting-list frequency limits to prevent
candidate explosion.

Every candidate also records:
    - blocks_matched
    - block_score

These become useful diagnostics and later model features.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Callable

import pandas as pd

from .blocking_config import BlockingConfig


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _safe_tokens(
    value: str | None,
    min_length: int = 4,
) -> list[str]:
    """Return sufficiently long whitespace-separated tokens."""

    if not value:
        return []

    return [
        token
        for token in str(value).split()
        if len(token) >= min_length
    ]


def _char_ngrams_for_block(
    value: str | None,
    n: int,
) -> list[str]:
    """Return character n-grams from a normalized string."""

    if not value:
        return []

    value = str(value)

    if len(value) < n:
        return []

    return [
        value[i : i + n]
        for i in range(len(value) - n + 1)
    ]


def _numeric_tokens_for_block(
    value,
) -> list[str]:
    """
    Normalize the representation of numeric_tokens.

    The normalize module may store numeric tokens as a list/tuple/set
    or as a whitespace-separated string.
    """

    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [
            str(token)
            for token in value
            if str(token)
        ]

    return [
        token
        for token in str(value).split()
        if token
    ]


def _country_name_prefix_for_block(
    country: str | None,
    name: str | None,
    prefix_len: int = 4,
) -> list[str]:
    """
    Create a country-scoped name-prefix blocking key.

    Country is deliberately treated as an open string value. We do not
    hard-code countries here so that unseen test countries such as France
    work automatically.
    """

    if not country or not name:
        return []

    country = str(country).strip().lower()
    name = str(name).strip()

    if not country or not name:
        return []

    prefix = name[:prefix_len]

    if len(prefix) < prefix_len:
        return []

    return [
        f"{country}||{prefix}"
    ]


# ---------------------------------------------------------------------------
# Inverted-index construction
# ---------------------------------------------------------------------------

def _build_inverted_index(
    df: pd.DataFrame,
    key_fn: Callable,
    max_block_frequency: int = 5000,
    max_candidates_per_block: int = 5000,
) -> dict[str, list[str]]:
    """
    Build an inverted index.

    Blocks whose posting lists are larger than max_block_frequency are
    discarded completely.

    This is important for common words/ngrams such as:
        "the"
        "restaurant"
        "company"
        "and"

    which could otherwise generate enormous candidate sets.
    """

    postings: defaultdict[str, list[str]] = defaultdict(list)

    for row in df.itertuples(index=False):
        entity_id = row.entity_id

        keys = key_fn(row)

        if isinstance(keys, str):
            keys = [keys]

        # An entity contributes at most once to a given block.
        for key in set(keys):
            if not key:
                continue

            postings[key].append(entity_id)

    filtered: dict[str, list[str]] = {}

    for key, entity_ids in postings.items():

        # Drop overly-common blocks.
        if len(entity_ids) > max_block_frequency:
            continue

        # Remove accidental duplicates and make ordering deterministic.
        entity_ids = sorted(set(entity_ids))

        # Safety cap.
        filtered[key] = entity_ids[
            :max_candidates_per_block
        ]

    return filtered


def _build_token_frequency(
    df: pd.DataFrame,
    column: str,
    min_token_length: int = 4,
) -> Counter:
    """
    Count the number of entities containing each token.

    Each entity contributes at most once for a particular token.
    """

    counter = Counter()

    for value in df[column].fillna(""):
        tokens = set(
            _safe_tokens(
                value,
                min_token_length,
            )
        )

        counter.update(tokens)

    return counter


def _build_rare_token_index(
    df: pd.DataFrame,
    column: str,
    token_frequency: Counter,
    rare_token_frequency: int,
    min_token_length: int = 4,
) -> dict[str, list[str]]:
    """
    Build an index containing only relatively rare tokens.
    """

    postings: defaultdict[str, list[str]] = defaultdict(list)

    for row in df.itertuples(index=False):

        value = getattr(row, column, "")

        tokens = set(
            _safe_tokens(
                value,
                min_token_length,
            )
        )

        for token in tokens:

            if token_frequency[token] <= rare_token_frequency:
                postings[token].append(
                    row.entity_id
                )

    return {
        token: sorted(set(entity_ids))
        for token, entity_ids in postings.items()
    }


# ---------------------------------------------------------------------------
# Candidate ranking
# ---------------------------------------------------------------------------

def _rank_candidates(
    candidate_scores: dict[str, int],
    max_candidates: int,
) -> list[str]:
    """
    Rank candidates by blocking evidence.

    More blocking agreements means a candidate is supported by more
    independent blocking signals.

    Candidate ID is used as a deterministic tie-breaker.
    """

    ranked = sorted(
        candidate_scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    return [
        candidate_id
        for candidate_id, _ in ranked[:max_candidates]
    ]


# ---------------------------------------------------------------------------
# Main candidate generation
# ---------------------------------------------------------------------------

def generate_candidates(
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    block_names: list[str] | None = None,
    config: BlockingConfig | None = None,
    source_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate candidates between Source 1 and one candidate source.

    Parameters
    ----------
    s1_df:
        Normalized Source-1 dataframe.

    other_df:
        Normalized Source-2 or Source-3 dataframe.

    block_names:
        Blocking strategies to use.

    config:
        Blocking configuration.

    source_name:
        "S2" or "S3", used only for diagnostics.

    Returns
    -------
    candidates:
        DataFrame containing:

            source1_entity_id
            candidate_entity_id
            blocks_matched
            block_score

    stats:
        Candidate-generation diagnostics.
    """

    if config is None:
        config = BlockingConfig()

    if block_names is None:
        block_names = config.enabled_blocks

    if block_names is None:
        block_names = [
            "exact_name",
            "name_tokens",
            "address_tokens",
            "char_ngrams",
            "numeric_tokens",
            "country_name_prefix",
        ]

    # ------------------------------------------------------------------
    # Token frequencies
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Build indices
    # ------------------------------------------------------------------

    indices: dict[str, dict[str, list[str]]] = {}

    if "exact_name" in block_names:

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

    if "name_tokens" in block_names:

        indices["name_tokens"] = _build_inverted_index(
            other_df,
            lambda row: _safe_tokens(
                getattr(row, "norm_name", ""),
                config.min_token_length,
            ),
            config.max_block_frequency,
            config.max_candidates_per_block,
        )

    if "address_tokens" in block_names:

        indices["address_tokens"] = _build_inverted_index(
            other_df,
            lambda row: _safe_tokens(
                getattr(row, "norm_address", ""),
                config.min_token_length,
            ),
            config.max_block_frequency,
            config.max_candidates_per_block,
        )

    if "char_ngrams" in block_names:

        indices["char_ngrams"] = _build_inverted_index(
            other_df,
            lambda row: _char_ngrams_for_block(
                getattr(row, "norm_name", ""),
                config.char_ngram_size,
            ),
            config.max_block_frequency,
            config.max_candidates_per_block,
        )

    if "numeric_tokens" in block_names:

        indices["numeric_tokens"] = _build_inverted_index(
            other_df,
            lambda row: _numeric_tokens_for_block(
                getattr(row, "numeric_tokens", ""),
            ),
            config.max_block_frequency,
            config.max_candidates_per_block,
        )

    if "country_name_prefix" in block_names:

        indices["country_name_prefix"] = _build_inverted_index(
            other_df,
            lambda row: _country_name_prefix_for_block(
                getattr(row, "country", ""),
                getattr(row, "norm_name", ""),
            ),
            config.max_block_frequency,
            config.max_candidates_per_block,
        )

    # ------------------------------------------------------------------
    # Rare-token indices
    # ------------------------------------------------------------------

    rare_name_index: dict[str, list[str]] = {}

    if "name_tokens" in block_names:

        rare_name_index = _build_rare_token_index(
            other_df,
            "norm_name",
            name_frequency,
            config.rare_token_frequency,
            config.min_token_length,
        )

    rare_address_index: dict[str, list[str]] = {}

    if "address_tokens" in block_names:

        rare_address_index = _build_rare_token_index(
            other_df,
            "norm_address",
            address_frequency,
            config.rare_token_frequency,
            config.min_token_length,
        )

    # ------------------------------------------------------------------
    # Generate candidates
    # ------------------------------------------------------------------

    rows = []

    candidate_count_by_s1: dict[str, int] = {}

    for s1_row in s1_df.itertuples(index=False):

        s1_id = s1_row.entity_id

        # candidate_id -> number of blocking agreements
        candidate_scores: defaultdict[str, int] = defaultdict(int)

        # candidate_id -> blocking strategies supporting it
        candidate_blocks: defaultdict[str, set[str]] = defaultdict(set)

        # --------------------------------------------------------------
        # Standard blocking strategies
        # --------------------------------------------------------------

        for block_name in block_names:

            index = indices.get(block_name)

            if index is None:
                continue

            if block_name == "exact_name":

                keys = [
                    getattr(
                        s1_row,
                        "norm_name",
                        "",
                    )
                ]

            elif block_name == "name_tokens":

                keys = _safe_tokens(
                    getattr(
                        s1_row,
                        "norm_name",
                        "",
                    ),
                    config.min_token_length,
                )

            elif block_name == "address_tokens":

                keys = _safe_tokens(
                    getattr(
                        s1_row,
                        "norm_address",
                        "",
                    ),
                    config.min_token_length,
                )

            elif block_name == "char_ngrams":

                keys = _char_ngrams_for_block(
                    getattr(
                        s1_row,
                        "norm_name",
                        "",
                    ),
                    config.char_ngram_size,
                )

            elif block_name == "numeric_tokens":

                keys = _numeric_tokens_for_block(
                    getattr(
                        s1_row,
                        "numeric_tokens",
                        "",
                    ),
                )

            elif block_name == "country_name_prefix":

                keys = _country_name_prefix_for_block(
                    getattr(
                        s1_row,
                        "country",
                        "",
                    ),
                    getattr(
                        s1_row,
                        "norm_name",
                        "",
                    ),
                )

            else:
                continue

            for key in set(keys):

                if not key:
                    continue

                candidate_ids = index.get(
                    key,
                    [],
                )

                for candidate_id in candidate_ids:

                    candidate_scores[
                        candidate_id
                    ] += 1

                    candidate_blocks[
                        candidate_id
                    ].add(
                        block_name
                    )

        # --------------------------------------------------------------
        # Rare name-token evidence
        # --------------------------------------------------------------

        if "name_tokens" in block_names:

            name_tokens = _safe_tokens(
                getattr(
                    s1_row,
                    "norm_name",
                    "",
                ),
                config.min_token_length,
            )

            for token in name_tokens:

                candidate_ids = rare_name_index.get(
                    token,
                    [],
                )

                for candidate_id in candidate_ids:

                    candidate_scores[
                        candidate_id
                    ] += 1

                    candidate_blocks[
                        candidate_id
                    ].add(
                        "rare_name_token"
                    )

        # --------------------------------------------------------------
        # Rare address-token evidence
        # --------------------------------------------------------------

        if "address_tokens" in block_names:

            address_tokens = _safe_tokens(
                getattr(
                    s1_row,
                    "norm_address",
                    "",
                ),
                config.min_token_length,
            )

            for token in address_tokens:

                candidate_ids = rare_address_index.get(
                    token,
                    [],
                )

                for candidate_id in candidate_ids:

                    candidate_scores[
                        candidate_id
                    ] += 1

                    candidate_blocks[
                        candidate_id
                    ].add(
                        "rare_address_token"
                    )

        # --------------------------------------------------------------
        # Candidate ranking and cap
        # --------------------------------------------------------------

        selected_ids = _rank_candidates(
            candidate_scores,
            config.max_candidates_per_entity,
        )

        candidate_count_by_s1[
            s1_id
        ] = len(selected_ids)

        # --------------------------------------------------------------
        # Save candidate rows
        # --------------------------------------------------------------

        for candidate_id in selected_ids:

            rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "blocks_matched": "|".join(
                        sorted(
                            candidate_blocks[
                                candidate_id
                            ]
                        )
                    ),
                    "block_score": candidate_scores[
                        candidate_id
                    ],
                }
            )

    # ------------------------------------------------------------------
    # Candidate dataframe
    # ------------------------------------------------------------------

    candidates = pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "blocks_matched",
            "block_score",
        ],
    )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    if candidate_count_by_s1:

        counts = pd.Series(
            candidate_count_by_s1,
            dtype="int64",
        )

    else:

        counts = pd.Series(
            dtype="int64"
        )

    stats = {
        "source": source_name,
        "n_source1_entities": int(
            len(s1_df)
        ),
        "n_candidate_pairs": int(
            len(candidates)
        ),
        "n_s1_with_candidates": int(
            (counts > 0).sum()
        ),
        "n_s1_without_candidates": int(
            (counts == 0).sum()
        ),
        "mean_candidates_per_s1": float(
            counts.mean()
        ) if len(counts) else 0.0,
        "median_candidates_per_s1": float(
            counts.median()
        ) if len(counts) else 0.0,
        "p90_candidates_per_s1": float(
            counts.quantile(0.90)
        ) if len(counts) else 0.0,
        "p95_candidates_per_s1": float(
            counts.quantile(0.95)
        ) if len(counts) else 0.0,
        "p99_candidates_per_s1": float(
            counts.quantile(0.99)
        ) if len(counts) else 0.0,
        "max_candidates_per_s1": int(
            counts.max()
        ) if len(counts) else 0,
    }

    return candidates, stats


# ---------------------------------------------------------------------------
# Generate S2 + S3 candidates
# ---------------------------------------------------------------------------

def generate_all_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    block_names: list[str] | None = None,
    config: BlockingConfig | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Generate candidates against both Source 2 and Source 3.
    """

    if config is None:
        config = BlockingConfig()

    s2_candidates, s2_stats = generate_candidates(
        s1_df=s1_df,
        other_df=s2_df,
        block_names=block_names,
        config=config,
        source_name="S2",
    )

    s3_candidates, s3_stats = generate_candidates(
        s1_df=s1_df,
        other_df=s3_df,
        block_names=block_names,
        config=config,
        source_name="S3",
    )

    candidates = pd.concat(
        [
            s2_candidates,
            s3_candidates,
        ],
        ignore_index=True,
    )

    if not candidates.empty:

        candidates = candidates.sort_values(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        ).reset_index(
            drop=True
        )

    combined_stats = {
        "s2": s2_stats,
        "s3": s3_stats,
        "n_s1_entities": int(
            len(s1_df)
        ),
        "n_candidate_pairs": int(
            len(candidates)
        ),
    }

    return candidates, combined_stats


# ---------------------------------------------------------------------------
# Blocking recall
# ---------------------------------------------------------------------------

def compute_blocking_recall(
    candidates_df: pd.DataFrame,
    gt_dict: dict,
) -> dict:
    """
    Compute blocking recall against ground truth.

    Only entities with at least one true match contribute to pair recall.

    Returns:
        blocking_recall
        total_true_pairs
        total_found_pairs
        n_entities_with_missed_matches
        missed_examples
        candidate_pair_count
        avg_candidates_per_s1
    """

    candidate_map: defaultdict[str, set[str]] = defaultdict(set)

    if not candidates_df.empty:

        for s1_id, candidate_id in zip(
            candidates_df[
                "source1_entity_id"
            ],
            candidates_df[
                "candidate_entity_id"
            ],
        ):

            candidate_map[
                s1_id
            ].add(
                candidate_id
            )

    total_true = 0
    total_found = 0

    missed_entities = []

    for s1_id, true_ids in gt_dict.items():

        if not true_ids:
            continue

        found = (
            true_ids
            & candidate_map.get(
                s1_id,
                set(),
            )
        )

        total_true += len(true_ids)
        total_found += len(found)

        if found != true_ids:

            missed_entities.append(
                (
                    s1_id,
                    true_ids - found,
                )
            )

    recall = (
        total_found / total_true
        if total_true
        else 1.0
    )

    candidate_counts = (
        candidates_df
        .groupby(
            "source1_entity_id"
        )
        .size()
        if not candidates_df.empty
        else pd.Series(
            dtype="int64"
        )
    )

    return {
        "blocking_recall": recall,
        "total_true_pairs": total_true,
        "total_found_pairs": total_found,
        "n_entities_with_missed_matches": len(
            missed_entities
        ),
        "missed_examples": missed_entities[
            :20
        ],
        "candidate_pair_count": len(
            candidates_df
        ),
        "avg_candidates_per_s1": (
            float(
                candidate_counts.mean()
            )
            if len(candidate_counts)
            else 0.0
        ),
        "median_candidates_per_s1": (
            float(
                candidate_counts.median()
            )
            if len(candidate_counts)
            else 0.0
        ),
        "p95_candidates_per_s1": (
            float(
                candidate_counts.quantile(
                    0.95
                )
            )
            if len(candidate_counts)
            else 0.0
        ),
        "max_candidates_per_s1": (
            int(
                candidate_counts.max()
            )
            if len(candidate_counts)
            else 0
        ),
    }