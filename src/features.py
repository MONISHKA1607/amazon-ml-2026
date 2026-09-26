"""
features.py
Pairwise feature engineering for candidate pairs.

Runtime rule (non-negotiable): never loop `for s1: for s2: levenshtein(...)`.
- String similarity uses RapidFuzz's C-backed batch functions.
- TF-IDF similarity uses sparse-matrix row dot products (vectors are L2
  normalized by TfidfVectorizer, so dot product == cosine similarity),
  computed only for the candidate pairs we actually have — never a full
  S1 x S2 dense similarity matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import extract_numeric_tokens


def _build_lookup(df: pd.DataFrame, cols):
    """entity_id -> dict(col -> value), for O(1) row access when joining features to candidate pairs."""
    return {eid: {c: row[c] for c in cols} for eid, row in zip(df["entity_id"], df[cols].to_dict("records"))}


def build_tfidf_vectorizer(
    *fit_series,
    analyzer="char_wb",
    ngram_range=(2, 4),
):
    """
    Fit one shared TF-IDF vectorizer on the supplied training-side text.

    The caller can then use the returned vectorizer to transform any
    train/validation/test series without refitting it.
    """
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=1,
    )

    all_text = pd.concat(
        fit_series,
        ignore_index=True,
    )

    vectorizer.fit(all_text)

    return vectorizer


def build_tfidf_matrix(
    *name_series,
    analyzer="char_wb",
    ngram_range=(2, 4),
):
    """
    Fit one shared TF-IDF vectorizer across all supplied series and
    transform each series.

    Kept for compatibility with existing callers.
    """
    vectorizer = build_tfidf_vectorizer(
        *name_series,
        analyzer=analyzer,
        ngram_range=ngram_range,
    )

    return vectorizer, [
        vectorizer.transform(series)
        for series in name_series
    ]


def compute_features(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    name_tfidf_vectorizer=None,
    addr_tfidf_vectorizer=None,
) -> pd.DataFrame:
    """
    candidates_df: source1_entity_id, candidate_entity_id[, blocks_matched]
    Returns candidates_df with feature columns appended.
    """
    cols = ["norm_name", "norm_address", "country", "numeric_tokens"]
    s1_lookup = _build_lookup(s1_df, cols)
    other_lookup = _build_lookup(other_df, cols)

    s1_names, cand_names = [], []
    s1_addrs, cand_addrs = [], []
    s1_countries, cand_countries = [], []
    s1_nums, cand_nums = [], []

    for s1_id, cid in zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]):
        a, b = s1_lookup[s1_id], other_lookup[cid]
        s1_names.append(a["norm_name"]); cand_names.append(b["norm_name"])
        s1_addrs.append(a["norm_address"]); cand_addrs.append(b["norm_address"])
        s1_countries.append(a["country"]); cand_countries.append(b["country"])
        s1_nums.append(a["numeric_tokens"]); cand_nums.append(b["numeric_tokens"])

    n = len(candidates_df)
    feats = pd.DataFrame(index=candidates_df.index)

    # --- RapidFuzz string similarities (batched C loop, not Python loop) ---
    feats["name_ratio"] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(s1_names, cand_names)]
    feats["name_partial_ratio"] = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_names, cand_names)]
    feats["name_token_sort_ratio"] = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_names, cand_names)]
    feats["name_token_set_ratio"] = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_names, cand_names)]
    feats["addr_ratio"] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(s1_addrs, cand_addrs)]
    feats["addr_token_sort_ratio"] = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_addrs, cand_addrs)]

    # --- exact / structural ---
    feats["name_exact"] = [int(a == b and a != "") for a, b in zip(s1_names, cand_names)]
    feats["addr_exact"] = [int(a == b and a != "") for a, b in zip(s1_addrs, cand_addrs)]
    feats["country_match"] = [int(a == b) for a, b in zip(s1_countries, cand_countries)]
    feats["name_len_diff"] = [abs(len(a) - len(b)) for a, b in zip(s1_names, cand_names)]
    feats["addr_len_diff"] = [abs(len(a) - len(b)) for a, b in zip(s1_addrs, cand_addrs)]

    def token_jaccard(a, b):
        ta, tb = set(a.split()), set(b.split())
        if not ta and not tb:
            return 0.0
        return len(ta & tb) / max(1, len(ta | tb))

    feats["name_token_jaccard"] = [token_jaccard(a, b) for a, b in zip(s1_names, cand_names)]
    feats["addr_token_jaccard"] = [token_jaccard(a, b) for a, b in zip(s1_addrs, cand_addrs)]

    def numeric_overlap(a, b):
        sa, sb = set(a.split(",")) - {""}, set(b.split(",")) - {""}
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    feats["numeric_overlap"] = [numeric_overlap(a, b) for a, b in zip(s1_nums, cand_nums)]

    # --- TF-IDF cosine similarity (sparse dot product, vectorized) ---
    if name_tfidf_vectorizer is not None:
        v1 = name_tfidf_vectorizer.transform(s1_names)
        v2 = name_tfidf_vectorizer.transform(cand_names)
        feats["name_tfidf_cosine"] = np.asarray(v1.multiply(v2).sum(axis=1)).ravel()
    if addr_tfidf_vectorizer is not None:
        v1 = addr_tfidf_vectorizer.transform(s1_addrs)
        v2 = addr_tfidf_vectorizer.transform(cand_addrs)
        feats["addr_tfidf_cosine"] = np.asarray(v1.multiply(v2).sum(axis=1)).ravel()

    return pd.concat([candidates_df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

def compute_features_batch(
    candidates_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    other_df: pd.DataFrame,
    name_tfidf_vectorizer=None,
    addr_tfidf_vectorizer=None,
) -> pd.DataFrame:
    """
    Compute pairwise features for a candidate batch.

    Unlike compute_features(), this function expects s1_df and other_df
    to contain only the entity rows referenced by candidates_df.

    This keeps memory usage bounded during chunked training/inference.

    Feature definitions intentionally match compute_features().
    """

    if candidates_df.empty:
        return candidates_df.copy()

    cols = [
        "entity_id",
        "norm_name",
        "norm_address",
        "country",
        "numeric_tokens",
    ]

    s1_lookup = (
        s1_df[
            cols
        ]
        .drop_duplicates("entity_id")
        .set_index("entity_id")
    )

    other_lookup = (
        other_df[
            cols
        ]
        .drop_duplicates("entity_id")
        .set_index("entity_id")
    )

    s1_ids = candidates_df[
        "source1_entity_id"
    ].tolist()

    candidate_ids = candidates_df[
        "candidate_entity_id"
    ].tolist()

    # Reindex preserves candidate-row order.
    s1_rows = s1_lookup.reindex(s1_ids)
    other_rows = other_lookup.reindex(candidate_ids)

    if s1_rows.isna().all(axis=1).any():
        raise ValueError(
            "Candidate batch contains Source-1 IDs that are missing "
            "from s1_df."
        )

    if other_rows.isna().all(axis=1).any():
        raise ValueError(
            "Candidate batch contains candidate IDs that are missing "
            "from other_df."
        )

    s1_names = (
        s1_rows["norm_name"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    cand_names = (
        other_rows["norm_name"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    s1_addrs = (
        s1_rows["norm_address"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    cand_addrs = (
        other_rows["norm_address"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    s1_countries = (
        s1_rows["country"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    cand_countries = (
        other_rows["country"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    s1_nums = (
        s1_rows["numeric_tokens"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    cand_nums = (
        other_rows["numeric_tokens"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    feats = pd.DataFrame(
        index=range(
            len(candidates_df)
        )
    )

    # --------------------------------------------------------------
    # RapidFuzz similarities
    # --------------------------------------------------------------

    feats["name_ratio"] = [
        fuzz.ratio(a, b) / 100.0
        for a, b in zip(
            s1_names,
            cand_names,
        )
    ]

    feats["name_partial_ratio"] = [
        fuzz.partial_ratio(a, b) / 100.0
        for a, b in zip(
            s1_names,
            cand_names,
        )
    ]

    feats["name_token_sort_ratio"] = [
        fuzz.token_sort_ratio(a, b) / 100.0
        for a, b in zip(
            s1_names,
            cand_names,
        )
    ]

    feats["name_token_set_ratio"] = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(
            s1_names,
            cand_names,
        )
    ]

    feats["addr_ratio"] = [
        fuzz.ratio(a, b) / 100.0
        for a, b in zip(
            s1_addrs,
            cand_addrs,
        )
    ]

    feats["addr_token_sort_ratio"] = [
        fuzz.token_sort_ratio(a, b) / 100.0
        for a, b in zip(
            s1_addrs,
            cand_addrs,
        )
    ]

    # --------------------------------------------------------------
    # Exact / structural
    # --------------------------------------------------------------

    feats["name_exact"] = [
        int(
            a == b and a != ""
        )
        for a, b in zip(
            s1_names,
            cand_names,
        )
    ]

    feats["addr_exact"] = [
        int(
            a == b and a != ""
        )
        for a, b in zip(
            s1_addrs,
            cand_addrs,
        )
    ]

    feats["country_match"] = [
        int(a == b)
        for a, b in zip(
            s1_countries,
            cand_countries,
        )
    ]

    feats["name_len_diff"] = [
        abs(
            len(a) - len(b)
        )
        for a, b in zip(
            s1_names,
            cand_names,
        )
    ]

    feats["addr_len_diff"] = [
        abs(
            len(a) - len(b)
        )
        for a, b in zip(
            s1_addrs,
            cand_addrs,
        )
    ]

    # --------------------------------------------------------------
    # Token Jaccard
    # --------------------------------------------------------------

    def token_jaccard(a, b):
        ta = set(a.split())
        tb = set(b.split())

        if not ta and not tb:
            return 0.0

        return len(
            ta & tb
        ) / max(
            1,
            len(ta | tb),
        )

    feats["name_token_jaccard"] = [
        token_jaccard(a, b)
        for a, b in zip(
            s1_names,
            cand_names,
        )
    ]

    feats["addr_token_jaccard"] = [
        token_jaccard(a, b)
        for a, b in zip(
            s1_addrs,
            cand_addrs,
        )
    ]

    # --------------------------------------------------------------
    # Numeric overlap
    # --------------------------------------------------------------

    def numeric_overlap(a, b):
        sa = set(a.split(",")) - {""}
        sb = set(b.split(",")) - {""}

        if not sa or not sb:
            return 0.0

        return len(
            sa & sb
        ) / max(
            1,
            len(sa | sb),
        )

    feats["numeric_overlap"] = [
        numeric_overlap(a, b)
        for a, b in zip(
            s1_nums,
            cand_nums,
        )
    ]

    # --------------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------------

    if name_tfidf_vectorizer is not None:
        v1 = name_tfidf_vectorizer.transform(
            s1_names
        )

        v2 = name_tfidf_vectorizer.transform(
            cand_names
        )

        feats["name_tfidf_cosine"] = (
            np.asarray(
                v1.multiply(v2).sum(
                    axis=1
                )
            ).ravel()
        )

    if addr_tfidf_vectorizer is not None:
        v1 = addr_tfidf_vectorizer.transform(
            s1_addrs
        )

        v2 = addr_tfidf_vectorizer.transform(
            cand_addrs
        )

        feats["addr_tfidf_cosine"] = (
            np.asarray(
                v1.multiply(v2).sum(
                    axis=1
                )
            ).ravel()
        )

    return pd.concat(
        [
            candidates_df.reset_index(
                drop=True
            ),
            feats.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

FEATURE_COLUMNS = [
    "name_ratio", "name_partial_ratio", "name_token_sort_ratio", "name_token_set_ratio",
    "addr_ratio", "addr_token_sort_ratio", "name_exact", "addr_exact", "country_match",
    "name_len_diff", "addr_len_diff", "name_token_jaccard", "addr_token_jaccard",
    "numeric_overlap", "name_tfidf_cosine", "addr_tfidf_cosine",
]
