"""
blocking.py
Multi-pass candidate generation using inverted indices (never a raw
S1 x S2 x S3 nested loop — that is the #1 way teams die on runtime).

Each block type builds a token -> [entity_ids] index on the S2/S3 side
once, then looks up each S1 entity's tokens against it. This is O(n) index
construction + O(matches) lookup, not O(n*m).

blocking recall = (# true pairs that appear in the candidate set) / (# all true pairs)
This is the single most important diagnostic for this stage — a pair that
never becomes a candidate can never be recovered later, however good the
matching model is.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .normalize import char_ngrams, extract_numeric_tokens


def _build_inverted_index(df: pd.DataFrame, key_fn) -> dict:
    """key_fn(row) -> iterable of keys. Returns key -> list[entity_id]."""
    index = defaultdict(list)
    for eid, keys in zip(df["entity_id"], df.apply(key_fn, axis=1)):
        for k in keys:
            if k:
                index[k].append(eid)
    return index


def _lookup_candidates(s1_df: pd.DataFrame, key_fn, index: dict) -> dict:
    """Returns s1_entity_id -> set(candidate_ids) for a single block type."""
    out = defaultdict(set)
    for eid, keys in zip(s1_df["entity_id"], s1_df.apply(key_fn, axis=1)):
        for k in keys:
            if k in index:
                out[eid].update(index[k])
    return out


def block_exact_name(s1_df, other_df):
    idx = _build_inverted_index(other_df, lambda r: [r["norm_name"]] if r["norm_name"] else [])
    return _lookup_candidates(s1_df, lambda r: [r["norm_name"]] if r["norm_name"] else [], idx)


def block_name_tokens(s1_df, other_df, min_token_len=4):
    """Any shared, sufficiently-long name token (skips short/common tokens)."""
    def keys(row):
        return {t for t in row["norm_name"].split(" ") if len(t) >= min_token_len}
    idx = _build_inverted_index(other_df, keys)
    return _lookup_candidates(s1_df, keys, idx)


def block_address_tokens(s1_df, other_df, min_token_len=4):
    def keys(row):
        return {t for t in row["norm_address"].split(" ") if len(t) >= min_token_len}
    idx = _build_inverted_index(other_df, keys)
    return _lookup_candidates(s1_df, keys, idx)


def block_char_ngrams(s1_df, other_df, n=3):
    """Catches typos/transpositions that token blocks miss (e.g. 'Amazon Fresh' vs 'Amazon Frehs')."""
    def keys(row):
        return char_ngrams(row["norm_name"], n=n)
    idx = _build_inverted_index(other_df, keys)
    return _lookup_candidates(s1_df, keys, idx)


def block_numeric_tokens(s1_df, other_df):
    """Shared house number / PIN / postal code tokens in the address."""
    def keys(row):
        return extract_numeric_tokens(row["norm_address"])
    idx = _build_inverted_index(other_df, keys)
    return _lookup_candidates(s1_df, keys, idx)


def block_country_name_prefix(s1_df, other_df, prefix_len=4):
    """Country-scoped name-prefix block. Country is an open string set — never
    hard-coded to {US, India}; France (or any other value) flows through untouched."""
    def keys(row):
        prefix = row["norm_name"][:prefix_len]
        return [f"{row['country']}||{prefix}"] if prefix else []
    idx = _build_inverted_index(other_df, keys)
    return _lookup_candidates(s1_df, keys, idx)


BLOCK_FUNCS = {
    "exact_name": block_exact_name,
    "name_tokens": block_name_tokens,
    "address_tokens": block_address_tokens,
    "char_ngrams": block_char_ngrams,
    "numeric_tokens": block_numeric_tokens,
    "country_name_prefix": block_country_name_prefix,
}


def generate_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame, block_names=None) -> pd.DataFrame:
    """
    Runs the requested blocks (default: all) against one other source
    (Source 2 or Source 3) and returns the union as a long-format DataFrame:
    source1_entity_id, candidate_entity_id, blocks_matched (comma list).
    """
    block_names = block_names or list(BLOCK_FUNCS.keys())
    per_block = {}
    for name in block_names:
        per_block[name] = BLOCK_FUNCS[name](s1_df, other_df)

    union = defaultdict(lambda: defaultdict(set))  # s1_id -> cand_id -> {block names}
    for name, mapping in per_block.items():
        for s1_id, cand_ids in mapping.items():
            for cid in cand_ids:
                union[s1_id][cid].add(name)

    rows = []
    for s1_id, cands in union.items():
        for cid, blocks in cands.items():
            rows.append((s1_id, cid, ",".join(sorted(blocks))))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "blocks_matched"])


def generate_all_candidates(s1_df, s2_df, s3_df, block_names=None) -> pd.DataFrame:
    cand2 = generate_candidates(s1_df, s2_df, block_names)
    cand3 = generate_candidates(s1_df, s3_df, block_names)
    return pd.concat([cand2, cand3], ignore_index=True)


def compute_blocking_recall(candidates_df: pd.DataFrame, gt_dict: dict) -> dict:
    """
    gt_dict: source1_entity_id -> set(true matched ids)
    Returns overall recall plus per-S1 counts so you can see WHICH entities'
    true matches are being missed by blocking (feed straight to error analysis).
    """
    cand_map = defaultdict(set)
    for s1, cid in zip(candidates_df["source1_entity_id"], candidates_df["candidate_entity_id"]):
        cand_map[s1].add(cid)

    total_true = 0
    total_found = 0
    missed_entities = []
    for s1_id, true_ids in gt_dict.items():
        if not true_ids:
            continue
        found = true_ids & cand_map.get(s1_id, set())
        total_true += len(true_ids)
        total_found += len(found)
        if found != true_ids:
            missed_entities.append((s1_id, true_ids - found))

    recall = total_found / total_true if total_true else 1.0
    return {
        "blocking_recall": recall,
        "total_true_pairs": total_true,
        "total_found_pairs": total_found,
        "n_entities_with_missed_matches": len(missed_entities),
        "missed_examples": missed_entities[:20],
        "candidate_pair_count": len(candidates_df),
        "avg_candidates_per_s1": candidates_df.groupby("source1_entity_id").size().mean() if len(candidates_df) else 0,
    }
