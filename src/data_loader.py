"""
data_loader.py
Load the challenge TSVs and build a leakage-free train/validation split.

The split MUST be by Source-1 entity, not by pair — otherwise the same S1
business leaks between train and validation and your local macro-F0.5
becomes an overestimate of the real (private-leaderboard) score.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import pandas as pd

from .utils import read_tsv, validate_source_schema, parse_id_list


@dataclass
class Dataset:
    source1: pd.DataFrame
    source2: pd.DataFrame
    source3: pd.DataFrame
    ground_truth: pd.DataFrame | None  # None for test set


def load_split(data_dir: str, split: str) -> Dataset:
    """split is 'train' or 'test'."""
    s1 = read_tsv(os.path.join(data_dir, split, f"{split}_source1.tsv"))
    s2 = read_tsv(os.path.join(data_dir, split, f"{split}_source2.tsv"))
    s3 = read_tsv(os.path.join(data_dir, split, f"{split}_source3.tsv"))
    for df, name in [(s1, "source1"), (s2, "source2"), (s3, "source3")]:
        validate_source_schema(df, f"{split}_{name}")

    gt = None
    gt_path = os.path.join(data_dir, split, f"{split}_ground_truth.tsv")
    if os.path.isfile(gt_path):
        gt = read_tsv(gt_path)
        if "source1_entity_id" not in gt.columns or "matched_entity_ids" not in gt.columns:
            raise ValueError(f"{gt_path}: unexpected ground-truth columns {list(gt.columns)}")

    return Dataset(source1=s1, source2=s2, source3=s3, ground_truth=gt)


def ground_truth_to_dict(gt: pd.DataFrame) -> dict:
    """source1_entity_id -> set(matched_entity_ids). Missing S1 rows -> not in dict (treat as empty)."""
    out = {}
    for _, row in gt.iterrows():
        out[row["source1_entity_id"]] = parse_id_list(row["matched_entity_ids"])
    return out


def entity_level_split(s1_df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42):
    """
    Split Source-1 entity_ids into train/val groups. Every S2/S3 record tied
    to a given S1 entity should be evaluated only within that entity's split
    (this is naturally handled downstream since scoring/candidates are keyed
    by source1_entity_id) — we just need the S1 id partition itself.
    """
    ids = list(s1_df["entity_id"])
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def basic_eda(ds: Dataset) -> dict:
    stats = {
        "n_source1": len(ds.source1),
        "n_source2": len(ds.source2),
        "n_source3": len(ds.source3),
        "country_counts_s1": ds.source1["country"].value_counts().to_dict(),
        "country_counts_s2": ds.source2["country"].value_counts().to_dict(),
        "country_counts_s3": ds.source3["country"].value_counts().to_dict(),
        "missing_name_s1": int((ds.source1["business_name"].str.strip() == "").sum()),
        "missing_address_s1": int((ds.source1["business_address"].str.strip() == "").sum()),
    }
    if ds.ground_truth is not None:
        gt_dict = ground_truth_to_dict(ds.ground_truth)
        match_counts = [len(v) for v in gt_dict.values()]
        n_singleton = sum(1 for c in match_counts if c == 0)
        n_multi = sum(1 for c in match_counts if c > 1)
        n_single_match = sum(1 for c in match_counts if c == 1)
        stats.update({
            "n_gt_rows": len(gt_dict),
            "n_singleton": n_singleton,
            "n_single_match": n_single_match,
            "n_multi_match": n_multi,
            "singleton_pct": round(100 * n_singleton / max(1, len(gt_dict)), 2),
            "avg_matches_per_s1": round(sum(match_counts) / max(1, len(match_counts)), 3),
            "max_matches_per_s1": max(match_counts) if match_counts else 0,
        })
    return stats
