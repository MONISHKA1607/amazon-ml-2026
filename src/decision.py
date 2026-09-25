"""
decision.py
Turns per-pair match scores into the final entity-level prediction that
matching_results.tsv actually needs: for every Source-1 entity, WHICH
(zero, one, or many) candidates to keep.

Two things live here that pure classification doesn't give you for free:
  1. threshold selection tuned to macro F0.5 (precision-heavy, not 0.5 default)
  2. singleton / margin logic — "should this entity have ANY match at all?"
     is a different question from "score the candidates".
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .scorer import macro_f05


def scores_to_predictions(scored_df: pd.DataFrame, threshold: float) -> dict:
    """scored_df needs columns: source1_entity_id, candidate_entity_id, score."""
    preds = defaultdict(set)
    for s1_id, cid, score in zip(
        scored_df["source1_entity_id"], scored_df["candidate_entity_id"], scored_df["score"]
    ):
        if score >= threshold:
            preds[s1_id].add(cid)
    return dict(preds)


def sweep_thresholds(scored_df: pd.DataFrame, truth: dict, all_s1_ids, thresholds=None) -> pd.DataFrame:
    """Evaluate macro F0.5 at each threshold; returns a DataFrame sorted best-first."""
    thresholds = thresholds if thresholds is not None else [i / 100 for i in range(10, 96, 5)]
    rows = []
    for t in thresholds:
        preds = scores_to_predictions(scored_df, t)
        result = macro_f05(preds, truth, all_s1_ids=all_s1_ids)
        rows.append({
            "threshold": t,
            "macro_f05": result["macro_f05"],
            "micro_precision": result["micro_precision"],
            "micro_recall": result["micro_recall"],
            "singleton_accuracy": result["singleton_accuracy"],
        })
    return pd.DataFrame(rows).sort_values("macro_f05", ascending=False).reset_index(drop=True)


def add_margin_features(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per S1 entity: top1_score, top2_score, margin = top1 - top2, n_candidates.
    A large margin (0.96 vs 0.45) is a confident single match; a small margin
    (0.96 vs 0.94) is ambiguous and worth a stricter rule.
    """
    df = scored_df.sort_values(["source1_entity_id", "score"], ascending=[True, False]).copy()
    df["rank"] = df.groupby("source1_entity_id").cumcount() + 1

    top2 = (
        df[df["rank"] <= 2]
        .pivot(index="source1_entity_id", columns="rank", values="score")
        .rename(columns={1: "top1_score", 2: "top2_score"})
    )
    top2["top2_score"] = top2.get("top2_score", pd.Series(dtype=float)).fillna(0.0)
    top2["margin"] = top2["top1_score"] - top2["top2_score"]
    n_cand = df.groupby("source1_entity_id").size().rename("n_candidates")

    df = df.merge(top2[["top1_score", "top2_score", "margin"]], on="source1_entity_id", how="left")
    df = df.merge(n_cand, on="source1_entity_id", how="left")
    return df


def singleton_aware_predictions(
    scored_df: pd.DataFrame,
    match_threshold: float,
    singleton_margin_threshold: float | None = None,
) -> dict:
    """
    Base rule: keep every candidate with score >= match_threshold (supports
    0/1/many matches per entity, as the task requires).

    Optional refinement: when an entity's top1 candidate barely clears the
    threshold AND top2 is very close behind (margin < singleton_margin_threshold),
    treat it as ambiguous and drop to no-match rather than risk a false merge
    — tune this on validation, it is not free precision by default.
    """
    df = add_margin_features(scored_df)
    preds = defaultdict(set)
    for s1_id, group in df.groupby("source1_entity_id"):
        keep = group[group["score"] >= match_threshold]
        if singleton_margin_threshold is not None and len(keep) >= 1:
            top1 = group.iloc[0]
            if top1["margin"] < singleton_margin_threshold and top1["score"] < match_threshold + 0.05:
                continue  # ambiguous top match near the threshold -> abstain
        preds[s1_id] = set(keep["candidate_entity_id"])
    return dict(preds)
