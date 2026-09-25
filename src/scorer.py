"""
scorer.py
THE most important file in the repo — every threshold/blocking/feature
decision is judged against this. Implements the official metric exactly:

F_0.5 = (1.25 * P * R) / (0.25 * P + R), computed PER Source-1 entity,
then macro-averaged across all Source-1 entities.

Singleton rule (explicit in the problem statement):
  - true matches empty, predicted empty  -> F0.5 = 1.0
  - true matches empty, predicted non-empty -> F0.5 = 0.0
  - true matches non-empty, predicted empty -> F0.5 = 0.0 (recall = 0)

Verified against the worked example in the problem statement:
  pred = {S2-00047, S2-00193, S3-00812}, true = {S2-00047, S3-00812}
  -> P=2/3, R=1.0, F0.5 = 0.714
"""
from __future__ import annotations

from typing import Dict, Set

BETA2 = 0.25  # beta=0.5 -> beta^2 = 0.25


def entity_f05(pred: Set[str], true: Set[str]) -> float:
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    denom = BETA2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + BETA2) * precision * recall / denom


def macro_f05(
    predictions: Dict[str, Set[str]],
    truth: Dict[str, Set[str]],
    all_s1_ids=None,
) -> dict:
    """
    predictions / truth: source1_entity_id -> set(matched_entity_ids)
    all_s1_ids: the full evaluation universe (every S1 entity in the split).
    Missing keys in `predictions` are treated as an empty prediction ([]),
    exactly as an empty matched_entity_ids row means in the real file.
    """
    if all_s1_ids is None:
        all_s1_ids = set(truth.keys()) | set(predictions.keys())

    scores = []
    singleton_correct = 0
    singleton_total = 0
    exact_match_count = 0

    for s1_id in all_s1_ids:
        true_set = truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        f = entity_f05(pred_set, true_set)
        scores.append(f)

        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
        if pred_set == true_set:
            exact_match_count += 1

    n = len(scores)
    macro = sum(scores) / n if n else 0.0

    # micro precision/recall across all entities, purely diagnostic (NOT the
    # scored metric — the leaderboard uses the macro average above)
    tp = fp = fn = 0
    for s1_id in all_s1_ids:
        true_set = truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        tp += len(pred_set & true_set)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)
    micro_precision = tp / (tp + fp) if (tp + fp) else 1.0
    micro_recall = tp / (tp + fn) if (tp + fn) else 1.0

    return {
        "macro_f05": macro,
        "n_entities": n,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total else None,
        "singleton_total": singleton_total,
        "exact_match_set_accuracy": exact_match_count / n if n else 0.0,
        "per_entity_scores": dict(zip(all_s1_ids, scores)),
    }


if __name__ == "__main__":
    # Sanity check against the worked example in the problem statement.
    pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}
    true = {"S1-00001": {"S2-00047", "S3-00812"}}
    result = macro_f05(pred, true)
    assert abs(result["macro_f05"] - 0.714) < 0.001, result
    print("scorer self-check passed:", result["macro_f05"])
