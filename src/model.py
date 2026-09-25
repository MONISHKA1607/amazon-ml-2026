"""
model.py
Pairwise match/no-match classifier on top of the candidate-pair features.

Label construction: a candidate pair (s1_id, cand_id) is a positive if
cand_id is in the ground-truth matched set for s1_id, negative otherwise.
This means every non-matching candidate that blocking produced becomes a
(mostly easy) negative for free — hard negatives are mined separately
(see mine_hard_negative_ids) because random blocked negatives are too easy
and don't teach the model much.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

from .features import FEATURE_COLUMNS


def label_candidates(features_df: pd.DataFrame, gt_dict: dict) -> pd.Series:
    def is_positive(row):
        return int(row["candidate_entity_id"] in gt_dict.get(row["source1_entity_id"], set()))
    return features_df.apply(is_positive, axis=1)


def mine_hard_negative_ids(features_df: pd.DataFrame, labels: pd.Series, top_k_per_entity=5):
    """
    Returns the row-indices of the hardest negatives: for each S1 entity,
    the highest-name-similarity NEGATIVE candidates. These are exactly the
    "same name, different address" / "similar name, same city" cases that
    a random-negative model won't have learned to reject.
    """
    df = features_df.copy()
    df["_label"] = labels.values
    hard_idx = []
    for s1_id, group in df[df["_label"] == 0].groupby("source1_entity_id"):
        top = group.sort_values("name_ratio", ascending=False).head(top_k_per_entity)
        hard_idx.extend(top.index.tolist())
    return hard_idx


def train_model(
    features_df: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,  # source1_entity_id, so CV folds never split one entity's candidates
    feature_columns=None,
    params=None,
    num_boost_round=300,
):
    if not HAS_LGBM:
        raise ImportError("lightgbm not installed — pip install lightgbm")
    feature_columns = feature_columns or FEATURE_COLUMNS
    X = features_df[feature_columns].fillna(0.0).values
    y = labels.values

    default_params = dict(
        objective="binary",
        metric="auc",
        num_leaves=31,
        learning_rate=0.05,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        is_unbalance=True,  # positives are rare relative to blocked negatives
    )
    if params:
        default_params.update(params)

    train_set = lgb.Dataset(X, label=y, group=None)
    model = lgb.train(default_params, train_set, num_boost_round=num_boost_round)
    return model


def predict_scores(model, features_df: pd.DataFrame, feature_columns=None) -> np.ndarray:
    feature_columns = feature_columns or FEATURE_COLUMNS
    X = features_df[feature_columns].fillna(0.0).values
    return model.predict(X)
