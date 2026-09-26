#!/usr/bin/env python3
"""
Build a versioned candidate_pairs file (Person 1's deliverable).

Usage:
  python scripts/make_candidates.py --data-dir dataset --split train \
      --version v2 --config configs/blocking_v2.json

On train, also prints blocking recall against the ground truth so you know
the recall ceiling of this candidate set BEFORE anyone trains a model on it.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.blocking_config import BlockingConfig
from src.data_loader import load_split, ground_truth_to_dict
from src.normalize import add_normalized_columns
from src.blocking import generate_all_candidates, compute_blocking_recall
from src.utils import write_tsv, timer, log_experiment


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--version", default="v1")
    ap.add_argument(
        "--blocks",
        default=None,
        help="comma-separated block names, default = all",
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/candidates",
    )
    ap.add_argument("--owner", default="P1")
    ap.add_argument(
        "--config",
        default="configs/blocking_v1.json",
    )

    args = ap.parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as f:
        config_data = json.load(f)

    config = BlockingConfig.from_dict(config_data)

    block_names = (
        args.blocks.split(",")
        if args.blocks
        else config.enabled_blocks
    )

    ds = load_split(
        args.data_dir,
        args.split,
    )

    s1 = add_normalized_columns(ds.source1)
    s2 = add_normalized_columns(ds.source2)
    s3 = add_normalized_columns(ds.source3)

    with timer("blocking"):
        candidates, blocking_stats = generate_all_candidates(
            s1_df=s1,
            s2_df=s2,
            s3_df=s3,
            block_names=block_names,
            config=config,
        )

    os.makedirs(args.out_dir, exist_ok=True)

    out_path = os.path.join(
        args.out_dir,
        f"candidate_pairs_{args.version}.tsv",
    )

    write_tsv(
        candidates,
        out_path,
    )

    print(
        f"wrote {len(candidates)} candidate rows -> {out_path}"
    )

    metrics = {
        "candidate_pair_count": len(candidates)
    }

    if ds.ground_truth is not None:
        gt_dict = ground_truth_to_dict(
            ds.ground_truth
        )

        metrics = compute_blocking_recall(
            candidates,
            gt_dict,
        )

        print(
            json.dumps(
                {
                    k: v
                    for k, v in metrics.items()
                    if k != "missed_examples"
                },
                indent=2,
            )
        )

        log_experiment(
            "experiments/logs/experiments.csv",
            experiment_id=f"block_{args.version}",
            owner=args.owner,
            candidate_version=args.version,
            blocking_recall=metrics.get(
                "blocking_recall"
            ),
            candidate_count=metrics.get(
                "candidate_pair_count"
            ),
            notes=(
                f"blocks={block_names}"
                f"; config={args.config}"
            ),
        )

    print()
    print("=== Candidate Generation Summary ===")
    print(
        f"Candidate version: {config.version}"
    )
    print(
        f"Candidate file: {out_path}"
    )

    print()
    print("S2:")

    for key, value in blocking_stats["s2"].items():
        print(
            f"  {key}: {value}"
        )

    print()
    print("S3:")

    for key, value in blocking_stats["s3"].items():
        print(
            f"  {key}: {value}"
        )

    print()
    print(
        f"Total candidate pairs: "
        f"{blocking_stats['n_candidate_pairs']}"
    )


if __name__ == "__main__":
    main()