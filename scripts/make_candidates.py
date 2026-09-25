#!/usr/bin/env python3
"""
Build a versioned candidate_pairs file (Person 1's deliverable).

Usage:
  python scripts/make_candidates.py --data-dir dataset --split train \
      --version v1 --blocks exact_name,name_tokens,address_tokens

On train, also prints blocking recall against the ground truth so you know
the recall ceiling of this candidate set BEFORE anyone trains a model on it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_split, ground_truth_to_dict  # noqa: E402
from src.normalize import add_normalized_columns  # noqa: E402
from src.blocking import generate_all_candidates, compute_blocking_recall  # noqa: E402
from src.utils import write_tsv, timer, log_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="train")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--blocks", default=None, help="comma-separated block names, default = all")
    ap.add_argument("--out-dir", default="outputs/candidates")
    ap.add_argument("--owner", default="P1")
    args = ap.parse_args()

    block_names = args.blocks.split(",") if args.blocks else None

    ds = load_split(args.data_dir, args.split)
    s1 = add_normalized_columns(ds.source1)
    s2 = add_normalized_columns(ds.source2)
    s3 = add_normalized_columns(ds.source3)

    with timer("blocking"):
        candidates = generate_all_candidates(s1, s2, s3, block_names)

    out_path = os.path.join(args.out_dir, f"candidate_pairs_{args.version}.tsv")
    write_tsv(candidates, out_path)
    print(f"wrote {len(candidates)} candidate rows -> {out_path}")

    metrics = {"candidate_pair_count": len(candidates)}
    if ds.ground_truth is not None:
        gt_dict = ground_truth_to_dict(ds.ground_truth)
        metrics = compute_blocking_recall(candidates, gt_dict)
        print(json.dumps({k: v for k, v in metrics.items() if k != "missed_examples"}, indent=2))

        log_experiment(
            "experiments/logs/experiments.csv",
            experiment_id=f"block_{args.version}",
            owner=args.owner,
            candidate_version=args.version,
            blocking_recall=metrics.get("blocking_recall"),
            candidate_count=metrics.get("candidate_pair_count"),
            notes=f"blocks={block_names or 'all'}",
        )


if __name__ == "__main__":
    main()
