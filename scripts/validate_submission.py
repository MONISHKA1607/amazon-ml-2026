#!/usr/bin/env python3
"""
Local re-implementation of the official rule set, so you catch a rejection
before spending one of your 15 submissions on it. Always ALSO run the
official utils/validate_submission.py provided with the challenge before
uploading — this local copy is a fast pre-check, not a replacement.

Usage:
  python scripts/validate_submission.py \
      --matching outputs/submissions/submission_01/matching_results.tsv \
      --candidate outputs/submissions/submission_01/candidate_pairs.tsv \
      --test-dir dataset/test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils import read_tsv, parse_id_list, source_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--test-dir", required=True)
    args = ap.parse_args()

    issues = []

    s1 = read_tsv(os.path.join(args.test_dir, "test_source1.tsv"))
    s2 = read_tsv(os.path.join(args.test_dir, "test_source2.tsv"))
    s3 = read_tsv(os.path.join(args.test_dir, "test_source3.tsv"))
    valid_s1_ids = set(s1["entity_id"])
    valid_cand_ids = set(s2["entity_id"]) | set(s3["entity_id"])

    matching = read_tsv(args.matching)
    candidate = read_tsv(args.candidate)

    if list(matching.columns[:2]) != ["source1_entity_id", "matched_entity_ids"]:
        issues.append(f"matching_results.tsv columns are {list(matching.columns)}, "
                       f"expected ['source1_entity_id', 'matched_entity_ids']")
    if list(candidate.columns[:2]) != ["source1_entity_id", "candidate_entity_ids"]:
        issues.append(f"candidate_pairs.tsv columns are {list(candidate.columns)}, "
                       f"expected ['source1_entity_id', 'candidate_entity_ids']")

    # every S1 exists exactly once
    m_ids = list(matching["source1_entity_id"])
    dup_m = {x for x in m_ids if m_ids.count(x) > 1}
    if dup_m:
        issues.append(f"duplicate source1_entity_id rows in matching_results.tsv: {list(dup_m)[:5]}")
    missing_m = valid_s1_ids - set(m_ids)
    if missing_m:
        issues.append(f"{len(missing_m)} S1 entities missing from matching_results.tsv, "
                       f"e.g. {list(missing_m)[:5]}")
    extra_m = set(m_ids) - valid_s1_ids
    if extra_m:
        issues.append(f"matching_results.tsv has S1 ids not in the test set: {list(extra_m)[:5]}")

    cand_map = {}
    for _, row in candidate.iterrows():
        cand_map[row["source1_entity_id"]] = parse_id_list(row["candidate_entity_ids"])

    for _, row in matching.iterrows():
        s1_id = row["source1_entity_id"]
        matched = parse_id_list(row["matched_entity_ids"])

        raw = str(row["matched_entity_ids"])
        raw_list = [x.strip() for x in raw.split(",")] if raw.strip() else []
        if len(raw_list) != len(set(raw_list)):
            issues.append(f"{s1_id}: duplicate IDs within matched_entity_ids")

        bad_source = [m for m in matched if source_of(m) not in ("S2", "S3")]
        if bad_source:
            issues.append(f"{s1_id}: matched_entity_ids references non-S2/S3 id(s) {bad_source}")

        nonexistent = matched - valid_cand_ids
        if nonexistent:
            issues.append(f"{s1_id}: matched id(s) not present in test set: {nonexistent}")

        candidates_for_s1 = cand_map.get(s1_id, set())
        not_in_candidates = matched - candidates_for_s1
        if not_in_candidates:
            issues.append(f"{s1_id}: matched id(s) never appeared as a candidate "
                           f"(pipeline bug): {not_in_candidates}")

    if issues:
        print(f"FAIL — {len(issues)} issue(s):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        sys.exit(1)
    else:
        print("PASS — safe to run the official validator and submit.")
        sys.exit(0)


if __name__ == "__main__":
    main()
