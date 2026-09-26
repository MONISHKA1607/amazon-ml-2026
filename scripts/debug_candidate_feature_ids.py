from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.blocking import (
    build_blocking_context,
    generate_candidates_batch,
)
from src.blocking_config import BlockingConfig
from src.data_loader import load_split
from src.normalize import add_normalized_columns


def main() -> None:
    dataset = load_split("dataset", "train")

    s1 = dataset.source1.head(5000).copy()
    s2 = dataset.source2.head(50000).copy()

    print("Normalizing...")
    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)

    train_s1 = s1.head(250).copy()

    config = BlockingConfig()

    print("Building context...")
    context = build_blocking_context(
        s2,
        config=config,
    )

    print("Generating candidates...")
    candidates, stats = generate_candidates_batch(
        train_s1,
        context,
        config,
        "s2",
    )

    print(f"Candidate rows: {len(candidates):,}")

    candidate_ids = set(
        candidates["candidate_entity_id"].astype(str)
    )

    other_ids = set(
        s2["entity_id"].astype(str)
    )

    missing = candidate_ids - other_ids

    print(f"Unique candidate IDs: {len(candidate_ids):,}")
    print(f"Unique S2 IDs:        {len(other_ids):,}")
    print(f"Missing IDs:          {len(missing):,}")

    if missing:
        print("\nFirst missing IDs:")
        for entity_id in sorted(missing)[:20]:
            print(entity_id)

    print("\nCandidate source IDs:")
    print(
        candidates["candidate_entity_id"]
        .head(20)
        .tolist()
    )

    print("\nS2 source IDs:")
    print(
        s2["entity_id"]
        .head(20)
        .tolist()
    )


if __name__ == "__main__":
    main()