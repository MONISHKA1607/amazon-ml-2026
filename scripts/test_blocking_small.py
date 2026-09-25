import sys

sys.path.insert(0, ".")

from src.data_loader import load_split
from src.normalize import add_normalized_columns
from src.blocking import generate_all_candidates
from src.blocking_config import BlockingConfig


def main():
    ds = load_split("dataset", "train")

    s1 = add_normalized_columns(
        ds.source1.head(1000).copy()
    )

    s2 = add_normalized_columns(
        ds.source2.head(5000).copy()
    )

    s3 = add_normalized_columns(
        ds.source3.head(5000).copy()
    )

    config = BlockingConfig(
        max_block_frequency=500,
        max_candidates_per_block=500,
        max_candidates_per_entity=500,
    )

    candidates, stats = generate_all_candidates(
        s1_df=s1,
        s2_df=s2,
        s3_df=s3,
        config=config,
    )

    print()
    print("Candidates:")
    print(candidates.head(20))

    print()
    print("Shape:")
    print(candidates.shape)

    print()
    print("Stats:")
    print(stats)


if __name__ == "__main__":
    main()