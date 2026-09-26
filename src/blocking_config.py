from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List


@dataclass
class BlockingConfig:
    version: str = "v2"

    enabled_blocks: List[str] | None = None

    min_token_length: int = 4

    max_block_frequency: int = 5000

    max_candidates_per_block: int = 5000

    max_candidates_per_entity: int = 5000

    char_ngram_size: int = 3

    rare_token_frequency: int = 1000

    address_anchor_min_token_length: int = 4

    short_name_token_min_length: int = 3

    short_name_token_max_frequency: int = 10000

    def to_dict(self) -> dict:
        return asdict(self)

        @classmethod
        def from_dict(
            cls,
            data: dict,
        ) -> "BlockingConfig":
            return cls(**data)
