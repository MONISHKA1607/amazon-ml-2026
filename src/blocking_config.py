from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class BlockingConfig:
    """
    Configuration for candidate generation.

    These values are engineering choices for the current baseline.
    They are not requirements from the challenge statement.
    """

    version: str = "v1"

    # Which blocking strategies are enabled.
    enabled_blocks: List[str] | None = None

    # Token filtering.
    min_token_length: int = 4

    # Ignore blocks whose posting list is too large.
    max_block_frequency: int = 5000

    # Safety limit for a single block.
    max_candidates_per_block: int = 5000

    # Maximum number of candidates retained for one S1 entity.
    max_candidates_per_entity: int = 5000

    # Character n-gram configuration.
    char_ngram_size: int = 3

    # A token occurring fewer times than this is considered "rare".
    rare_token_frequency: int = 1000

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BlockingConfig":
        return cls(**data)