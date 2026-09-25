"""
utils.py
Shared I/O, ID-list helpers, and experiment logging.

Nothing in this file is ML logic — it is the plumbing every other module
relies on, so it must be boring and bug-free.
"""
from __future__ import annotations

import csv
import os
import time
from contextlib import contextmanager
from typing import Iterable

import pandas as pd


# --------------------------------------------------------------------------
# TSV I/O — always explicit sep="\t". Reading without it silently produces
# a single column containing the whole line (the challenge doc warns about
# exactly this), so every read/write in the project must go through here.
# --------------------------------------------------------------------------

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df


def write_tsv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def validate_source_schema(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing expected columns {missing}, got {list(df.columns)}")


# --------------------------------------------------------------------------
# ID-list parsing/formatting
# matched_entity_ids / candidate_entity_ids columns are comma-separated,
# no quoting, no whitespace, no duplicates.
# --------------------------------------------------------------------------

def parse_id_list(cell: str) -> set:
    if cell is None:
        return set()
    cell = str(cell).strip()
    if cell == "" or cell.lower() == "nan":
        return set()
    return set(x.strip() for x in cell.split(",") if x.strip())


def format_id_list(ids: Iterable[str]) -> str:
    # Sorted for reproducibility; sorting is stable and makes diffs readable.
    return ",".join(sorted(set(ids)))


def source_of(entity_id: str) -> str:
    """S1-xxx -> 'S1', S2-xxx -> 'S2', S3-xxx -> 'S3'."""
    return entity_id.split("-")[0]


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

@contextmanager
def timer(label: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[{label}] {dt:.2f}s")


# --------------------------------------------------------------------------
# Experiment logging — append-only, never overwrite a row.
# --------------------------------------------------------------------------

EXPERIMENT_LOG_FIELDS = [
    "experiment_id", "timestamp", "owner", "candidate_version", "feature_version",
    "model_version", "threshold", "macro_f05", "precision", "recall",
    "singleton_accuracy", "blocking_recall", "candidate_count", "runtime_sec", "notes",
]


def log_experiment(log_path: str, **kwargs) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    file_exists = os.path.isfile(log_path)
    row = {k: kwargs.get(k, "") for k in EXPERIMENT_LOG_FIELDS}
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
