#!/usr/bin/env python3
"""Usage: python scripts/eda.py --data-dir dataset --split train"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_loader import load_split, basic_eda  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    ds = load_split(args.data_dir, args.split)
    stats = basic_eda(ds)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
