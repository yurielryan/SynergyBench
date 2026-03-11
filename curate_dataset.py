#!/usr/bin/env python3

"""Create reproducible train/val/test splits from docmsu_all.json.

Constraints:
- Total samples: 2500
- Class balance: 1250 with is_sar=0 and 1250 with is_sar=1
- Split ratio: 80/10/10 for train/val/test
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


TOTAL_SAMPLES = 2500
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DEFAULT_SEED = 20260311


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Curate balanced, reproducible train/val/test splits from docmsu_all.json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docmsu_all.json"),
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docmsu_2500_split.json"),
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "Input JSON must be a top-level object mapping IDs to samples."
        )

    return data


def stratified_balanced_sample(
    dataset: dict[str, dict[str, Any]],
    total_samples: int,
    rng: random.Random,
) -> tuple[list[str], list[str]]:
    if total_samples % 2 != 0:
        raise ValueError("total_samples must be even to enforce exact class balance.")

    target_per_class = total_samples // 2
    class_0_ids: list[str] = []
    class_1_ids: list[str] = []

    for sample_id, sample in dataset.items():
        label = sample.get("is_sar")
        if label == 0:
            class_0_ids.append(sample_id)
        elif label == 1:
            class_1_ids.append(sample_id)

    if len(class_0_ids) < target_per_class:
        raise ValueError(
            f"Not enough is_sar=0 samples: need {target_per_class}, found {len(class_0_ids)}"
        )
    if len(class_1_ids) < target_per_class:
        raise ValueError(
            f"Not enough is_sar=1 samples: need {target_per_class}, found {len(class_1_ids)}"
        )

    # Sort before shuffling so output is stable regardless of JSON key order.
    class_0_ids.sort()
    class_1_ids.sort()

    rng.shuffle(class_0_ids)
    rng.shuffle(class_1_ids)

    return class_0_ids[:target_per_class], class_1_ids[:target_per_class]


def split_class_ids(class_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    train_n = int(len(class_ids) * TRAIN_RATIO)
    val_n = int(len(class_ids) * VAL_RATIO)
    test_n = len(class_ids) - train_n - val_n

    if train_n <= 0 or val_n <= 0 or test_n <= 0:
        raise ValueError("Invalid split sizes. Check split ratios and sample count.")

    train_ids = class_ids[:train_n]
    val_ids = class_ids[train_n : train_n + val_n]
    test_ids = class_ids[train_n + val_n :]
    return train_ids, val_ids, test_ids


def build_split_map(
    sample_ids: list[str], dataset: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {sample_id: dataset[sample_id] for sample_id in sample_ids}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    dataset = load_dataset(args.input)
    class_0_ids, class_1_ids = stratified_balanced_sample(dataset, TOTAL_SAMPLES, rng)

    train_0, val_0, test_0 = split_class_ids(class_0_ids)
    train_1, val_1, test_1 = split_class_ids(class_1_ids)

    train_ids = train_0 + train_1
    val_ids = val_0 + val_1
    test_ids = test_0 + test_1

    # Shuffle each split to avoid all class-0 then class-1 ordering.
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    output = {
        "meta": {
            "input_file": str(args.input),
            "total_samples": TOTAL_SAMPLES,
            "seed": args.seed,
            "class_balance": {
                "is_sar_0": len(class_0_ids),
                "is_sar_1": len(class_1_ids),
            },
            "split_counts": {
                "train": len(train_ids),
                "validation": len(val_ids),
                "test": len(test_ids),
            },
            "split_class_balance": {
                "train": {
                    "is_sar_0": len(train_0),
                    "is_sar_1": len(train_1),
                },
                "validation": {
                    "is_sar_0": len(val_0),
                    "is_sar_1": len(val_1),
                },
                "test": {
                    "is_sar_0": len(test_0),
                    "is_sar_1": len(test_1),
                },
            },
        },
        "splits": {
            "train": build_split_map(train_ids, dataset),
            "validation": build_split_map(val_ids, dataset),
            "test": build_split_map(test_ids, dataset),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Wrote curated split to {args.output}")
    print(
        "Counts -> "
        f"train: {len(train_ids)}, validation: {len(val_ids)}, test: {len(test_ids)}"
    )


if __name__ == "__main__":
    main()
