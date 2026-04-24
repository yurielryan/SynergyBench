#!/usr/bin/env python3
"""prepare_docmsu.py — build docmsu_10000_split.json from archive/docmsu_all.json.

Pipeline
--------
1. Load every record from ``archive/docmsu_all.json`` (id -> record dict).
2. Sample a balanced subset of 10,000 records: 5,000 with ``is_sar == 1`` and
   5,000 with ``is_sar == 0`` (deterministic via ``--seed``).
3. Assign each sampled record a ``split`` field — 70% train, 15% validation,
   15% test — while preserving the 5,000 / 5,000 class balance inside each
   split (so every split is also class-balanced).
4. Write the result to ``docmsu_10000_split.json`` keeping every original field
   from ``docmsu_all.json`` and adding the new ``split`` key.
5. Verify class balance + split counts; abort before deleting anything if the
   checks fail.
6. Optionally prune ``./img/``: by default the script is a DRY RUN and only
   reports how many images *would* be removed. Pass ``--prune-images`` to
   actually delete unreferenced images.

Typical usage
-------------
    # 1. Build the JSON and preview how many images would be pruned.
    python prepare_docmsu.py

    # 2. After reviewing the dry-run output, actually prune ./img/.
    python prepare_docmsu.py --prune-images
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_INPUT = Path("archive/docmsu_all.json")
DEFAULT_OUTPUT = Path("docmsu_10000_split.json")
DEFAULT_IMG_DIR = Path("img")

DEFAULT_TOTAL = 10_000
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15
DEFAULT_SEED = 20260424

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source JSON with all DocMSU records.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write the curated split JSON.")
    parser.add_argument("--img-dir", type=Path, default=DEFAULT_IMG_DIR, help="Image directory to prune.")
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL, help="Total balanced samples (must be even).")
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--test-frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--prune-images",
        action="store_true",
        help="Delete images in --img-dir that are not referenced by the new split. "
             "Without this flag the script is a dry run that only reports counts.",
    )
    return parser.parse_args()


def load_records(path: Path) -> Dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a top-level JSON object in {path}, got {type(data).__name__}.")
    return data


def balanced_sample(
    records: Dict[str, dict],
    per_class: int,
    rng: random.Random,
) -> Tuple[List[str], List[str]]:
    """Return (pos_ids, neg_ids), each a deterministic random subsample of size ``per_class``."""
    pos_ids = [k for k, v in records.items() if int(v.get("is_sar", 0)) == 1]
    neg_ids = [k for k, v in records.items() if int(v.get("is_sar", 0)) == 0]
    if len(pos_ids) < per_class:
        raise ValueError(f"Only {len(pos_ids)} positive (is_sar=1) records; need {per_class}.")
    if len(neg_ids) < per_class:
        raise ValueError(f"Only {len(neg_ids)} negative (is_sar=0) records; need {per_class}.")
    # Sort for reproducibility — dict iteration order is insertion-dependent,
    # but sorting + seeded shuffle gives the same subset across Python runs.
    pos_ids.sort()
    neg_ids.sort()
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)
    return pos_ids[:per_class], neg_ids[:per_class]


def split_counts(per_class: int, train_frac: float, val_frac: float, test_frac: float) -> Tuple[int, int, int]:
    """Split ``per_class`` samples into (train, val, test) using largest-remainder rounding."""
    total_frac = train_frac + val_frac + test_frac
    if abs(total_frac - 1.0) > 1e-8:
        raise ValueError(f"Split fractions must sum to 1.0 (got {total_frac}).")
    raw = [per_class * train_frac, per_class * val_frac, per_class * test_frac]
    floors = [int(x) for x in raw]
    remainder = per_class - sum(floors)
    # Assign leftover to the splits with the largest fractional part.
    fracs = sorted(range(3), key=lambda i: -(raw[i] - floors[i]))
    for i in fracs[:remainder]:
        floors[i] += 1
    return floors[0], floors[1], floors[2]


def assign_splits(
    ids: List[str],
    n_train: int,
    n_val: int,
    n_test: int,
) -> Dict[str, str]:
    """Slice ``ids`` (already shuffled) into contiguous train/val/test chunks."""
    assert n_train + n_val + n_test == len(ids), (n_train, n_val, n_test, len(ids))
    out = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            out[sid] = "train"
        elif i < n_train + n_val:
            out[sid] = "validation"
        else:
            out[sid] = "test"
    return out


def build_split_json(
    records: Dict[str, dict],
    pos_ids: List[str],
    neg_ids: List[str],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    input_name: str,
) -> Dict[str, object]:
    per_class = len(pos_ids)
    assert len(neg_ids) == per_class, "pos_ids and neg_ids must be the same size"

    n_train_cls, n_val_cls, n_test_cls = split_counts(per_class, train_frac, val_frac, test_frac)
    pos_split = assign_splits(pos_ids, n_train_cls, n_val_cls, n_test_cls)
    neg_split = assign_splits(neg_ids, n_train_cls, n_val_cls, n_test_cls)

    samples: Dict[str, dict] = {}
    for sid in pos_ids + neg_ids:
        rec = dict(records[sid])  # shallow copy; preserve every original field
        rec["split"] = pos_split.get(sid) or neg_split.get(sid)
        samples[sid] = rec

    total = per_class * 2
    split_count = {"train": n_train_cls * 2, "validation": n_val_cls * 2, "test": n_test_cls * 2}
    split_class = {
        name: {"is_sar_0": n_cls, "is_sar_1": n_cls}
        for name, n_cls in (("train", n_train_cls), ("validation", n_val_cls), ("test", n_test_cls))
    }

    return {
        "meta": {
            "source_file": input_name,
            "seed": seed,
            "total_samples": total,
            "class_balance": {"is_sar_0": per_class, "is_sar_1": per_class},
            "split_counts": split_count,
            "split_class_balance": split_class,
            "split_fractions": {"train": train_frac, "validation": val_frac, "test": test_frac},
        },
        "samples": samples,
    }


def verify_balance(payload: Dict[str, object], expected_total: int) -> None:
    """Sanity-check the assembled payload; raises ``AssertionError`` on any mismatch."""
    samples = payload["samples"]
    assert isinstance(samples, dict)
    assert len(samples) == expected_total, f"expected {expected_total} samples, got {len(samples)}"

    from collections import Counter
    sar = Counter(int(r["is_sar"]) for r in samples.values())
    assert sar[0] == sar[1] == expected_total // 2, f"class imbalance: {dict(sar)}"

    by_split: Dict[str, Counter] = {}
    for rec in samples.values():
        split = rec.get("split")
        assert split in {"train", "validation", "test"}, f"bad split value: {split!r}"
        by_split.setdefault(split, Counter())[int(rec["is_sar"])] += 1

    for split_name, counter in by_split.items():
        assert counter[0] == counter[1], (
            f"split {split_name!r} is imbalanced: is_sar=0 {counter[0]}, is_sar=1 {counter[1]}"
        )

    print("Balance check PASSED:")
    print(f"  total: {len(samples)}  (is_sar=0: {sar[0]}, is_sar=1: {sar[1]})")
    for name in ("train", "validation", "test"):
        c = by_split[name]
        print(f"  {name:<10} n={c[0]+c[1]:<5} (is_sar=0: {c[0]}, is_sar=1: {c[1]})")


def referenced_image_names(payload: Dict[str, object]) -> set:
    names = set()
    for rec in payload["samples"].values():  # type: ignore[index]
        img_name = str(rec.get("img_name", "")).strip()
        if img_name:
            names.add(img_name)
    return names


def prune_images(img_dir: Path, keep: set, *, actually_delete: bool) -> None:
    if not img_dir.is_dir():
        print(f"[prune] image directory not found: {img_dir} — skipping.")
        return
    all_files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMG_EXTS]
    to_remove = [p for p in all_files if p.name not in keep]
    kept = len(all_files) - len(to_remove)
    missing_refs = sorted(keep - {p.name for p in all_files})

    print(f"[prune] image dir: {img_dir}")
    print(f"[prune] on-disk images: {len(all_files)}")
    print(f"[prune] referenced by split: {len(keep)}")
    print(f"[prune] keep: {kept}")
    print(f"[prune] remove: {len(to_remove)}")
    if missing_refs:
        print(f"[prune] WARNING: {len(missing_refs)} referenced images are NOT on disk "
              f"(first 5: {missing_refs[:5]})")

    if not actually_delete:
        print("[prune] dry run — no files deleted. Rerun with --prune-images to delete.")
        return

    for path in to_remove:
        path.unlink()
    print(f"[prune] deleted {len(to_remove)} files from {img_dir}")


def main() -> None:
    args = parse_args()

    if args.total % 2 != 0:
        raise ValueError("--total must be even for a balanced class split.")
    per_class = args.total // 2

    rng = random.Random(args.seed)
    records = load_records(args.input)
    print(f"[load] {len(records)} records from {args.input}")

    pos_ids, neg_ids = balanced_sample(records, per_class, rng)
    print(f"[sample] picked {len(pos_ids)} positive and {len(neg_ids)} negative ids (seed={args.seed})")

    payload = build_split_json(
        records,
        pos_ids,
        neg_ids,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        input_name=str(args.input.name),
    )
    verify_balance(payload, expected_total=args.total)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[write] {args.output}")

    keep = referenced_image_names(payload)
    prune_images(args.img_dir, keep, actually_delete=args.prune_images)


if __name__ == "__main__":
    main()
