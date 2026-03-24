"""Evaluation pipeline for sarcasm classification with evaluator model backends.

Expected evaluator model interface:
    model.evaluate(text: str | None, image: str | Path | None) -> str
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluator_models.openai import load_openai_evaluator
from evaluator_models.utils import parse_yes_no_prediction


def load_dataset(dataset_path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Dataset JSON must be a top-level mapping.")

    return data


def select_dataset_samples(
    dataset_json: dict[str, Any],
    split: str = "all",
) -> dict[str, dict[str, Any]]:
    """Normalize raw or curated dataset JSON into sample_id -> sample mapping.

    Supports two JSON layouts:
    1) raw: {sample_id: sample}
    2) curated: {"meta": ..., "splits": {"train": {...}, "validation": {...}, "test": {...}}} 
    
    NOTE: This "meta" field simply contains the metadata. See docmsu_2500_split.json.
    """
    if "splits" not in dataset_json: # assume raw
        return dataset_json

    splits = dataset_json.get("splits", {}) # available splits: train, validation, test
    split_name = split.lower()
    
    if not isinstance(splits, dict):
        raise ValueError("If present, 'splits' must be a mapping.")

    if split_name == "all": # default case for evaluating synergy creation.
        merged: dict[str, dict[str, Any]] = {}
        for key in ("train", "validation", "test"): # we assume the dataset will always have these three splits, even if some are empty.
            split_map = splits.get(key, {})
            if isinstance(split_map, dict):
                merged.update(split_map)
        if merged:
            return merged

    elif split_name in {"train", "validation", "test"}: # in case we need to separately train/validate then test later on.
        selected = splits.get(split_name, {})
        if not isinstance(selected, dict):
            raise ValueError(f"Split '{split_name}' must be a mapping.")
        return selected
    
    else:
        raise ValueError("split must be one of: all, train, validation, test")


def init_evaluator_model(
    provider: str = "openai",
) -> Any:
    """Initialize an evaluator model from evaluator_models by provider."""

    provider_name = provider.lower()
    if provider_name == "openai":
        return load_openai_evaluator()

    raise ValueError("Unsupported evaluator provider. Currently supported: openai")


def evaluate_text(model: Any, dataset: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Run text-only evaluation by calling model.evaluate(text=..., image=None)."""
    results: dict[str, str] = {}
    for sample_id, sample in dataset.items():
        text = sample.get("text")
        if not isinstance(text, str) or not text.strip():
            results[sample_id] = "unknown"
            continue

        raw_output = model.evaluate(text=text, image=None)
        results[sample_id] = parse_yes_no_prediction(raw_output)
    return results


def evaluate_image(
    model: Any,
    dataset: dict[str, dict[str, Any]],
    image_dir: str | Path,
) -> dict[str, str]:
    """Run image-only evaluation by calling model.evaluate(text=None, image=...)."""
    results: dict[str, str] = {}
    image_root = Path(image_dir)

    for sample_id, sample in dataset.items():
        img_name = sample.get("img_name")
        if not isinstance(img_name, str) or not img_name:
            results[sample_id] = "unknown"
            continue

        image_path = image_root / img_name
        if not image_path.exists():
            results[sample_id] = "unknown"
            continue

        raw_output = model.evaluate(text=None, image=image_path)
        results[sample_id] = parse_yes_no_prediction(raw_output)
    return results


def evaluate_multimodal(
    model: Any,
    dataset: dict[str, dict[str, Any]],
    image_dir: str | Path,
) -> dict[str, str]:
    """Run multimodal evaluation by calling model.evaluate(text=..., image=...)."""
    results: dict[str, str] = {}
    image_root = Path(image_dir)

    for sample_id, sample in dataset.items():
        text = sample.get("text")
        img_name = sample.get("img_name")

        if not isinstance(text, str) or not text.strip():
            results[sample_id] = "unknown"
            continue
        if not isinstance(img_name, str) or not img_name:
            results[sample_id] = "unknown"
            continue

        image_path = image_root / img_name
        if not image_path.exists():
            results[sample_id] = "unknown"
            continue

        raw_output = model.evaluate(text=text, image=image_path)
        results[sample_id] = parse_yes_no_prediction(raw_output)
    return results


def aggregate_results(
    dataset: dict[str, dict[str, Any]],
    text_results: dict[str, str],
    image_results: dict[str, str],
    multimodal_results: dict[str, str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for sample_id, sample in dataset.items():
        output[sample_id] = {
            "sample_id": sample_id,
            "ground_truth": "yes" if sample.get("is_sar") == 1 else "no",
            "text_only": text_results.get(sample_id, "unknown"),
            "image_only": image_results.get(sample_id, "unknown"),
            "multimodal": multimodal_results.get(sample_id, "unknown"),
        }
    return output


def write_results(results: dict[str, Any], output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluator-based sarcasm inference.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("docmsu_2500_split.json"),
        help="Path to dataset JSON (raw mapping or curated split JSON).",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "validation", "test"],
        default="all",
        help="Which split to evaluate when dataset has curated splits.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("img"),
        help="Directory containing image files referenced by img_name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/evaluator_inference.json"),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai"],
        default="openai",
        help="Evaluator model provider.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional sample limit for quick smoke tests (0 means no limit).",
    )
    return parser.parse_args()


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    dataset_json = load_dataset(args.dataset)
    dataset = select_dataset_samples(dataset_json, split=args.split)

    if args.limit and args.limit > 0: # set limits for tests
        limited_items = list(dataset.items())[: args.limit]
        dataset = dict(limited_items)

    model = init_evaluator_model(provider=args.provider)

    text_results = evaluate_text(model, dataset) # note that dataset here is the json: sample_id + annotations. raw images are in /img/{sample_id}.jpg
    image_results = evaluate_image(model, dataset, image_dir=args.image_dir)
    multimodal_results = evaluate_multimodal(model, dataset, image_dir=args.image_dir)

    final_results = {
        "run_config": {
            "dataset": str(args.dataset),
            "split": args.split,
            "image_dir": str(args.image_dir),
            "output": str(args.output),
            "provider": args.provider,
            "limit": args.limit,
        },
        "results": aggregate_results(
            dataset,
            text_results,
            image_results,
            multimodal_results,
        ),
    }
    return final_results


def main() -> None:
    args = parse_args()
    final_results = run_evaluation(args)
    write_results(final_results, args.output)
    print(f"Saved {len(final_results['results'])} results to {args.output}")


if __name__ == "__main__":
    main()
