"""
Generation pipeline for image-based synergy generation.
"""

from __future__ import annotations

import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from image_generator_models.openai import load_openai_image_generator
from image_generator_models.gemini import load_gemini_image_generator
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


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


def init_generator_model(
    args: dict[str, Any] | None = None,
) -> Any:
    """Initialize a generator model from generator_models by provider."""

    if args is None:
        raise ValueError("Missing arguments")

    provider_name = args.provider.lower()
    if provider_name == "openai":
        return load_openai_image_generator()
    elif provider_name == "google":
        return load_gemini_image_generator()

    raise ValueError("Unsupported generator provider. Currently supported: openai")


ProgressCallback = Callable[[str, int, int], None]

def generate_synergy(
    model: Any,
    dataset: dict[str, dict[str, Any]],
    image_dir: str | Path,
    save_dir: str | Path,
    results: dict[str, str] | None = None,
    on_progress: ProgressCallback | None = None,
    max_workers: int = 4,
) -> dict[str, str]:
    """Run synergy generation with parallel inference via ThreadPoolExecutor."""
    if results is None:
        results = {}

    with open('results/base_dataset.json', 'r', encoding='utf-8') as f:
        evaluated_dataset = json.load(f)

    image_root = Path(image_dir)
    total = len(dataset)

    # Separate already-done samples from those that need inference.
    pending: list[tuple[str, str, Path]] = []  # (sample_id, text, image_path)
    for sample_id, sample in dataset.items():
        if evaluated_dataset['results'].get(sample_id, {}).get("interaction", "unknown") != "U1":
            continue

        output_path = Path(save_dir) / f"{sample_id}.jpg"
        if os.path.exists(output_path):
            results[sample_id] = str(output_path)
            continue

        text = sample.get("text")
        img_name = sample.get("img_name")

        if not isinstance(text, str) or not text.strip():
            results[sample_id] = "unknown"
        elif not isinstance(img_name, str) or not img_name:
            results[sample_id] = "unknown"
        else:
            image_path = image_root / img_name
            if not image_path.exists():
                results[sample_id] = "unknown"
            else:
                pending.append((sample_id, text, image_path))

    def _infer(sample_id: str, text: str, image_path: Path) -> tuple[str, bytes | None]:
        return sample_id, model.inference(text=text, image=image_path)

    completed_count = len(results)  # already-done samples count toward progress

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(_infer, sample_id, text, image_path): sample_id
            for sample_id, text, image_path in pending
        }

        with tqdm(total=total, desc="Generating synergy", initial=completed_count) as pbar:
            for future in as_completed(future_to_id):
                sample_id = future_to_id[future]
                output_path = Path(save_dir) / f"{sample_id}.jpg"
                completed_count += 1

                try:
                    _, raw_output = future.result()
                except Exception as exc:
                    print(f"[{sample_id}] inference raised: {exc}")
                    raw_output = None

                if raw_output is None:
                    results[sample_id] = "unknown"
                else:
                    with open(output_path, "wb") as img_file:
                        img_file.write(raw_output)
                    results[sample_id] = str(output_path)

                pbar.update(1)

                if on_progress is not None:
                    on_progress("synergy generation", completed_count, total)

    return results


def aggregate_response(
    dataset: dict[str, dict[str, Any]],
    save_dir: str | Path,
    synergy_response: dict[str, str],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for sample_id, sample in dataset.items():
        if sample_id not in synergy_response:
            continue
        output[sample_id] = {
            "sample_id": sample_id,
            "text": sample.get("text", ""),
            "image_path": synergy_response.get(sample_id, "unknown")
        }
    return output


def build_response_payload(
    dataset: dict[str, dict[str, Any]],
    run_config: dict[str, Any],
    synergy_response: dict[str, str],
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_config": run_config,
        "response": aggregate_response(
            dataset,
            run_config.get("generated_image_dir"),
            synergy_response,
        ),
    }
    if checkpoint is not None:
        payload["checkpoint"] = checkpoint
    return payload


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
        help="Directory containing DOCMSU image files referenced by img_name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("responses/img_synergy_generation.json"),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "google"],
        default="openai",
        help="Generation model provider.",
    )
    parser.add_argument(
        "--generated-image-dir",
        type=Path,
        default=Path('generated_images'),
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional sample limit for quick smoke tests (0 means no limit).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save partial results every N sample generations (0 disables checkpointing).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel inference threads (default: 4).",
    )
    return parser.parse_args()


def run_generation(args: argparse.Namespace) -> dict[str, Any]:
    dataset_json = load_dataset(args.dataset)
    dataset = select_dataset_samples(dataset_json, split=args.split)

    if args.limit and args.limit > 0: # set limits for tests
        limited_items = list(dataset.items())[: args.limit]
        dataset = dict(limited_items)

    model = init_generator_model(args=args)

    run_config: dict[str, Any] = {
        "dataset": str(args.dataset),
        "split": args.split,
        "image_dir": str(args.image_dir),
        "output": str(args.output),
        "provider": args.provider,
        "generated_image_dir": str(args.generated_image_dir),
        "limit": args.limit,
        "checkpoint_every": args.checkpoint_every,
        "max_workers": args.max_workers,
    }

    synergy_response: dict[str, str] = {}

    total_steps = len(dataset)
    completed_steps = 0

    def checkpoint_callback(mode: str, mode_index: int, mode_total: int) -> None:
        nonlocal completed_steps
        completed_steps += 1

        should_save = False
        if args.checkpoint_every > 0 and completed_steps % args.checkpoint_every == 0:
            should_save = True
        if mode_index == mode_total:
            should_save = True

        if not should_save:
            return

        checkpoint_payload = build_response_payload(
            dataset,
            run_config,
            synergy_response,
            checkpoint={
                "status": "in_progress",
                "completed_generations": completed_steps,
                "total_generations": total_steps,
                "last_mode": mode,
                "last_mode_progress": f"{mode_index}/{mode_total}",
            },
        )
        write_results(checkpoint_payload, args.output)
        print(
            f"[checkpoint] Saved progress {completed_steps}/{total_steps} "
            f"(mode={mode}, step={mode_index}/{mode_total}) to {args.output}"
        )

    try:
        generate_synergy(
            model,
            dataset,
            image_dir=args.image_dir,
            save_dir=args.generated_image_dir,
            results=synergy_response,
            on_progress=checkpoint_callback,
            max_workers=args.max_workers,
        )
    except Exception as exc:
        failure_payload = build_response_payload(
            dataset,
            run_config,
            synergy_response,
            checkpoint={
                "status": "failed",
                "completed_generations": completed_steps,
                "total_generations": total_steps,
                "error": str(exc),
            },
        )
        write_results(failure_payload, args.output)
        print(
            f"[checkpoint] Saved failure state at {completed_steps}/{total_steps} "
            f"to {args.output}"
        )
        raise

    final_results = build_response_payload(
        dataset,
        run_config,
        synergy_response,
    )
    return final_results


def main() -> None:
    args = parse_args()

    # Make sure the directory exists before generation
    os.makedirs(args.generated_image_dir, exist_ok=True)

    final_results = run_generation(args)
    write_results(final_results, args.output)
    print(f"Saved {len(final_results['response'])} responses to {args.output}")


if __name__ == "__main__":
    main()
