"""Evaluation pipeline for sarcasm classification with evaluator model backends.

Expected evaluator model interface:
    model.evaluate(text: str | None, image: str | Path | None) -> str
"""

from __future__ import annotations

import os
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from evaluator_models.openai import load_openai_evaluator
from evaluator_models.utils import parse_yes_no_prediction
from dotenv import load_dotenv

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
    if "samples" not in dataset_json: # assume raw
        return dataset_json

    samples = dataset_json.get("samples", {}) # available splits: train, validation, test
    return samples


def init_evaluator_model(
    provider: str = "openai",
) -> Any:
    """Initialize an evaluator model from evaluator_models by provider."""

    provider_name = provider.lower()
    if provider_name == "openai":
        return load_openai_evaluator(
            model_id="gpt-5.4-mini",
            api_key=os.getenv("AZURE_API_KEY"), 
            base_url=os.getenv("AZURE_ENDPOINT"), 
        )

    raise ValueError("Unsupported evaluator provider. Currently supported: openai")


ProgressCallback = Callable[[str, int, int], None]

def _infer(model: Any, text: str, image_path: Path) -> tuple[str, bytes | None]:
    return model.evaluate(text=text, image=image_path)

def evaluate_text(
    model: Any,
    dataset: dict[str, dict[str, Any]],
    results: dict[str, str] | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, str]:
    """Run text-only evaluation by calling model.evaluate(text=..., image=None)."""
    if results is None:
        results = {}

    total = len(dataset)
    pending: list[tuple[str, str, Path]] = []  # (sample_id, text, image_path)
    for index, (sample_id, sample) in enumerate(dataset.items(), start=1):
        if sample_id in results:
            if on_progress is not None:
                on_progress("text_only", index, total)
            continue

        text = sample.get("text")
        if not isinstance(text, str) or not text.strip():
            results[sample_id] = "unknown"
        else:
            pending.append((sample_id, text, None))

    completed_count = total - len(pending)

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_sample = {
            executor.submit(_infer, model, text, None): sample_id
            for sample_id, text, _ in pending
        }
        for future in as_completed(future_to_sample):
            sample_id = future_to_sample[future]
            try:
                raw_output = future.result()
                results[sample_id] = parse_yes_no_prediction(raw_output)
            except Exception as exc:
                print(f"[error] text_only sample {sample_id}: {exc}")
                results[sample_id] = "unknown"
            completed_count += 1

            if on_progress is not None:
                on_progress("text_only", completed_count, total)

    return results


def evaluate_image(
    model: Any,
    dataset: dict[str, dict[str, Any]],
    image_dir: str | Path,
    results: dict[str, str] | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, str]:
    """Run image-only evaluation by calling model.evaluate(text=None, image=...)."""
    if results is None:
        results = {}

    image_root = Path(image_dir)
    total = len(dataset)

    pending: list[tuple[str, str, Path]] = []  # (sample_id, text, image_path)
    for index, (sample_id, sample) in enumerate(dataset.items(), start=1):
        if sample_id in results:
            if on_progress is not None:
                on_progress("image_only", index, total)
            continue

        img_name = sample.get("img_name")
        if not isinstance(img_name, str) or not img_name:
            results[sample_id] = "unknown"
        else:
            image_path = image_root / img_name
            pending.append((sample_id, None, image_path))

    completed_count = total - len(pending)

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_sample = {
            executor.submit(_infer, model, None, image_path): sample_id
            for sample_id, _, image_path in pending
        }
        for future in as_completed(future_to_sample):
            sample_id = future_to_sample[future]
            try:
                raw_output = future.result()
                results[sample_id] = parse_yes_no_prediction(raw_output)
            except Exception as exc:
                print(f"[error] image_only sample {sample_id}: {exc}")

                results[sample_id] = "unknown"
            completed_count += 1

            if on_progress is not None:
                on_progress("image_only", completed_count, total)

    return results


def evaluate_multimodal(
    model: Any,
    dataset: dict[str, dict[str, Any]],
    image_dir: str | Path,
    results: dict[str, str] | None = None,
    on_progress: ProgressCallback | None = None,
) -> dict[str, str]:
    """Run multimodal evaluation by calling model.evaluate(text=..., image=...)."""
    if results is None:
        results = {}

    image_root = Path(image_dir)
    total = len(dataset)

    pending: list[tuple[str, str, Path]] = []  # (sample_id, text, image_path)
    for index, (sample_id, sample) in enumerate(dataset.items(), start=1):
        if sample_id in results:
            if on_progress is not None:
                on_progress("multimodal", index, total)
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

    completed_count = total - len(pending)
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_sample = {
            executor.submit(_infer, model, text, image_path): sample_id
            for sample_id, text, image_path in pending
        }
        for future in as_completed(future_to_sample):
            sample_id = future_to_sample[future]
            try:
                raw_output = future.result()
                results[sample_id] = parse_yes_no_prediction(raw_output)
            except Exception as exc:
                print(f"[error] multimodal sample {sample_id}: {exc}")
                results[sample_id] = "unknown"
            completed_count += 1

            if on_progress is not None:
                on_progress("multimodal", completed_count, total)

    return results


def load_checkpoint(
    output_path: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]] | None:
    """Load partial results from an existing results/checkpoint file.

    Returns (text_results, image_results, multimodal_results) if the file
    exists and contains a "results" mapping, otherwise None. "unknown" and
    missing per-modality entries are filtered out so they get re-evaluated.
    """
    if not output_path.exists():
        return None

    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data.get("results"), dict):
        return None

    text_results: dict[str, str] = {}
    image_results: dict[str, str] = {}
    multimodal_results: dict[str, str] = {}

    for sample_id, result in data.get("results", {}).items():
        if result.get("text_only") not in (None, "unknown"):
            text_results[sample_id] = result["text_only"]
        if result.get("image_only") not in (None, "unknown"):
            image_results[sample_id] = result["image_only"]
        if result.get("multimodal") not in (None, "unknown"):
            multimodal_results[sample_id] = result["multimodal"]

    return text_results, image_results, multimodal_results


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


def build_results_payload(
    dataset: dict[str, dict[str, Any]],
    run_config: dict[str, Any],
    text_results: dict[str, str],
    image_results: dict[str, str],
    multimodal_results: dict[str, str],
    checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_config": run_config,
        "results": aggregate_results(
            dataset,
            text_results,
            image_results,
            multimodal_results,
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
        default=Path("docmsu_10000_split.json"),
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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Save partial results every N sample evaluations (0 disables checkpointing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from an existing checkpoint at --output path if one exists.",
    )
    parser.add_argument(
        "--text-override",
        type=Path,
        default=None,
        help=(
            "Optional path to a generated_text JSON with a top-level 'response' "
            "mapping of sample_id -> {'context': str}. When set, each sample's "
            "'text' field is replaced by the matching 'context' for evaluation."
        ),
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="text_only,image_only,multimodal",
        help=(
            "Comma-separated subset of modes to evaluate. "
            "Options: text_only, image_only, multimodal."
        ),
    )
    parser.add_argument(
        "--image-results-from",
        type=Path,
        default=None,
        help=(
            "Optional path to a prior results JSON (e.g. base-dataset run). "
            "The 'image_only' predictions (yes/no) from that file are loaded "
            "into the final output, so you can skip re-running image_only."
        ),
    )
    parser.add_argument(
        "--text-results-from",
        type=Path,
        default=None,
        help=(
            "Optional path to a prior results JSON. The 'text_only' predictions "
            "(yes/no) from that file are loaded into the final output, so you "
            "can skip re-running text_only (mirror of --image-results-from)."
        ),
    )
    parser.add_argument(
        "--filter-interactions",
        type=str,
        default=None,
        help=(
            "Comma-separated interaction labels (e.g. 'U1' or 'U1,S') to keep. "
            "Requires --interaction-source (or --text-results-from) pointing to "
            "a results JSON with per-sample 'interaction' fields."
        ),
    )
    parser.add_argument(
        "--interaction-source",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to the results JSON used for --filter-interactions. "
            "Defaults to --text-results-from when omitted."
        ),
    )
    parser.add_argument(
        "--merge-from",
        type=Path,
        default=None,
        help=(
            "Optional path to a prior results JSON. After evaluation, any sample "
            "in that file but not in the (possibly filtered) evaluated dataset is "
            "copied into the final output, so you end up with a full-coverage file."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["modify_r_eval"],
        default=None,
        help="Run a named preset workflow instead of the default evaluation.",
    )
    return parser.parse_args()


def load_image_only_results(path: Path) -> dict[str, str]:
    """Extract yes/no image_only predictions from a prior results JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_results: dict[str, str] = {}
    for sample_id, result in data.get("results", {}).items():
        value = result.get("image_only")
        if value in ("yes", "no"):
            image_results[sample_id] = value
    return image_results


def load_text_only_results(path: Path) -> dict[str, str]:
    """Extract yes/no text_only predictions from a prior results JSON."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    text_results: dict[str, str] = {}
    for sample_id, result in data.get("results", {}).items():
        value = result.get("text_only")
        if value in ("yes", "no"):
            text_results[sample_id] = value
    return text_results


def filter_by_interaction(
    dataset: dict[str, dict[str, Any]],
    source_path: Path,
    labels: set[str],
) -> dict[str, dict[str, Any]]:
    """Keep only dataset samples whose 'interaction' in source_path matches labels."""
    with source_path.open("r", encoding="utf-8") as f:
        source = json.load(f)

    interactions = source.get("results", {})
    keep: dict[str, dict[str, Any]] = {}
    for sample_id, sample in dataset.items():
        entry = interactions.get(sample_id)
        if isinstance(entry, dict) and entry.get("interaction") in labels:
            keep[sample_id] = sample
    return keep


def merge_untouched_samples(
    results: dict[str, dict[str, Any]],
    base_path: Path,
    evaluated_sample_ids: set[str],
) -> dict[str, dict[str, Any]]:
    """Copy samples from base_path into results for IDs not in evaluated_sample_ids."""
    with base_path.open("r", encoding="utf-8") as f:
        base = json.load(f)

    merged = dict(results)
    added = 0
    for sample_id, entry in base.get("results", {}).items():
        if sample_id not in evaluated_sample_ids:
            merged[sample_id] = entry
            added += 1
    print(f"[merge] Added {added} untouched samples from {base_path} to output.")
    return merged


def apply_text_override(
    dataset: dict[str, dict[str, Any]],
    override_path: Path,
) -> dict[str, dict[str, Any]]:
    """Replace each sample's 'text' with the matching 'context' from override_path."""
    with override_path.open("r", encoding="utf-8") as f:
        override_json = json.load(f)

    response = override_json.get("response", {})
    if not isinstance(response, dict):
        raise ValueError(f"{override_path} must contain a top-level 'response' mapping.")

    overridden: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for sample_id, sample in dataset.items():
        entry = response.get(sample_id)
        context = None
        if isinstance(entry, dict):
            for key in ("context", "synergy_context"):
                value = entry.get(key)
                if isinstance(value, str):
                    context = value
                    break
        if context is None:
            missing.append(sample_id)
            continue
        overridden[sample_id] = {**sample, "text": context}

    if missing:
        print(
            f"[text-override] {len(missing)} samples missing 'context' in "
            f"{override_path}; those will be dropped."
        )

    return overridden


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    dataset_json = load_dataset(args.dataset)
    dataset = select_dataset_samples(dataset_json, split=args.split)

    if args.text_override is not None:
        dataset = apply_text_override(dataset, args.text_override)

    if args.filter_interactions:
        labels = {lbl.strip() for lbl in args.filter_interactions.split(",") if lbl.strip()}
        source = args.interaction_source or args.text_results_from
        if source is None:
            raise ValueError(
                "--filter-interactions requires --interaction-source "
                "(or --text-results-from) to point at a results JSON."
            )
        before = len(dataset)
        dataset = filter_by_interaction(dataset, source, labels)
        print(
            f"[filter] Kept {len(dataset)}/{before} samples with interaction in "
            f"{sorted(labels)} (source={source})."
        )

    if args.limit and args.limit > 0: # set limits for tests
        limited_items = list(dataset.items())[: args.limit]
        dataset = dict(limited_items)

    valid_modes = {"text_only", "image_only", "multimodal"}
    selected_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(selected_modes) - valid_modes
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}. Valid: {sorted(valid_modes)}")

    model = init_evaluator_model(provider=args.provider)

    run_config: dict[str, Any] = {
        "dataset": str(args.dataset),
        "split": args.split,
        "image_dir": str(args.image_dir),
        "output": str(args.output),
        "provider": args.provider,
        "limit": args.limit,
        "checkpoint_every": args.checkpoint_every,
        "text_override": str(args.text_override) if args.text_override else None,
        "modes": selected_modes,
        "image_results_from": (
            str(args.image_results_from) if args.image_results_from else None
        ),
        "text_results_from": (
            str(args.text_results_from) if args.text_results_from else None
        ),
        "filter_interactions": args.filter_interactions,
        "interaction_source": (
            str(args.interaction_source) if args.interaction_source else None
        ),
    }

    text_results: dict[str, str] = {}
    image_results: dict[str, str] = {}
    multimodal_results: dict[str, str] = {}

    if args.resume:
        prior = load_checkpoint(args.output)
        if prior is not None:
            text_results, image_results, multimodal_results = prior
            print(
                f"[resume] Loaded checkpoint: {len(text_results)} text, "
                f"{len(image_results)} image, {len(multimodal_results)} multimodal results."
            )
        else:
            print("[resume] No resumable checkpoint found at output path; starting fresh.")

    if args.image_results_from is not None:
        reused = load_image_only_results(args.image_results_from)
        for sample_id, value in reused.items():
            image_results.setdefault(sample_id, value)
        print(
            f"[image-reuse] Loaded {len(reused)} image_only predictions from "
            f"{args.image_results_from}; {len(image_results)} total in image_results."
        )

    if args.text_results_from is not None:
        reused = load_text_only_results(args.text_results_from)
        for sample_id, value in reused.items():
            text_results.setdefault(sample_id, value)
        print(
            f"[text-reuse] Loaded {len(reused)} text_only predictions from "
            f"{args.text_results_from}; {len(text_results)} total in text_results."
        )

    total_steps = len(dataset) * len(selected_modes)
    completed_steps = 0

    def checkpoint_callback(mode: str, mode_index: int, mode_total: int) -> None:
        nonlocal completed_steps
        completed_steps += 1

        if mode_index % 10 == 0 or mode_index == 1 or mode_index == mode_total:
            print(
                f"[progress] {mode} {mode_index}/{mode_total} "
                f"(overall {completed_steps}/{total_steps})",
                flush=True,
            )

        should_save = False
        if args.checkpoint_every > 0 and completed_steps % args.checkpoint_every == 0:
            should_save = True
        if mode_index == mode_total:
            should_save = True

        if not should_save:
            return

        checkpoint_payload = build_results_payload(
            dataset,
            run_config,
            text_results,
            image_results,
            multimodal_results,
            checkpoint={
                "status": "in_progress",
                "completed_evaluations": completed_steps,
                "total_evaluations": total_steps,
                "last_mode": mode,
                "last_mode_progress": f"{mode_index}/{mode_total}",
            },
        )
        write_results(checkpoint_payload, args.output)
        print(
            f"[checkpoint] Saved progress {completed_steps}/{total_steps} "
            f"(mode={mode}, step={mode_index}/{mode_total}) to {args.output}",
            flush=True,
        )

    try:
        if "text_only" in selected_modes:
            evaluate_text(
                model,
                dataset,
                results=text_results,
                on_progress=checkpoint_callback,
            )
        if "image_only" in selected_modes:
            evaluate_image(
                model,
                dataset,
                image_dir=args.image_dir,
                results=image_results,
                on_progress=checkpoint_callback,
            )
        if "multimodal" in selected_modes:
            evaluate_multimodal(
                model,
                dataset,
                image_dir=args.image_dir,
                results=multimodal_results,
                on_progress=checkpoint_callback,
            )
    except Exception as exc:
        failure_payload = build_results_payload(
            dataset,
            run_config,
            text_results,
            image_results,
            multimodal_results,
            checkpoint={
                "status": "failed",
                "completed_evaluations": completed_steps,
                "total_evaluations": total_steps,
                "error": str(exc),
            },
        )
        write_results(failure_payload, args.output)
        print(
            f"[checkpoint] Saved failure state at {completed_steps}/{total_steps} "
            f"to {args.output}"
        )
        raise

    final_results = build_results_payload(
        dataset,
        run_config,
        text_results,
        image_results,
        multimodal_results,
    )

    if args.merge_from is not None:
        final_results["results"] = merge_untouched_samples(
            final_results["results"],
            args.merge_from,
            set(dataset.keys()),
        )

    return final_results


def modify_r_eval(
    text_override: Path = Path("generated_text/gpt-5.4_none.json"),
    base_results: Path = Path("results/base_dataset.json"),
    dataset_path: Path = Path("docmsu_10000_split.json"),
    image_dir: Path = Path("img"),
    output: Path = Path("results/R_mod_eval.json"),
    resume: bool = False,
    checkpoint_every: int = 250,
) -> dict[str, Any]:
    """Re-evaluate the 'R' interaction samples from base_results with overridden text.

    - Text is sourced from text_override (synergy_context/context field).
    - Images come from image_dir (base /img by default).
    - Only text_only and multimodal modes are run; image_only is reused from base.
    - Non-R samples are copied verbatim from base_results so the output covers
      all samples in the base file.
    """
    args = argparse.Namespace(
        dataset=dataset_path,
        split="all",
        image_dir=image_dir,
        output=output,
        provider="openai",
        limit=0,
        checkpoint_every=checkpoint_every,
        resume=resume,
        text_override=text_override,
        modes="text_only,multimodal",
        image_results_from=base_results,
        text_results_from=None,
        filter_interactions="R",
        interaction_source=base_results,
        merge_from=base_results,
        preset="modify_r_eval",
    )
    return run_evaluation(args)


def main() -> None:
    args = parse_args()
    if args.preset == "modify_r_eval":
        # If the user didn't pass --output, fall back to the preset's default.
        output = args.output
        if output == Path("results/evaluator_inference.json"):
            output = Path("results/R_mod_eval.json")
        final_results = modify_r_eval(
            dataset_path=args.dataset,
            image_dir=args.image_dir,
            output=output,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
        )
        write_results(final_results, output)
        print(f"Saved {len(final_results['results'])} results to {output}")
        return

    final_results = run_evaluation(args)
    write_results(final_results, args.output)
    print(f"Saved {len(final_results['results'])} results to {args.output}")


if __name__ == "__main__":
    main()
