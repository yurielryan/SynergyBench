"""Evaluation pipeline for sarcasm classification with VLM backends.

Expected model interface:
        model.inference(text: str | None, image: str | Path | None) -> str
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import yaml

from models.qwen import load_qwen_vl


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must contain a top-level mapping.")
    return cfg


def load_dataset(dataset_path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            "Dataset JSON must be a top-level mapping of sample_id to sample."
        )
    return data


def select_dataset_samples(
    dataset_json: dict[str, Any],
    split: str = "all",
) -> dict[str, dict[str, Any]]:
    """Normalize raw or curated dataset JSON into sample_id->sample mapping.

    Supports two JSON layouts:
    1) raw: {sample_id: sample}
    2) curated: {"meta": ..., "splits": {"train": {...}, "validation": {...}, "test": {...}}}
    """
    if "splits" not in dataset_json:
        return dataset_json  # raw layout

    splits = dataset_json.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError("If present, 'splits' must be a mapping.")

    split_name = split.lower()
    if split_name in {
        "train",
        "validation",
        "test",
    }:  # take samples only from train/val/test
        selected = splits.get(split_name, {})
        if not isinstance(selected, dict):
            raise ValueError(f"Split '{split_name}' must be a mapping.")
        return selected

    if split_name == "all":
        merged: dict[str, dict[str, Any]] = {}  # include all samples
        for key in ("train", "validation", "test"):
            split_map = splits.get(key, {})
            if isinstance(split_map, dict):
                merged.update(split_map)
        if merged:
            return merged
        # Fallback for non-standard split names.
        for split_map in splits.values():
            if isinstance(split_map, dict):
                merged.update(split_map)
        return merged

    raise ValueError("data.split must be one of: all, train, validation, test")


def init_model(config: dict[str, Any]) -> Any:
    model_cfg = config.get("model", {})
    provider = str(model_cfg.get("provider", "qwen")).lower()

    # Generic loader hook for arbitrary VLM wrappers.
    # Example: "my_models.llava:load_model"
    custom_loader = model_cfg.get("loader")
    if custom_loader:
        if ":" not in custom_loader:
            raise ValueError(
                "model.loader must use format '<module_path>:<function_name>'"
            )
        module_path, func_name = custom_loader.split(":", 1)
        module = importlib.import_module(module_path)
        loader_fn = getattr(module, func_name)
        kwargs = model_cfg.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise ValueError(
                "model.kwargs must be a mapping when model.loader is used."
            )
        return loader_fn(**kwargs)

    if provider == "qwen":
        return load_qwen_vl(
            size=str(model_cfg.get("size", "8b")),
            model_id=model_cfg.get("model_id"),
            torch_dtype=str(model_cfg.get("torch_dtype", "auto")),
            device_map=str(model_cfg.get("device_map", "auto")),
            max_new_tokens=int(model_cfg.get("max_new_tokens", 64)),
            system_prompt=model_cfg.get("system_prompt"),
        )

    raise ValueError("Unsupported model provider. Currently supported: qwen")


def parse_yes_no_prediction(raw_output: str) -> str:
    output = raw_output.strip().lower()
    if output.startswith("yes"):
        return "yes"
    if output.startswith("no"):
        return "no"
    if "yes" in output and "no" not in output:
        return "yes"
    if "no" in output and "yes" not in output:
        return "no"
    return "unknown"


def evaluate_text(model: Any, dataset: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Run text-only evaluation by calling model.inference(text=..., image=None)."""
    results: dict[str, str] = {}
    for sample_id, sample in dataset.items():
        text = sample.get("text")
        if text is None:
            results[sample_id] = "unknown"
            continue
        raw_output = model.inference(text=text, image=None)
        results[sample_id] = parse_yes_no_prediction(raw_output)
    return results  # dictionary of sample_id: answer


def evaluate_image(
    model: Any,
    dataset: dict[str, dict[str, Any]],
    image_dir: str | Path,
) -> dict[str, str]:
    """Run image-only evaluation by calling model.inference(text=None, image=...)."""
    results: dict[str, str] = {}
    image_root = Path(image_dir)

    for sample_id, sample in dataset.items():
        img_name = sample.get("img_name")
        if not img_name:
            results[sample_id] = "unknown"
            continue

        image_path = image_root / img_name
        if not image_path.exists():
            results[sample_id] = "unknown"
            continue

        raw_output = model.inference(text=None, image=image_path)
        results[sample_id] = parse_yes_no_prediction(raw_output)
    return results


def evaluate_multimodal(
    model: Any,
    dataset: dict[str, dict[str, Any]],
    image_dir: str | Path,
) -> dict[str, str]:
    """Run multimodal evaluation by calling model.inference(text=..., image=...)."""
    results: dict[str, str] = {}
    image_root = Path(image_dir)

    for sample_id, sample in dataset.items():
        text = sample.get("text")
        img_name = sample.get("img_name")
        # if text is None or not img_name:
        #     results[sample_id] = "unknown"
        #     continue

        image_path = image_root / img_name
        # if not image_path.exists():
        #     results[sample_id] = "unknown"
        #     continue

        raw_output = model.inference(text=text, image=image_path)
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
            # text only, image only, and multimodal inference
            "text_only": text_results.get(sample_id),
            "image_only": image_results.get(sample_id),
            "multimodal": multimodal_results.get(sample_id),
        }
    return output


def write_results(results: dict[str, Any], output_path: str | Path) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_evaluation(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    data_cfg = config.get("data", {})

    dataset_json = load_dataset(data_cfg.get("dataset_path", "docmsu_all.json"))
    dataset = select_dataset_samples(
        dataset_json, split=str(data_cfg.get("split", "all"))
    )
    image_dir = data_cfg.get("image_dir", "img")

    model = init_model(config)  # initialize vlm for inference

    # inference: text only, image only, and both
    text_results = evaluate_text(model, dataset)
    image_results = evaluate_image(model, dataset, image_dir=image_dir)
    multimodal_results = evaluate_multimodal(model, dataset, image_dir=image_dir)

    final_results = {
        "config": config,
        "results": aggregate_results(
            dataset,
            text_results,
            image_results,
            multimodal_results,
        ),
    }
    return final_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sarcasm inference evaluation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to evaluation YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_path = config.get("output", {}).get("output_path", "results/inference.json")

    results = run_evaluation(args.config)
    write_results(results, output_path)
    print(f"Wrote inference results to {output_path}")


if __name__ == "__main__":
    main()
