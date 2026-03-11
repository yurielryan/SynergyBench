"""Qwen 3 VL model wrapper.

This module exposes a small, importable interface for loading a Qwen 3 VL model
and running inference with text-only, image-only, or multimodal inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


# Canonical size aliases for Qwen3-VL checkpoints.
# If a size is not in this map, we try to construct "Qwen/Qwen3-VL-<SIZE>-Instruct".
QWEN3_VL_SIZE_ALIASES: dict[str, str] = {
    "4b": "4B",
    "8b": "8B",
    "30b": "30B-A3B",
}


@dataclass
class QwenVLConfig:
    """Configuration for Qwen VL model loading and generation."""

    size: str = "3b"
    model_id: str | None = None
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 4
    system_prompt: str = (
        "You are a classifier for sarcasm. " "Answer with exactly one token: yes or no."
    )


def _resolve_model_id(size: str) -> str:
    size_key = size.strip().lower()
    if not size_key:
        raise ValueError("Model size cannot be empty.")

    if size_key in QWEN3_VL_SIZE_ALIASES:
        canonical = QWEN3_VL_SIZE_ALIASES[size_key]
        return f"Qwen/Qwen3-VL-{canonical}-Instruct"

    # Accept direct forms such as "8B", "30B-A3B", "4B-thinking" and normalize casing.
    cleaned = size.strip().upper()
    return f"Qwen/Qwen3-VL-{cleaned}-Instruct"


def _resolve_dtype(dtype_name: str) -> Any:
    name = dtype_name.strip().lower()
    if name == "auto":
        return "auto"
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(
        "torch_dtype must be one of: auto, float16/fp16, bfloat16/bf16, float32/fp32"
    )


class QwenVLModel:
    """Thin wrapper that normalizes inference API for evaluation scripts."""

    def __init__(self, config: QwenVLConfig) -> None:
        self.config = config
        model_id = config.model_id or _resolve_model_id(config.size)
        dtype = _resolve_dtype(config.torch_dtype)

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=config.device_map,
            trust_remote_code=True,
        )

    def inference(
        self, text: str | None = None, image: str | Path | None = None
    ) -> str:
        """Run a single inference call.

        Either text or image must be provided.
        Returns generated text without chat-template prefixes.
        """
        if text is None and image is None:
            raise ValueError("Either text or image must be provided.")

        user_content: list[dict[str, Any]] = []
        if image is not None:
            image_path = Path(image)
            pil_img = Image.open(image_path).convert("RGB")
            user_content.append({"type": "image", "image": pil_img})
        if text is not None:
            user_content.append({"type": "text", "text": text})

        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.processor(
            text=[prompt],
            images=[user_content[0]["image"]] if image is not None else None,
            return_tensors="pt",
        )
        model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.config.max_new_tokens,
            )

        generated_ids = output_ids[:, model_inputs["input_ids"].shape[1] :]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        return output_text.strip()


def load_qwen_vl(
    size: str = "8b",
    model_id: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    max_new_tokens: int = 64,
    system_prompt: str | None = None,
) -> QwenVLModel:
    """Factory function for easy import in evaluate.py."""
    config = QwenVLConfig(
        size=size,
        model_id=model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt
        or (
            "You are a classifier for sarcasm. "
            "Answer with exactly one token: yes or no."
        ),
    )
    return QwenVLModel(config)
