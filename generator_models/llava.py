# TODO: CHECK

"""LLaVA wrapper implementing the benchmark inference interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


@dataclass
class LlavaConfig:
	model_id: str = "llava-hf/llava-1.5-7b-hf"
	torch_dtype: str = "auto"
	device_map: str = "auto"
	max_new_tokens: int = 16
	system_prompt: str = (
		"You are a sarcasm classifier. Answer with exactly one token: yes or no."
	)


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


class LlavaModel:
	def __init__(self, config: LlavaConfig) -> None:
		self.config = config
		dtype = _resolve_dtype(config.torch_dtype)

		self.processor = AutoProcessor.from_pretrained(config.model_id)
		self.model = LlavaForConditionalGeneration.from_pretrained(
			config.model_id,
			torch_dtype=dtype,
			device_map=config.device_map,
		)

	def inference(self, text: str | None = None, image: str | Path | None = None) -> str:
		if text is None and image is None:
			raise ValueError("Either text or image must be provided.")

		if image is None:
			# LLaVA is vision-language centric; when no image is provided,
			# we return unknown to avoid implicit text-only behavior.
			return "unknown"

		pil_image = Image.open(Path(image)).convert("RGB")
		user_text = (
			text
			if text is not None
			else "Classify this image for sarcasm and answer with exactly: yes or no."
		)

		messages = [
			{"role": "system", "content": self.config.system_prompt},
			{
				"role": "user",
				"content": [
					{"type": "image"},
					{"type": "text", "text": user_text},
				],
			},
		]

		prompt = self.processor.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
		)

		inputs = self.processor(
			text=prompt,
			images=pil_image,
			return_tensors="pt",
		)
		inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

		with torch.no_grad():
			output_ids = self.model.generate(
				**inputs,
				max_new_tokens=self.config.max_new_tokens,
				do_sample=False,
			)

		generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
		out_text = self.processor.batch_decode(
			generated_ids,
			skip_special_tokens=True,
			clean_up_tokenization_spaces=True,
		)[0]
		return out_text.strip()


def load_llava(
	model_id: str = "llava-hf/llava-1.5-7b-hf",
	torch_dtype: str = "auto",
	device_map: str = "auto",
	max_new_tokens: int = 16,
	system_prompt: str | None = None,
) -> LlavaModel:
	config = LlavaConfig(
		model_id=model_id,
		torch_dtype=torch_dtype,
		device_map=device_map,
		max_new_tokens=max_new_tokens,
		system_prompt=system_prompt
		or "You are a sarcasm classifier. Answer with exactly one token: yes or no.",
	)
	return LlavaModel(config)