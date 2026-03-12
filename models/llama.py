# TODO: CHECK

"""Llama text-model wrapper implementing the benchmark inference interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LlamaConfig:
	model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
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


class LlamaModel:
	def __init__(self, config: LlamaConfig) -> None:
		self.config = config
		dtype = _resolve_dtype(config.torch_dtype)

		self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
		self.model = AutoModelForCausalLM.from_pretrained(
			config.model_id,
			torch_dtype=dtype,
			device_map=config.device_map,
		)

	def inference(self, text: str | None = None, image: str | Path | None = None) -> str:
		del image  # Text-only model; image input is ignored by design.

		if text is None:
			return "unknown"

		messages = [
			{"role": "system", "content": self.config.system_prompt},
			{"role": "user", "content": text},
		]
		if hasattr(self.tokenizer, "apply_chat_template"):
			prompt = self.tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True,
			)
		else:
			prompt = f"{self.config.system_prompt}\nUser: {text}\nAssistant:"

		model_inputs = self.tokenizer(prompt, return_tensors="pt")
		model_inputs = {k: v.to(self.model.device) for k, v in model_inputs.items()}

		with torch.no_grad():
			output_ids = self.model.generate(
				**model_inputs,
				max_new_tokens=self.config.max_new_tokens,
				do_sample=False,
			)

		generated_ids = output_ids[:, model_inputs["input_ids"].shape[1] :]
		text_out = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
		return text_out.strip()


def load_llama(
	model_id: str = "meta-llama/Llama-3.2-3B-Instruct",
	torch_dtype: str = "auto",
	device_map: str = "auto",
	max_new_tokens: int = 16,
	system_prompt: str | None = None,
) -> LlamaModel:
	config = LlamaConfig(
		model_id=model_id,
		torch_dtype=torch_dtype,
		device_map=device_map,
		max_new_tokens=max_new_tokens,
		system_prompt=system_prompt
		or "You are a sarcasm classifier. Answer with exactly one token: yes or no.",
	)
	return LlamaModel(config)